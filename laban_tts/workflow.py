"""
Audiobook TTS Toolchain (Normalize → Cue → Synthesize)

This module implements a deterministic, file-first toolchain for audiobook
production. Stages are exposed as explicit CLI commands:

parts/ → normalize/ → cues/ → audio/ (→ finalize/)

Each stage validates its preconditions, produces typed artifacts (Pydantic v2
models), and fails fast on policy violations. Humans can inspect or edit
artifacts between stages and rerun stages with ``--force`` when they need to
regenerate outputs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import fire
import soundfile as sf
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage

# NOTE: Lazy imports are illegal.
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_xml import BaseXmlModel, attr, element, wrapped
from pydub import AudioSegment

from laban_tts.cues import CUE_PRIMER, CUE_PROMPT, CuedScript
from laban_tts.normalize import (
    NormalizedPart,
    TextType,
    load_normalized_parts,
    load_parts,
    normalize_parts,
    partition_text,
)


# ─────────────────────────────────────────────────────────────────────────────
# Prompts (concise). We rely on the Pydantic field descriptions for formatting.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic helper models for on-disk artifacts
# ─────────────────────────────────────────────────────────────────────────────


class Manifest(BaseModel):
    text_name: str
    workspace: Path
    kind: Literal["book"] = "book"


class CueEntry(BaseModel):
    """Cue artifacts ready for synthesis."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    text_name: str
    part: int
    script: CuedScript
    xml_path: Path


class ChunkEntry(BaseModel):
    duration: float
    position: float
    phrases: List[str]


class CueRequest(BaseXmlModel, tag="cue-request", skip_empty=True):
    """Metadata payload for cue generation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text_name: str = attr(name="text-name")
    part: int = attr(ge=1)
    category: TextType = attr()
    speakers: List[str] = wrapped(
        "speakers",
        element(tag="speaker", default_factory=list),
    )
    previous_script: Optional[CuedScript] = element(default=None)


# ─────────────────────────────────────────────────────────────────────────────
# Partitioning helpers
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Toolchain implementation
# ─────────────────────────────────────────────────────────────────────────────

WORKSPACE_DIR = Path(os.environ.get("WORKSPACE_DIR", "/data/workspace"))


class Toolchain:
    """Text-to-speech pipeline exposed as sequential CLI stages."""

    def __init__(
        self,
        debug: bool = False,
        in_dir: Path | str = WORKSPACE_DIR / "in",
        voices_dir: Path | str = WORKSPACE_DIR / "voices",
        workspace_dir: Path | str = WORKSPACE_DIR / "tts",
        force: bool = False,
        model_name: str = "gpt-5-mini",
        language_id: Optional[str] = None,
    ) -> None:
        """Initialize directories, execution flags, and the backing LLM client.

        Args:
            debug: Enable verbose logging for manual runs.
            in_dir: Root directory holding input documents.
            voices_dir: Directory containing reference voice WAV files.
            workspace_dir: Base output directory for derived artifacts.
            force: Whether to overwrite existing stage directories.
            llm: Optional preconfigured language model client for cueing.
            model_name: OpenAI model identifier used when `llm` is omitted.
            language_id: Optional locale hint propagated to downstream stages.
        """
        self.debug = debug
        self.force = force
        self.language_id = language_id
        self.in_dir = Path(in_dir)
        self.in_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

    # —————————————————— Utilities ——————————————————

    def voices_from_spec(self, voice_specs: Sequence[str]) -> Dict[str, Path]:
        """Resolve `speaker:voice` specs to WAV files under `voices_dir`.

        Args:
            voice_specs: Iterable of `speaker:voice_name` entries that may
                include a `default` key.

        Returns:
            Mapping from normalized speaker identifiers to existing WAV paths.
        """
        voices: Dict[str, Path] = {}
        for spec in voice_specs:
            if not spec:
                continue
            parts = spec.split(":", maxsplit=1)
            assert (
                len(parts) == 2
            ), f"Invalid voice format: {spec}. Expected speaker:voice_name."
            speaker = parts[0].strip().lower()
            voices[speaker] = self.voice_file(parts[1].strip())
        return voices

    def voice_file(self, voice_name: str) -> Path:
        """Return the absolute WAV path for a named reference voice sample."""
        voice_file = self.voices_dir / f"{voice_name}.wav"
        assert voice_file.exists(), f"Voice file {voice_file} does not exist."
        return voice_file

    @staticmethod
    def _choose_voice(
        speaker: str, voices: Dict[str, Path]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Return (voice_path, voice_name) using speaker match or default."""
        normalized = speaker.lower()
        if normalized in voices:
            voice_path = voices[normalized]
            return str(voice_path), voice_path.stem
        if "default" in voices:
            voice_path = voices["default"]
            return str(voice_path), voice_path.stem
        return None, None

    def choose_book(self) -> Path:
        """Use `sk` to choose an EPUB/PDF from the input directory.

        Returns:
            Absolute path to the selected source file under `in_dir`.

        Raises:
            ValueError: If the input directory does not contain any book files.
        """
        books = list(self.in_dir.rglob("*.epub")) + list(self.in_dir.rglob("*.pdf"))
        if not books:
            raise ValueError("No book files found in the input directory.")
        choices = [str(path.relative_to(self.in_dir)) for path in books]
        result = subprocess.run(
            ["sk"], input="\n".join(choices), text=True, capture_output=True, check=True
        )
        file_path = result.stdout.strip()
        return self.in_dir / file_path

    def _prepare_output_dir(self, path: Path, stage: str) -> None:
        """Ensure an output directory is writable, honoring the `--force` policy."""
        if path.exists():
            if not path.is_dir():
                raise FileExistsError(
                    f"Stage {stage} encountered a non-directory path: {path}"
                )
            if self.force:
                logger.warning(
                    "Overwriting existing directory for stage {stage}: {path}",
                    stage=stage,
                    path=path,
                )
                shutil.rmtree(path)
                path.mkdir(parents=True, exist_ok=True)
            else:
                # Directory already exists; reuse for incremental processing.
                return
        else:
            path.mkdir(parents=True, exist_ok=True)

    def _resolve_workspace(self, work_dir: Path | str) -> Path:
        """Resolve a workspace path, accepting relative names under `workspace_dir`."""
        work_path = Path(work_dir)
        if not work_path.is_absolute():
            work_path = self.workspace_dir / work_path
        if not work_path.exists():
            raise FileNotFoundError(
                f"Workspace {work_path} does not exist. Run `book` or `text` first."
            )
        return work_path

    # —————————————————— Stage helpers ——————————————————

    @staticmethod
    def _parse_part_filename(stem: str, suffix: str) -> Tuple[str, int]:
        """Extract (text_name, part) from a stage artifact stem."""
        if not stem.endswith(suffix):
            raise ValueError(f"Stem {stem} does not end with expected suffix {suffix}.")
        base = stem[: -len(suffix)]
        if base.endswith("-"):
            base = base[:-1]
        if "-part" not in base:
            raise ValueError(f"Stem {stem} is missing -partNNN.")
        text_name, part_str = base.rsplit("-part", 1)
        return text_name, int(part_str)

    def _iter_cues(self, cue_dir: Path) -> List[CueEntry]:
        entries: List[CueEntry] = []
        for xml_path in sorted(cue_dir.glob("*-cues.xml")):
            text_name, part = self._parse_part_filename(xml_path.stem, "-cues")
            script = CuedScript.from_xml(xml_path.read_text())
            entries.append(
                CueEntry(
                    text_name=text_name,
                    part=part,
                    script=script,
                    xml_path=xml_path,
                )
            )
        entries.sort(key=lambda entry: (entry.text_name, entry.part))
        return entries

    def normalize(self, workspace: Path) -> None:
        """Generate normalized JSON artifacts from `parts/` inputs.

        Args:
            workspace: Target workspace containing a populated `parts` directory.

        Raises:
            FileNotFoundError: If the `parts` directory is missing.
            ValueError: If no part files are available to normalize.
        """
        parts_dir = workspace / "parts"
        if not parts_dir.exists():
            raise FileNotFoundError(f"Parts directory missing: {parts_dir}")

        logger.info("normalize.start work_dir={work_dir}", work_dir=workspace)
        normalize_dir = workspace / "normalize"
        self._prepare_output_dir(normalize_dir, stage="normalize")

        parts = load_parts(parts_dir)
        if not parts:
            raise ValueError("No parts found to normalize.")

        llm = ChatOpenAI(
            model=self.model_name,
            temperature=0,
            max_retries=3,
        )
        normalized_entries = normalize_parts(
            parts, normalize_dir, llm, force=self.force
        )
        logger.info(
            "normalize.done work_dir={work_dir} parts={count}",
            work_dir=workspace,
            count=len(normalized_entries),
        )

    # —————————————————— Stage commands ——————————————————

    def cue(self, work_dir: Path | str) -> None:
        """Derive structured cue scripts (`cues/`) from normalized parts.

        Args:
            work_dir: Absolute or relative workspace identifier with
                `normalize` outputs.

        Raises:
            FileNotFoundError: If the workspace or normalized artifacts are
                missing.
            ValueError: If no normalized entries were produced by Stage A.
        """
        workspace = self._resolve_workspace(work_dir)
        normalize_dir = workspace / "normalize"
        if not normalize_dir.exists():
            raise FileNotFoundError(f"Normalized directory missing: {normalize_dir}")
        logger.info("cue.start work_dir={work_dir}", work_dir=workspace)
        cue_dir = workspace / "cues"
        self._prepare_output_dir(cue_dir, stage="cue")

        normalized_entries = load_normalized_parts(normalize_dir)
        if not normalized_entries:
            raise ValueError("No normalized outputs found; run `normalize` first.")

        llm = ChatOpenAI(
            model=self.model_name,
            temperature=0,
            max_retries=3,
        )
        previous_script: Optional[CuedScript] = None
        for entry in normalized_entries:
            stem = f"{entry.text_name}-part{entry.part:03d}-cues"
            xml_path = cue_dir / f"{stem}.xml"
            if xml_path.exists() and not self.force:
                logger.info(
                    "cue.skip_existing text_name={name} part={part}",
                    name=entry.text_name,
                    part=entry.part,
                )
                script = CuedScript.from_xml(xml_path.read_text())
                previous_script = script
                continue
            metadata = CueRequest(
                text_name=entry.text_name,
                part=entry.part,
                category=entry.category,
                speakers=entry.speaker_names(),
                previous_script=previous_script,
            )
            metadata_block = metadata.to_xml(
                encoding="unicode", pretty_print=True, skip_empty=True
            )
            assert isinstance(metadata_block, str)
            messages = [
                SystemMessage(content=CUE_PRIMER),
                SystemMessage(content=CUE_PROMPT),
                HumanMessage(
                    content=(
                        "== CONTEXT ==\n"
                        f"{metadata_block}\n\n"
                        "== CLEANED TEXT ==\n"
                        f"{entry.cleaned_text()}"
                    )
                ),
            ]
            logger.info(
                "cue.part text_name={name} part={part}",
                name=entry.text_name,
                part=entry.part,
            )
            callback = UsageMetadataCallbackHandler()
            retries = 3
            while True:
                try:
                    logger.debug(
                        "Invoking LLM for cue generation messages={} retries={}",
                        len(messages),
                        retries,
                    )
                    res = llm.with_structured_output(
                        CuedScript, include_raw=True
                    ).invoke(messages, config=RunnableConfig(callbacks=[callback]))
                    logger.debug("LLM responded usage={}", callback.usage_metadata)
                    script: CuedScript = res["parsed"]
                    # Ensure speaker registry covers all chunk speakers even if the LLM omits narrator/default entries.
                    chunk_speakers = [chunk.speaker for chunk in script.chunks]
                    merged_speakers = list(
                        dict.fromkeys([*script.speakers, *chunk_speakers])
                    )
                    if merged_speakers != script.speakers:
                        script = script.model_copy(update={"speakers": merged_speakers})
                        logger.debug(
                            "cue.speakers merged={speakers}", speakers=merged_speakers
                        )
                    xml_payload = script.to_xml(
                        encoding="unicode", pretty_print=True, skip_empty=True
                    )
                    break
                except Exception as ve:
                    logger.error(
                        "cue.validation_failed text_name={name} part={part}: {errors}",
                        name=entry.text_name,
                        part=entry.part,
                        errors=ve,
                    )
                    retries -= 1
                    if retries <= 0:
                        raise ve
            assert isinstance(xml_payload, str)
            xml_path.write_text(xml_payload)

            norm_len = len(entry.cleaned_text())
            chunk_len = sum(len(chunk.text) for chunk in script.chunks)
            coverage = chunk_len / max(1, norm_len)
            logger.debug(
                "cue.coverage norm_chars={norm} chunk_chars={chunk} ratio={ratio:.3f}",
                norm=norm_len,
                chunk=chunk_len,
                ratio=coverage,
            )
            if coverage < 0.5:
                logger.warning(
                    "cue.coverage_low ratio={coverage:.3f} norm_chars={norm} chunk_chars={chunks}",
                    coverage=coverage,
                    norm=norm_len,
                    chunks=chunk_len,
                )
            previous_script = script

        logger.info(
            "cue.done work_dir={work_dir} scripts={count}",
            work_dir=workspace,
            count=len(normalized_entries),
        )

    def synthesize(
        self,
        work_dir: Path | str,
        voice_files: str = "default:enoch",
        prepare_conditionals: bool = False,
    ) -> None:
        """Render WAV audio for every cue chunk using chatterbox-tts.

        Args:
            work_dir: Workspace name or path containing cue XML artifacts.
            voice_files: Comma-separated `speaker:voice` hints used to resolve
                reference voices.
            prepare_conditionals: Whether to prime the TTS model with the
                default voice before chunk synthesis.

        Raises:
            FileNotFoundError: If the cue directory is missing.
            ValueError: If no cue scripts are available.
        """

        # NOTE: Exceptionally importing inside the local scope given the heavy dependencies.
        import torch
        import torchaudio  # type: ignore[import]
        from chatterbox.tts import ChatterboxTTS  # type: ignore[import]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_model = ChatterboxTTS.from_pretrained(device=device)

        backends = getattr(torchaudio, "list_audio_backends", lambda: [])()
        logger.info(
            "torchaudio.version={version} backends={backends}",
            version=torchaudio.__version__,
            backends=backends,
        )

        workspace = self._resolve_workspace(work_dir)
        cue_dir = workspace / "cues"
        if not cue_dir.exists():
            raise FileNotFoundError(f"Cue directory missing: {cue_dir}")
        logger.info("synthesize.start work_dir={work_dir}", work_dir=workspace)
        audio_dir = workspace / "audio"
        self._prepare_output_dir(audio_dir, stage="synthesize")

        cue_entries = self._iter_cues(cue_dir)
        if not cue_entries:
            raise ValueError("No cue scripts found; run `cue` first.")

        voice_specs = [spec.strip() for spec in voice_files.split(",") if spec.strip()]
        voices = self.voices_from_spec(voice_specs)
        if voices:
            logger.info("synthesize.voices count={count}", count=len(voices))

        if prepare_conditionals and hasattr(tts_model, "prepare_conditionals"):
            default_voice = voices.get("default")
            chosen = default_voice or (next(iter(voices.values())) if voices else None)
            if chosen is not None:
                try:
                    tts_model.prepare_conditionals(str(chosen))  # type: ignore[attr-defined]
                    logger.debug("synthesize.prepared voice={voice}", voice=chosen)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "synthesize.prepare_conditionals_failed voice={voice} error={error}",
                        voice=chosen,
                        error=exc,
                    )

        total_saved = 0
        position = 0.0
        for entry in cue_entries:
            logger.info(
                "synthesize.script text_name={name} part={part} chunks={chunks}",
                name=entry.text_name,
                part=entry.part,
                chunks=len(entry.script.chunks),
            )
            script = entry.script
            for chunk in script.chunks:
                voice_path, voice_name = self._choose_voice(chunk.speaker, voices)
                audio_prompt_path = chunk.audio_prompt_path or voice_path

                logger.info(
                    "Chunk {idx} / {count}: size={size} speaker={speaker} pre_ms={pre}ms post_ms={post}ms voice={voice}",
                    idx=chunk.idx,
                    count=len(script.chunks),
                    size=len(chunk.text),
                    speaker=chunk.speaker,
                    pre=chunk.pre_pause_ms,
                    post=chunk.post_pause_ms,
                    voice=voice_name,
                )
                phrases = chunk.split_text()
                out_file = (
                    audio_dir
                    / f"{entry.text_name}_p{entry.part:03d}_{chunk.idx:04d}_{chunk.speaker}.wav"
                )
                meta_path = out_file.with_suffix(".json")
                if out_file.exists() and not self.force:
                    logger.info(
                        "synthesize.skip_existing text_name={name} part={part} chunk={chunk}",
                        name=entry.text_name,
                        part=entry.part,
                        chunk=chunk.idx,
                    )
                    if meta_path.exists():
                        try:
                            existing_meta = ChunkEntry.model_validate_json(
                                meta_path.read_text()
                            )
                            position = max(position, existing_meta.position)
                        except (ValidationError, ValueError):
                            logger.warning(
                                "synthesize.meta_parse_failed path={path}",
                                path=meta_path,
                            )
                    continue
                base_audio = AudioSegment.silent(duration=chunk.pre_pause_ms)
                sample_rate = int(tts_model.sr)

                spoken_phrases: List[str] = []
                for idx, phrase in enumerate(phrases, start=1):
                    clean_phrase = phrase.strip()
                    if not clean_phrase:
                        continue
                    logger.debug("Phrase {} / {}: {}", idx, len(phrases), clean_phrase)
                    params = chunk.engine_params()
                    wav = tts_model.generate(
                        text=clean_phrase,
                        audio_prompt_path=audio_prompt_path,
                        exaggeration=params.exaggeration,
                        cfg_weight=params.cfg_weight,
                        temperature=params.temperature,
                        repetition_penalty=params.repetition_penalty,
                        min_p=params.min_p,
                        top_p=params.top_p,
                    )
                    waveform = wav.detach().cpu()
                    if waveform.ndim == 1:
                        samples = waveform.unsqueeze(1).contiguous().numpy()
                    elif waveform.ndim == 2:
                        samples = waveform.transpose(0, 1).contiguous().numpy()
                    else:  # pragma: no cover - defensive guard
                        raise ValueError(
                            f"Unexpected waveform shape: {tuple(waveform.shape)}"
                        )
                    buf = BytesIO()
                    # TODO: Why is it necessary to do this and not just use pydub directly?
                    sf.write(buf, samples, sample_rate, format="WAV")
                    buf.seek(0)
                    base_audio += AudioSegment.from_file(buf, format="wav")
                    buf.close()
                    spoken_phrases.append(clean_phrase)

                if chunk.post_pause_ms:
                    base_audio += AudioSegment.silent(duration=chunk.post_pause_ms)

                base_audio.export(out_file, format="wav")
                position += base_audio.duration_seconds
                logger.debug(
                    "Saved '{}' duration={:.1f}s position={:.1f}s",
                    out_file,
                    base_audio.duration_seconds,
                    position,
                )
                meta = ChunkEntry.model_validate(
                    {
                        "duration": base_audio.duration_seconds,
                        "position": position,
                        "phrases": spoken_phrases or phrases,
                    }
                )
                out_file.with_suffix(".json").write_text(
                    meta.model_dump_json(
                        indent=2, exclude_none=True, exclude_unset=True
                    )
                )
                total_saved += 1

        logger.info(
            "synthesize.done work_dir={work_dir} files={count}",
            work_dir=workspace,
            count=total_saved,
        )

    def finalize(self, work_dir: Path | str) -> None:
        """Log the inventory of synthesized audio as a placeholder final stage.

        Args:
            work_dir: Workspace name or path with a populated `audio` directory.

        Raises:
            FileNotFoundError: If synthesized audio assets are not present.
        """
        workspace = self._resolve_workspace(work_dir)
        audio_dir = workspace / "audio"
        if not audio_dir.exists():
            raise FileNotFoundError(f"Audio directory missing: {audio_dir}")
        wav_files = list(audio_dir.glob("*.wav"))
        logger.info("finalize.start work_dir={work_dir}", work_dir=workspace)
        logger.debug(
            "finalize.files wav_files={files}",
            files=[file.name for file in wav_files],
        )
        logger.info(
            "finalize.done work_dir={work_dir} wav_count={count}",
            work_dir=workspace,
            count=len(wav_files),
        )

    def info(
        self,
        work_dir: Path | str = "",
    ) -> None:
        workspace = self._resolve_workspace(work_dir)
        parts_dir = workspace / "parts"
        parts = load_parts(parts_dir)
        normalize_dir = workspace / "normalize"
        results: List[NormalizedPart] = []
        for part in parts:
            xml_path = (
                normalize_dir / f"{part.text_name}-part{part.part:03d}-normalized.xml"
            )
            if xml_path.exists():
                logger.debug(
                    "Reusing normalized part {}",
                    xml_path,
                )
                result = NormalizedPart.from_xml(xml_path.read_text())
                results.append(result)
        logger.info(
            "info.work_dir={work_dir} parts={part_count} normalized={norm_count}",
            work_dir=workspace,
            part_count=len(parts),
            norm_count=len(results),
        )

    def run(
        self,
        text_file: Path | str = "",
        auto: bool = False,
        voice_files: str = "default:enoch",
        prepare_conditionals: bool = False,
    ) -> Path:
        """Create a workspace from a source document and optionally run stages.

        Args:
            text_file: Source document path; prompts for selection when empty.
            auto: Whether to execute normalize → cue → synthesize → finalize.
            voice_files: Voice selection string forwarded to `synthesize`.
            prepare_conditionals: Pass-through flag for conditional preparation.

        Returns:
            Workspace directory created for the supplied text file.
        """
        if not text_file:
            text_file = self.choose_book()
        else:
            text_file = Path(text_file)

        assert text_file.exists(), f"Book file {text_file} does not exist."

        work_dir = self.workspace_dir / text_file.stem
        work_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = work_dir / "manifest.json"
        if not manifest_file.exists():
            manifest = Manifest(
                text_name=text_file.stem, workspace=work_dir, kind="book"
            )
            manifest_file.write_text(manifest.model_dump_json(indent=2))

        parts_dir = work_dir / "parts"
        self._prepare_output_dir(parts_dir, stage="parts")
        partition_text(text_file, parts_dir)

        if auto:
            logger.info("auto.pipeline.start work_dir={work_dir}", work_dir=work_dir)
            self.normalize(work_dir)
            self.cue(work_dir)
            self.synthesize(
                work_dir,
                voice_files=voice_files,
                prepare_conditionals=prepare_conditionals,
            )
            self.finalize(work_dir)
            logger.info("auto.pipeline.done work_dir={work_dir}", work_dir=work_dir)
        return work_dir


if __name__ == "__main__":
    fire.Fire(Toolchain)
