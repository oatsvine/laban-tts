from __future__ import annotations

from enum import Enum
import re
from pathlib import Path
import shutil
from typing import List, Literal, Optional, Sequence

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import ConfigDict
from pydantic_xml import BaseXmlModel, attr, element
from unstructured.partition.auto import partition

_PRIMARY_CHUNK_STRATEGY = "by_title"
_FALLBACK_CHUNK_STRATEGY = "basic"
_DEFAULT_MAX_CHARACTERS = 8000

_XML_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def _sanitize_for_xml(text: str) -> str:
    """Strip control characters that are illegal in XML payloads."""

    if not text:
        return text
    return _XML_CONTROL_CHAR_RE.sub("", text)


class RemovedKind(str, Enum):
    """Types of non-spoken material removed during normalization."""

    FOREWORD = "foreword"
    PUBLISHER = "publisher_info"
    BLANK = "blank_line"
    PAGINATION = "pagination_marker"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_BREAK = "page_break"
    DEDICATION = "dedication"
    EPIGRAPH = "epigraph"
    COPYRIGHT = "copyright"
    OTHER = "other"


class TextType(str, Enum):
    """High-level discourse forms tailored to ancient/early-modern texts."""

    DIALOGUE = "dialogue"
    NARRATIVE_PERSONAL = "narrative_personal"
    NARRATIVE_HISTORICAL = "narrative_historical"
    TREATISE = "treatise"
    EPISTLE = "epistle"
    SECONDARY_EXPOSITION = "secondary_exposition"


class NormalizationSpeaker(BaseXmlModel, tag="speaker", skip_empty=True):
    """Speaker guess emitted by normalization."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = attr(
        description="Normalized speaker identifier (lowercase with dashes)."
    )
    evidence: Optional[str] = attr(
        default=None,
        description="Short justification for the speaker guess.",
    )


class NormalizedFragment(BaseXmlModel, tag="fragment", skip_empty=True):
    """Single fragment of the normalized text, optionally marking removals."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["text", "removed"] = attr(
        description="`text` retains spoken words; `removed` marks omissions."
    )
    removed_kind: Optional[RemovedKind] = attr(
        name="removed-kind",
        default=None,
        description="Removal classification when role is `removed`.",
    )
    content: str = element(
        tag="content",
        description="Literal text for this fragment.",
        default_factory=str,
    )

    def cleaned(self) -> str:
        return "" if self.role == "removed" else self.content


class TextPart(BaseXmlModel, tag="text-part", skip_empty=True):
    """Partitioned text block ready for normalization."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text_name: str = attr(
        name="text-name",
        description="Stable name derived from the source filename.",
    )
    part: int = attr(ge=1, description="One-based index for this partition segment.")
    source: str = attr(description="Absolute path of the source document.")
    content: str = element(tag="text", description="Exact text extracted.")


class NormalizedPart(BaseXmlModel, tag="normalized-part", skip_empty=True):
    """Normalization output persisted as XML."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text_name: str = attr(
        name="text-name",
        description="Stable name derived from the source filename.",
    )
    part: int = attr(ge=1, description="One-based index for this partition segment.")
    source: str = attr(description="Absolute path of the source document.")
    category: TextType = attr(description="Dominant discourse classification.")
    input_chars: int = attr(
        name="input-chars", ge=0, description="Character count before normalization."
    )
    output_chars: int = attr(
        name="output-chars", ge=0, description="Character count after normalization."
    )
    removed_chars: int = attr(
        name="removed-chars",
        ge=0,
        description="Characters removed during normalization.",
    )
    removal_ratio: float = attr(
        name="removal-ratio",
        description="removed_chars / input_chars ratio.",
    )
    paragraph_count: int = attr(
        name="paragraph-count", ge=0, description="Paragraph count after normalization."
    )
    speaker_candidate_count: int = attr(
        name="speaker-candidate-count",
        ge=0,
        description="Distinct speakers identified in the segment.",
    )
    likely_has_front_matter: Optional[bool] = attr(
        name="likely-has-front-matter",
        default=None,
        description="True when front matter was likely removed.",
    )
    fragments: List[NormalizedFragment] = element(
        tag="fragment",
        default_factory=list,
        description="Ordered fragments preserving removals inline.",
    )
    speakers: List[NormalizationSpeaker] = element(
        tag="speaker",
        default_factory=list,
        description="Speakers identified in this normalized segment.",
    )

    def cleaned_text(self) -> str:
        return "".join(fragment.cleaned() for fragment in self.fragments)

    def speaker_names(self) -> List[str]:
        return [speaker.name for speaker in self.speakers]


class NormalizeRequest(BaseXmlModel, tag="normalize-request", skip_empty=True):
    """Metadata envelope passed to the normalization LLM."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text_name: str = attr(name="text-name", description="Target text name.")
    part: int = attr(ge=1, description="Partition index to normalize.")
    source: str = attr(description="Absolute path of the source document.")


NORMALIZE_PROMPT = """
You are the text normalization lead preparing ancient and early-modern material for a professional single-narrator audiobook session. Deliver a performance-ready script foundation while preserving the author's intent.

Follow this workflow:
- Strip non-performable paratext (publisher notes, page furniture, dedications, epigraphs, front/back matter) while retaining context essential to the listener's comprehension. Evaluate introductions and forewords for storytelling value; keep, trim, or drop them accordingly.
- Guard the author's meaning: never summarize or paraphrase the main text. Only fix readability issues (OCR artifacts, broken hyphenation, illegible ligatures) and expand abbreviations when they would confuse a listener.
- Adapt the pace to the spoken word form by strategically enhancing text with pausing points (e.g. commas, semicolons, colon, em dashes) and breaking up overly complex sentences, aiming for breath-friendly segments without altering meaning.
- Preserve true paragraphing, merge mechanical line breaks, and keep rhetorical markers (vocatives, stage cues, verse lineation) that inform pacing and emphasis.
- Normalize spacing, punctuation, and encoding so the cleaned text reads smoothly aloud while respecting period diction.
- Convert all uppercase titles to standard capitalization. For all uppercase text, restore proper casing based on context, using punctuation instead of all-caps for emphasis.
- Convert any leftover numbers to words, except for dates and common single-digit enumerations (e.g., "Book 1", "ten thousand angels").
- To ensure text is readable aloud, spot OCR artifacts, special characters, and other fixable flaws, fixing them conservatively as you see fit to preserve substance and flow.
- Classify the dominant discourse form (category) and surface likely rhetorical cues for downstream cueing without repeating field descriptions. Stay consistent with the project's TextType definitions.
- Detect plausible speakers with concise supporting evidence when the text signals dialogic turns; never invent voices.
- Record every removal inline using <fragment role="removed"> so reviewers can audit the cleaning step.
"""


def partition_text(
    source: Path,
    parts_dir: Path,
    *,
    chunking_strategy: str = _PRIMARY_CHUNK_STRATEGY,
    max_characters: int = _DEFAULT_MAX_CHARACTERS,
) -> List[TextPart]:
    """Partition a source document into XML parts stored under ``parts_dir``.

    This implementation mirrors the disciplined workflow showcased in the Unstructured project's regression
    test ``test_unstructured/chunking/test_title.py::test_chunk_by_title_breaks_html_sections_at_headings``,
    because that fixture exercises the `chunk_by_title` strategy against a multipage HTML narrative and
    verifies that each heading-rooted section is collapsed into a single `CompositeElement` boundary with
    title-sensitive metadata intact.
    """

    if not source.exists():
        raise FileNotFoundError(f"File {source} does not exist.")

    parts_dir.mkdir(parents=True, exist_ok=True)
    text_name = source.stem

    elements = partition(
        filename=str(source.resolve()),
        chunking_strategy=chunking_strategy,
        max_characters=max_characters,
    )
    blocks: List[str] = []
    for element in elements:
        text = (element.text or "").strip()
        if text:
            blocks.append(text)

    if len(blocks) <= 1 and chunking_strategy == _PRIMARY_CHUNK_STRATEGY:
        logger.debug(
            "partition.blocks_insufficient text_name={} strategy={}",
            text_name,
            chunking_strategy,
        )
        fallback_elements = partition(
            filename=str(source.resolve()),
            chunking_strategy=_FALLBACK_CHUNK_STRATEGY,
            max_characters=max_characters,
        )
        fallback_blocks: List[str] = []
        for element in fallback_elements:
            text = (element.text or "").strip()
            if text:
                fallback_blocks.append(text)
        if len(fallback_blocks) > len(blocks):
            logger.debug(
                "partition.strategy_fallback text_name={} primary_chunks={} fallback_chunks={}",
                text_name,
                len(blocks),
                len(fallback_blocks),
            )
            blocks = fallback_blocks

    if not blocks:
        raise ValueError("Partitioning produced no text blocks.")

    parts: List[TextPart] = []
    for index, content in enumerate(blocks, start=1):
        document = TextPart(
            text_name=text_name,
            part=index,
            source=str(source.resolve()),
            content=content,
        )
        xml_path = parts_dir / f"{text_name}-part{index:03d}.xml"
        payload = document.to_xml(
            encoding="unicode", pretty_print=True, skip_empty=True
        )
        assert isinstance(payload, str)
        xml_path.write_text(payload)
        logger.debug("partition.part_saved path={} chars={}", xml_path, len(content))
        parts.append(document)

    logger.info(
        "partition.done source={source} parts={parts}",
        source=source,
        parts=len(parts),
    )
    return parts


def load_parts(parts_dir: Path) -> List[TextPart]:
    """Load partition outputs from disk into memory."""

    documents: List[TextPart] = []
    for xml_path in sorted(parts_dir.glob("*.xml")):
        logger.debug("load_parts.reading {}", xml_path)
        document = TextPart.from_xml(xml_path.read_text())
        documents.append(document)
    documents.sort(key=lambda part: (part.text_name, part.part))
    return documents


def normalize_parts(
    parts: Sequence[TextPart],
    normalize_dir: Path,
    llm: ChatOpenAI,
    *,
    force: bool = False,
) -> List[NormalizedPart]:
    """Run normalization over partitioned parts, persist XML, and return documents."""

    if not parts:
        raise ValueError("No parts provided for normalization.")

    if force:
        shutil.rmtree(normalize_dir, ignore_errors=True)
    normalize_dir.mkdir(parents=True, exist_ok=True)

    results: List[NormalizedPart] = []
    for part in parts:
        xml_path = (
            normalize_dir / f"{part.text_name}-part{part.part:03d}-normalized.xml"
        )
        if xml_path.exists():
            logger.info(
                "Reading existing normalized part {}",
                xml_path,
            )
            result = NormalizedPart.from_xml(xml_path.read_text())
            results.append(result)
            continue

        metadata_xml = NormalizeRequest(
            text_name=part.text_name,
            part=part.part,
            source=part.source,
        ).to_xml(encoding="unicode", pretty_print=True, skip_empty=True)
        assert isinstance(metadata_xml, str)
        messages = [
            SystemMessage(content=NORMALIZE_PROMPT),
            HumanMessage(
                content=(
                    "== METADATA ==\n"
                    f"{metadata_xml}\n\n"
                    "== INPUT TEXT ==\n"
                    f"{part.content}"
                )
            ),
        ]
        logger.info(
            "normalize.part text_name={name} part={part_index} chars={chars}",
            name=part.text_name,
            part_index=part.part,
            chars=len(part.content),
        )
        callback = UsageMetadataCallbackHandler()
        result = llm.with_structured_output(NormalizedPart, include_raw=True).invoke(
            messages, config=RunnableConfig(callbacks=[callback])
        )
        normalized: NormalizedPart = result["parsed"]
        assert (
            normalized.text_name == part.text_name
        ), "LLM returned mismatched text_name."
        assert normalized.part == part.part, "LLM returned mismatched part number."
        assert normalized.source == part.source, "LLM returned mismatched source path."
        assert normalized.fragments, "Normalized fragments missing."
        logger.debug("normalize.tokens usage={usage}", usage=callback.usage_metadata)

        persisted_payload = normalized.to_xml(
            encoding="unicode", pretty_print=True, skip_empty=True
        )
        persisted_payload_str = (
            persisted_payload
            if isinstance(persisted_payload, str)
            else persisted_payload.decode()
        )
        xml_path.write_text(persisted_payload_str)

        logger.debug(
            (
                "normalize.heuristics input={input_chars} output={output_chars} removed={removed_chars} "
                "removal_ratio={ratio:.3f} paragraphs={paragraphs} speakers={speakers}"
            ),
            input_chars=normalized.input_chars,
            output_chars=normalized.output_chars,
            removed_chars=normalized.removed_chars,
            ratio=normalized.removal_ratio,
            paragraphs=normalized.paragraph_count,
            speakers=normalized.speaker_candidate_count,
        )
        results.append(normalized)

    logger.info("normalize.done parts={count}", count=len(results))
    return results


def load_normalized_parts(normalize_dir: Path) -> List[NormalizedPart]:
    """Read normalized XML outputs from disk."""

    documents: List[NormalizedPart] = []
    for xml_path in sorted(normalize_dir.glob("*-normalized.xml")):
        document = NormalizedPart.from_xml(xml_path.read_text())
        documents.append(document)
    documents.sort(key=lambda doc: (doc.text_name, doc.part))
    return documents
