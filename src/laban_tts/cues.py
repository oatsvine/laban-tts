from __future__ import annotations

from loguru import logger
import re
from functools import lru_cache
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_xml import BaseXmlModel, attr, element, wrapped

from laban_tts.normalize import TextType

import tiktoken


_MAX_TOKENS_PER_CHUNK = 1000
_CLOSING_PUNCTUATION = "\"'”’»)]}››"
_ABBREVIATIONS = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "sr.",
    "jr.",
    "st.",
    "no.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
    "fig.",
    "a.m.",
    "p.m.",
    "u.s.",
    "u.k.",
    "u.n.",
    "jan.",
    "feb.",
    "mar.",
    "apr.",
    "jun.",
    "jul.",
    "aug.",
    "sep.",
    "sept.",
    "oct.",
    "nov.",
    "dec.",
}
_SECONDARY_SEPARATORS = [
    re.compile(r"(?<=;)[ \t]+"),
    re.compile(r"(?<=:)[ \t]+"),
    re.compile(r"(?<=,)\s+"),
    re.compile(r"\s+—\s+"),
    re.compile(r"\s+–\s+"),
    re.compile(r"\s+-\s+"),
]


def _split_into_sentences(text: str) -> List[str]:
    sentences: List[str] = []
    length = len(text)
    start = 0
    i = 0
    while i < length:
        ch = text[i]
        if ch == "." and text[i : i + 3] == "...":
            i += 3
            continue
        if ch in ".!?":
            if (
                ch == "."
                and i > 0
                and i + 1 < length
                and text[i - 1].isdigit()
                and text[i + 1].isdigit()
            ):
                i += 1
                continue

            end = i + 1
            while end < length and text[end] in _CLOSING_PUNCTUATION:
                end += 1

            segment = text[start:end]
            prev_words = re.findall(r"\b\w+\b", segment)
            prev_word = f"{prev_words[-1].lower()}." if prev_words else ""
            if prev_word in _ABBREVIATIONS:
                i += 1
                continue

            sentence = text[start:end].strip()
            if sentence:
                sentences.append(sentence)
            start = end
            while start < length and text[start].isspace():
                start += 1
            i = start
            continue
        i += 1

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


@lru_cache(maxsize=1)
def _token_encoder():
    if tiktoken is None:  # pragma: no cover - optional dependency
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        try:
            return tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            return None


def _estimate_tokens(text: str) -> int:
    encoder = _token_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:  # pragma: no cover - fallback path
            pass
    # simple heuristic fallback (~4 characters per token)
    text = text.strip()
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _split_on_pattern(text: str, pattern: re.Pattern[str]) -> List[str]:
    parts: List[str] = []
    last = 0
    for match in pattern.finditer(text):
        start = last
        end = match.start()
        chunk = text[start:end].strip()
        if chunk:
            parts.append(chunk)
        last = match.end()
    tail = text[last:].strip()
    if tail:
        parts.append(tail)
    return parts if parts else [text.strip()]


def _greedy_word_split(text: str) -> List[str]:
    parts: List[str] = []
    current = ""
    for match in re.finditer(r"\S+\s*", text):
        fragment = match.group()
        tentative = current + fragment
        if current and _estimate_tokens(tentative) > _MAX_TOKENS_PER_CHUNK:
            parts.append(current.rstrip())
            current = fragment
        else:
            current = tentative
    if current.strip():
        parts.append(current.rstrip())
    cleaned = [part.strip() for part in parts if part.strip()]
    for part in cleaned:
        if _estimate_tokens(part) > _MAX_TOKENS_PER_CHUNK:
            raise ValueError(
                f"Unable to split chunk within token limit (len={_estimate_tokens(part)})"
            )
    return cleaned


def _is_all_punctuation(text: str) -> bool:
    stripped = text.strip()
    return not stripped or all(not ch.isalnum() for ch in stripped)


# Primer content is shipped as markdown so the LLM receives a literature-backed
# acting guide before we ask for cues. citeturn1firecrawl_scrape1turn1firecrawl_scrape2turn1firecrawl_scrape0
# Literature grounding:
#   • Cicely Berry's RSC voice work insists actors fuse rhetorical analysis with embodied vocal choices, providing the
#     rehearsal-room baseline we reference. citeturn6firecrawl_scrape0
#   • Laban Effort Actions have been successfully translated to voice acting workflows, letting performers map physical
#     dynamics like Wring, Punch, or Float straight into booth delivery. citeturn7firecrawl_scrape0
#   • Contemporary rhetoric handbooks for actors (e.g., Brandreth) document how classical invention/disposition drills feed
#     directly into modern cueing conversations. citeturn1firecrawl_scrape1
CUE_PRIMER = """# Cue Performance Primer

This primer codifies the Royal Shakespeare Company–informed text and voice practice popularised by
Cicely Berry, weaving rhetorical invention/disposition work with Laban effort translations so cues stay
aligned with studio-proven narration craft while downstream synthesis remains deterministic.

## Delivery Profiles — Laban Effort Actions (Vocal Translation)
Eight Laban Effort Actions provide consistent vocal energy palettes by translating body dynamics into
speech attitude; keep selections steady unless the scene’s emotional motor changes.
- **Punch**: direct, strong, sudden articulation for decisive declarations and martial cues.
- **Press**: direct, strong, sustained weight that underpins earnest appeals or judicial argument.
- **Slash**: flexible strength with sudden release that surfaces danger, volatility, or exposed threat.
- **Wring**: indirect, strong, sustained tension that voices internal struggle or reverent grief.
- **Flick**: indirect, light, sudden sparkle suited to wit, discovery, or quick contrasts.
- **Dab**: direct, light, sustained precision that keeps procedural material tidy without heaviness.
- **Float**: indirect, light, sustained buoyancy for awe, reminiscence, or gentle consolation.
- **Glide**: indirect, light, sustained smoothness that holds descriptive exposition legibly.

## Rhetoric Tags — Classical & Extended Genres
Choose rhetoric tags to flag the communicative goal of each chunk; change them only when the text’s
purpose shifts. Classical genres used in RSC rehearsal rooms remain practical for actors because they
pair with invention/disposition drills that keep arguments intelligible for listeners. Lament functions
as a stylistic overlay rather than a separate genre.
- **deliberative**: future-facing persuasion that weighs benefits and harms; cue reflective linking and tightened pacing.
- **judicial**: past-tense evaluation assigning blame or vindication; use methodical pacing and verbal underlining near evidence.
- **epideictic**: present-tense praise or blame; brighten tone and allow expansive vowels.
- **didactic**: instructional scaffolding; stabilize rhythm and keep variance low so concepts land cleanly.
- **narrative**: storytelling or scene painting; widen tempo tolerance and support imagistic transitions.
- **lament** *(overlay)*: reverent communal grief layered on top of the active genre; elongate phrasing and soften attacks while keeping the base tag intact.

## Emphasis Techniques — Prosodic Toolkit
Use no more than three emphasis spans per chunk. Pair each span with a primer-supported technique, and
skip solutions that repeat identical spans in neighboring chunks.
- **triad_landing**: finish a three-part build with a sustained landing to leverage tricolon cadence.
- **antithesis**: sharpen opposing clauses with a pivot that spotlights the contrast pivot.
- **definition**: tighten before “term + is/means” and land crisply on the clarifying clause.
- **rhetorical_question**: lift pitch near the interrogative apex without sounding uncertain.
- **imperative**: add forward-driving energy and a micro-pre-beat to deliver direct calls to action.
- **proper_name**: give names a subtle lift to reinforce rapport and listener memory cues.

## Lament Application Notes
Reserve the `lament` emphasis overlay for passages that explicitly call for communal mourning or heroic
grief. Expand resonance downward, add gentle pre/post beats, and keep the underlying rhetoric tag in
place so downstream automation retains genre intent.
"""

CUE_PRIMER_TEXT = CUE_PRIMER + "\n\n## Machine-Readable Tuning Map\n"

# Scoring addendum draws on Singspiel structure (dialogue alternating with airs), orchestration practice
# from classical film scoring pedagogy, instrumentation-emotion pairings, evidence that full songs mask
# narration, and audiobook delivery norms discouraging continuous beds. citeturn2view0turn3view0turn10search2turn8search9turn9search2
SCORING_PROMPT_FRAGMENT = """## Scoring Cue Addendum — Classical Palettes Only
Music cues are optional. Use them sparingly when a classical texture will heighten the listener’s
experience without masking narration.

- Choose `placement` from {opening_credit, closing_credit, chapter_button, transition_bridge,
  underscore_swell}. Intro/outro values bookend projects, `chapter_button` and `transition_bridge`
  sit between spoken blocks, and `underscore_swell` is a brief under-bed inside a single beat.
- Choose `texture` from {chamber_strings, solo_piano, woodwind_trio, harp_and_flute, brass_chorale,
  solo_cello}. Stay acoustic; no hybrid or percussive scoring.
- Choose `emotion` from {wonder, tension, resolve, melancholy, levity, solemnity}. Pair the selection
  with the rhetoric and profile already assigned.
- Choose `tempo` from {adagio, andante, moderato, allegro}. Classical pacing terms keep cues readable.
- Choose `intensity` from {delicate, moderate, expansive}. `underscore_swell` must remain delicate or
  moderate so dialogue stays intelligible. Other placements may pick any intensity the moment can carry.
- Choose `cadence` from {tail_out, hard_out, suspended}. Tail outs suit fades into narration, hard outs
  nail sync points, suspended leaves air before the next beat.
- Never loop continuous beds under whole chapters. Credits or rare chapter buttons can include short stings;
  interstitial cues should respect breath between spoken segments.

Omit the `scoring` object when silence is stronger than music.
"""

# ---------- Rhetorical steering (secondary tag) ----------


# Rhetorical tags align with classical and rehearsal-room rhetoric training that
# RSC practitioners adapt for modern performance. citeturn1firecrawl_scrape1turn1firecrawl_scrape2
class RhetoricTag(str, Enum):
    deliberative = "deliberative"
    judicial = "judicial"
    epideictic = "epideictic"
    didactic = "didactic"
    narrative = "narrative"
    lament = "lament"


# ---------- Laban-derived delivery profiles ----------


# Delivery profiles adapt Laban's Effort Actions so the narrator's vocal energy
# tracks established movement/voice pairings used in studio training. citeturn1firecrawl_scrape0
class Profile(str, Enum):
    Punch = "Punch"
    Press = "Press"
    Slash = "Slash"
    Wring = "Wring"
    Flick = "Flick"
    Dab = "Dab"
    Float = "Float"
    Glide = "Glide"


# ---------- Scoring cues (optional music direction) ----------


class ScoringPlacement(str, Enum):
    """Where scoring should appear relative to narration."""

    before = "before"
    after = "after"
    chapter_button = "chapter_button"
    transition_bridge = "transition_bridge"
    underscore_swell = "underscore_swell"


class ScoringTexture(str, Enum):
    """Classical palettes that stay within acoustic instrumentation."""

    chamber_strings = "chamber_strings"
    solo_piano = "solo_piano"
    woodwind_trio = "woodwind_trio"
    harp_and_flute = "harp_and_flute"
    brass_chorale = "brass_chorale"
    solo_cello = "solo_cello"


class ScoringEmotion(str, Enum):
    wonder = "wonder"
    tension = "tension"
    resolve = "resolve"
    melancholy = "melancholy"
    levity = "levity"
    solemnity = "solemnity"


class ScoringTempo(str, Enum):
    adagio = "adagio"
    andante = "andante"
    moderato = "moderato"
    allegro = "allegro"


class ScoringIntensity(str, Enum):
    delicate = "delicate"
    moderate = "moderate"
    expansive = "expansive"


class ScoringCadence(str, Enum):
    tail_out = "tail_out"
    hard_out = "hard_out"
    suspended = "suspended"


class ScoringCue(BaseXmlModel, tag="scoring"):
    """Optional classical scoring direction attached to a cue chunk."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    placement: ScoringPlacement = attr(
        description="Where the cue should fall relative to narration blocks."
    )
    texture: ScoringTexture = attr(
        description="Primary instrumentation palette (classical acoustic)."
    )
    emotion: ScoringEmotion = attr(
        description="Emotional contour the music should reinforce."
    )
    tempo: ScoringTempo = attr(
        default=ScoringTempo.andante,
        description="Overall pace reference using classical terminology.",
    )
    intensity: ScoringIntensity = attr(
        default=ScoringIntensity.delicate,
        description="Relative dynamic weight to keep balance with narration.",
    )
    cadence: ScoringCadence = attr(
        default=ScoringCadence.tail_out,
        description="How the music should exit the moment.",
    )

    @model_validator(mode="after")
    def _validate_underscore_constraints(self) -> "ScoringCue":
        if (
            self.placement == ScoringPlacement.underscore_swell
            and self.intensity == ScoringIntensity.expansive
        ):
            raise ValueError(
                "underscore_swell cues must stay delicate or moderate to preserve narration intelligibility"
            )
        return self


# ---------- Engine params (exactly what Chatterbox accepts) ----------


class EngineParams(BaseModel):
    """
    Exact knobs passed to chatterbox.[Multilingual]TTS.generate().
    Keep conservative defaults to avoid over-acting.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Ranges mirror Chatterbox defaults (README, 2025) where exaggeration above
    # 0.7 accelerates pace unless cfg_weight drops toward 0.3. Keep adjustments
    # conservative so narration stays natural.
    exaggeration: float = Field(
        0.4, ge=0.3, le=1.1, description="Emotion intensity; 0.4-1.1."
    )
    cfg_weight: float = Field(
        0.5,
        ge=0.4,
        le=0.6,
        description="CFG weight; lower = more deliberate/slower pacing.",
    )
    temperature: float = Field(
        0.7,
        ge=0.5,
        le=0.9,
        description="Sampling variability; keep 0.6-0.9 for narration.",
    )
    repetition_penalty: float = Field(
        1.2, ge=1.0, le=2.0, description="Discourages token loops."
    )
    min_p: float = Field(0.05, ge=0.0, le=1.0, description="Typical 0.05.")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Typical 0.9-1.0.")
    # NOTE: We don't support this.
    # language_id: Optional[str] = Field(
    #     default=None, description="Multilingual model language code (e.g., 'en')."
    # )


# ---------- Emphasis (engine-agnostic; for renderer micro-prosody) ----------


class EmphasisStrategy(str, Enum):
    TRIAD_LANDING = "triad_landing"  # land on third item
    ANTITHESIS = "antithesis"  # A vs. B contrast
    DEFINITION = "definition"  # term + is/means
    RHETORICAL_QUESTION = "rhetorical_question"
    IMPERATIVE = "imperative"
    PROPER_NAME = "proper_name"


# Emphasis strategies mirror rehearsal-room rhetoric drills (triads, antithesis,
# interrogatives, directives, and name cues) used to support intelligibility and
# listener engagement. citeturn1firecrawl_scrape2


class EmphasisSpan(BaseXmlModel, tag="span"):
    """
    Inline emphasis guidance for renderer (no SSML): use to nudge micro-pauses or slight gain.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    phrase: str = Field(
        description="Exact substring to emphasize (must appear verbatim in text)."
    )
    strategy: EmphasisStrategy = attr(
        description="Acting technique informing emphasis."
    )
    pause_before_ms: int = attr(
        name="pause-before-ms",
        default=0,
        ge=0,
        le=600,
        description="Optional micro-pause before phrase.",
    )
    pause_after_ms: int = attr(
        name="pause-after-ms",
        default=0,
        ge=0,
        le=600,
        description="Optional micro-pause after phrase.",
    )


# Precomputed tuning tables blend Laban effort profiles with rhetorical intent.
EngineParamVector = Dict[str, float]

_ENGINE_PARAM_FIELD_LIMITS: Dict[str, Tuple[float, float]] = {
    "exaggeration": (0.3, 1.1),
    "cfg_weight": (0.4, 0.6),
    "temperature": (0.5, 0.9),
    "repetition_penalty": (1.0, 2.0),
    "min_p": (0.0, 1.0),
    "top_p": (0.0, 1.0),
}

_PROFILE_BASE_PARAMS: Dict[Profile, EngineParamVector] = {
    Profile.Punch: {
        "exaggeration": 0.84,
        "cfg_weight": 0.57,
        "temperature": 0.78,
        "repetition_penalty": 1.34,
        "min_p": 0.05,
        "top_p": 0.92,
    },
    Profile.Press: {
        "exaggeration": 0.63,
        "cfg_weight": 0.50,
        "temperature": 0.70,
        "repetition_penalty": 1.24,
        "min_p": 0.05,
        "top_p": 0.90,
    },
    Profile.Slash: {
        "exaggeration": 0.81,
        "cfg_weight": 0.56,
        "temperature": 0.82,
        "repetition_penalty": 1.32,
        "min_p": 0.05,
        "top_p": 0.91,
    },
    Profile.Wring: {
        "exaggeration": 0.72,
        "cfg_weight": 0.44,
        "temperature": 0.64,
        "repetition_penalty": 1.28,
        "min_p": 0.05,
        "top_p": 0.88,
    },
    Profile.Flick: {
        "exaggeration": 0.68,
        "cfg_weight": 0.53,
        "temperature": 0.84,
        "repetition_penalty": 1.18,
        "min_p": 0.05,
        "top_p": 0.94,
    },
    Profile.Dab: {
        "exaggeration": 0.60,
        "cfg_weight": 0.58,
        "temperature": 0.68,
        "repetition_penalty": 1.20,
        "min_p": 0.05,
        "top_p": 0.90,
    },
    Profile.Float: {
        "exaggeration": 0.56,
        "cfg_weight": 0.45,
        "temperature": 0.72,
        "repetition_penalty": 1.12,
        "min_p": 0.05,
        "top_p": 0.95,
    },
    Profile.Glide: {
        "exaggeration": 0.58,
        "cfg_weight": 0.52,
        "temperature": 0.70,
        "repetition_penalty": 1.14,
        "min_p": 0.05,
        "top_p": 0.94,
    },
}

_RHETORIC_MODIFIERS: Dict[RhetoricTag, EngineParamVector] = {
    RhetoricTag.deliberative: {
        "cfg_weight": -0.04,
        "temperature": -0.04,
        "repetition_penalty": 0.08,
        "top_p": -0.03,
    },
    RhetoricTag.judicial: {
        "cfg_weight": -0.03,
        "temperature": -0.05,
        "repetition_penalty": 0.10,
        "top_p": -0.04,
    },
    RhetoricTag.epideictic: {
        "exaggeration": 0.10,
        "temperature": 0.06,
        "top_p": 0.02,
    },
    RhetoricTag.didactic: {
        "cfg_weight": -0.02,
        "temperature": -0.03,
        "repetition_penalty": 0.05,
        "top_p": -0.02,
    },
    RhetoricTag.narrative: {
        "cfg_weight": 0.02,
        "temperature": 0.04,
        "top_p": 0.03,
    },
    RhetoricTag.lament: {
        "exaggeration": 0.08,
        "cfg_weight": -0.05,
        "temperature": -0.02,
        "top_p": -0.02,
    },
}

_TEXT_TYPE_MODIFIERS: Dict[TextType, EngineParamVector] = {
    TextType.DIALOGUE: {
        "temperature": 0.02,
        "repetition_penalty": 0.04,
    },
    TextType.NARRATIVE_PERSONAL: {
        "cfg_weight": -0.01,
        "temperature": 0.01,
        "top_p": 0.02,
    },
    TextType.NARRATIVE_HISTORICAL: {
        "cfg_weight": 0.01,
        "repetition_penalty": 0.03,
    },
    TextType.TREATISE: {
        "cfg_weight": 0.02,
        "temperature": -0.02,
        "repetition_penalty": 0.05,
        "top_p": -0.03,
    },
    TextType.EPISTLE: {
        "cfg_weight": -0.015,
        "exaggeration": 0.02,
        "temperature": -0.01,
    },
    TextType.SECONDARY_EXPOSITION: {
        "cfg_weight": 0.01,
        "temperature": -0.03,
        "repetition_penalty": 0.06,
        "top_p": -0.04,
    },
}

_EMPHASIS_BASE_RAISE = 0.015
_EMPHASIS_MAX_ACCENT = 3

_EMPHASIS_STRATEGY_MODIFIERS: Dict[EmphasisStrategy, EngineParamVector] = {
    EmphasisStrategy.TRIAD_LANDING: {
        "exaggeration": 0.02,
        "cfg_weight": 0.01,
    },
    EmphasisStrategy.ANTITHESIS: {
        "temperature": 0.015,
        "repetition_penalty": 0.04,
    },
    EmphasisStrategy.DEFINITION: {
        "temperature": -0.01,
        "repetition_penalty": 0.05,
    },
    EmphasisStrategy.RHETORICAL_QUESTION: {
        "temperature": 0.02,
        "top_p": 0.01,
    },
    EmphasisStrategy.IMPERATIVE: {
        "exaggeration": 0.04,
        "cfg_weight": 0.015,
    },
    EmphasisStrategy.PROPER_NAME: {
        "repetition_penalty": 0.06,
    },
}


def _apply_adjustments(base: EngineParamVector, adjustments: EngineParamVector) -> None:
    for key, delta in adjustments.items():
        base[key] = base.get(key, 0.0) + delta


def _clamp_param(name: str, value: float) -> float:
    lower, upper = _ENGINE_PARAM_FIELD_LIMITS[name]
    return max(lower, min(upper, value))


# ---------- Chunk and script ----------


class CuedChunk(BaseXmlModel, tag="chunk", skip_empty=True):
    """
    Atomic delivery unit for synthesis. One chunk = one contiguous line of delivery
    with a single set of engine params. No mid-chunk changes to params.
    """

    idx: int = attr(description="1-based index in reading order.")
    text: str = element(
        tag="text",
        description="Normalized text to synthesize. Punctuation already audience-friendly.",
        default_factory=str,
    )
    speaker: str = attr(
        description="Normalized speaker name for this chunk (lowercase, dashes), or 'narrator' if generic."
    )
    text_type: TextType = attr(
        name="text-type",
        description="Discourse category for this chunk (see Normalization stage).",
    )
    rhetoric: RhetoricTag = attr(
        description="Optional rhetorical function, if inferable."
    )
    profile: Profile = attr(
        description="Delivery palette influencing engine defaults (Laban-derived)."
    )
    audio_prompt_path: Optional[str] = attr(
        name="audio-prompt",
        default=None,
        description="Per-chunk voice conditioning file; optional.",
    )
    pre_pause_ms: int = attr(
        name="pre-pause-ms",
        default=0,
        ge=0,
        le=4000,
        description="Silence before this chunk (externalized).",
    )
    post_pause_ms: int = attr(
        name="post-pause-ms",
        default=0,
        ge=0,
        le=4000,
        description="Silence after this chunk (externalized).",
    )
    emphasis: List[EmphasisSpan] = wrapped(
        "emphasis",
        element(
            tag="span",
            default_factory=list,
            description="Optional emphasis spans for renderer to apply subtle micro-pauses/gain.",
        ),
    )

    def split_text(self) -> List[str]:
        """Split text into sentences for renderer chunking."""

        text = self.text.strip()
        if not text:
            return []

        sentences = _split_into_sentences(text)
        chunks: List[str] = []
        for sentence in sentences:
            working = [sentence]
            for pattern in _SECONDARY_SEPARATORS:
                updated: List[str] = []
                changed = False
                for fragment in working:
                    if _estimate_tokens(fragment) > _MAX_TOKENS_PER_CHUNK:
                        pieces = _split_on_pattern(fragment, pattern)
                        if len(pieces) > 1:
                            changed = True
                            updated.extend(pieces)
                        else:
                            updated.append(fragment)
                    else:
                        updated.append(fragment)
                working = updated
                if changed and all(
                    _estimate_tokens(fragment) <= _MAX_TOKENS_PER_CHUNK
                    for fragment in working
                ):
                    break

            for fragment in working:
                if _estimate_tokens(fragment) <= _MAX_TOKENS_PER_CHUNK:
                    chunks.append(fragment.strip())
                else:
                    chunks.extend(_greedy_word_split(fragment))

        normalised_original = " ".join(text.split())
        normalised_joined = " ".join(" ".join(chunk.split()) for chunk in chunks)
        if normalised_joined != normalised_original:
            logger.warning(
                "CuedChunk.split_text produced altered content: {} vs {}",
                normalised_original,
                normalised_joined,
            )
        sanitized_chunks: List[str] = []
        for i, chunk in enumerate(chunks, start=1):
            if not chunk.strip():
                logger.warning(
                    "CuedChunk.split_text produced empty chunk {} / {}", i, len(chunks)
                )
                continue
            if _is_all_punctuation(chunk):
                logger.warning(
                    "CuedChunk.split_text produced punctuation-only chunk {} / {}: {}",
                    i,
                    len(chunks),
                    chunk,
                )
                continue
            sanitized_chunks.append(chunk.strip())

        return sanitized_chunks

    def engine_params(self) -> EngineParams:
        base = dict(_PROFILE_BASE_PARAMS[self.profile])

        text_shift = _TEXT_TYPE_MODIFIERS.get(self.text_type)
        if text_shift:
            _apply_adjustments(base, text_shift)

        rhetoric_shift = _RHETORIC_MODIFIERS.get(self.rhetoric)
        if rhetoric_shift:
            _apply_adjustments(base, rhetoric_shift)

        if self.emphasis:
            base["exaggeration"] += (
                min(len(self.emphasis), _EMPHASIS_MAX_ACCENT) * _EMPHASIS_BASE_RAISE
            )
            for span in self.emphasis:
                if modifiers := _EMPHASIS_STRATEGY_MODIFIERS.get(span.strategy):
                    _apply_adjustments(base, modifiers)

        if base["exaggeration"] >= 0.8:
            base["cfg_weight"] -= min(0.1, 0.35 * (base["exaggeration"] - 0.8))

        if self.pre_pause_ms >= 600 or self.post_pause_ms >= 600:
            base["cfg_weight"] -= 0.01
        if self.post_pause_ms >= 1000:
            base["temperature"] -= 0.01

        base["top_p"] = max(base["top_p"], 0.85)

        return EngineParams(
            **{name: _clamp_param(name, value) for name, value in base.items()}
        )


class ScoredCuedChunk(CuedChunk):
    """Cue chunk variant that carries scoring direction via inheritance."""

    scoring: ScoringCue = element(
        tag="scoring",
        description="Classical scoring cue attached to this chunk.",
    )


class CuedScript(BaseXmlModel, tag="cue-script"):
    """
    Full cue plan for a document; iterate chunks and synthesize in order.
    """

    model_config = ConfigDict(extra="forbid")

    text_name: str = attr(
        name="text-name", description="Name of this text for filenames and logs."
    )
    speakers: List[str] = wrapped(
        "speakers",
        element(
            tag="speaker",
            default_factory=list,
            description="All speakers that appear in this script, normalized (lowercase, dashes).",
        ),
    )
    chunks: List[CuedChunk] = element(
        tag="chunk",
        default_factory=list,
        description="Ordered list of atomic chunks.",
    )


CUE_PROMPT = """
You are the cueing director for a single adult male narrator delivering long-form literature. Provide
production-ready cues that let a trained audiobook actor sustain stamina, emotional truth, and clarity.

Use the Cue Performance Primer (supplied earlier as a separate system message) to ground every
speaker, rhetoric tag, delivery profile, and emphasis choice. Apply that knowledge rather than
restating it.

Follow these directives:
- Segment the cleaned text into breath-aware chunks (about one to three sentences or up to roughly
  twelve seconds). Align boundaries with punctuation unless the primer indicates a motivated pause.
- Honor speaker annotations from normalization; every speaker change becomes its own chunk with any
  necessary pre-beat to signal turn-taking.
- Treat rhetoric tags as interpretive intent. Adjust only when the passage's purpose shifts, and use
  the primer's definitions to justify the new choice.
- Choose profiles that stay stable across adjacent chunks unless the emotional mechanics clearly
  change. Do not mention synthesis parameters; engine values are derived downstream from your
  categories.
- Use emphasis spans sparingly (max three per chunk) when the primer's techniques genuinely amplify
  meaning. Avoid duplicating spans that cover identical text in consecutive chunks.
- Add pre/post pauses only when you are creating a deliberate acting beat, not as formatting or
  breathing hacks.
- Only emit the optional `scoring` object when music meaningfully supports the beat. If you include it,
  draw values from the Scoring Cue Addendum (placement, texture, emotion, tempo, intensity, cadence)
  and keep any `underscore_swell` entries delicate or moderate so words stay intelligible.
- Do not propose continuous beds; music may bookend credits or appear as short interstitial buttons but
  silence is preferred between spoken segments.
- Preserve normalization's punctuation and metadata; never introduce commentary or structure beyond
  the schema.

Return JSON that conforms to the CuedScript schema exactly, with no text outside the JSON document.
"""
# Primer design summary:
#   - Profiles follow Laban Effort Actions as adapted for RSC voice work so vocal energy mirrors
#     movement dynamics familiar to trained narrators. citeturn1firecrawl_scrape0
#   - Rhetoric tags retain the classical core used in modern rehearsal rhetoric so actors can apply
#     invention/disposition choices consistently; lament remains an emphasis overlay. citeturn1firecrawl_scrape1turn1firecrawl_scrape2
#   - Emphasis strategies reflect rehearsal drills (triads, antithesis, interrogatives, directives, name
#     cues) that keep listener attention without over-marking the text. citeturn1firecrawl_scrape2

#   Design Adjustments Required

#   1. Author a “Performance Primer” resource
#       - Structure: (a) Laban Effort summaries with vocal implications; (b) Rhetoric tag definitions with exemplar cues; (c) Emphasis techniques with when/why guidance; (d) Optional appendix on lament as an emphasis/style toolkit rather than a separate rhetoric tag.
#       - Store this primer (e.g., tts/prompts/cue_primer.md) and load/concatenate it ahead of the cue prompt so the LLM receives stable, scholarly framing.
#   2. Rewrite CUE_PROMPT to reference the primer, not engine knobs
#       - Remove explicit parameter ranges.
#       - Instruct the model to choose profile/rhetoric/emphasis strictly from the primer and to justify transitions in those terms.
#       - Emphasize continuity (“keep profiles steady unless rhetoric shift demands it”) and make clear that numeric synthesis control happens downstream.
#   3. Embed a deterministic mapping layer in code (already partly done in engine_params)
#       - Replace the ad-hoc tables with a data structure derived from the primer (e.g., YAML or JSON) so docs and code stay synchronized.
#       - Ensure engine_params() looks up profile defaults and applies rhetoric/emphasis modifiers exactly as documented.
#   4. Validation & QA hooks
#       - Add unit tests that iterate over every profile/rhetoric combination to guarantee the mapping stays within allowed boundaries.
#       - Log primer-driven choices during cue generation to confirm the LLM is referencing the prescribed vocabulary.
#   5. Team alignment
#       - Circulate the primer and prompt rewrite to narration leads/SMEs for approval.
#       - Once ratified, freeze the vocabulary (profile/rhetoric/emphasis) to avoid drift; any future expansion should go back through literature review.
