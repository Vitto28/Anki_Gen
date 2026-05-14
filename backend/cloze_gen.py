"""
LLM-assisted cloze generation via Ollama.

Uses structured JSON output when the server supports it (format + JSON schema),
so the model cannot mix prose with the card. Falls back to plain text + cleanup
if structured generation fails.

Retry logic uses progressively stricter prompts.
"""
from __future__ import annotations
import json
import re
import httpx
from models import Chunk, ClozeCard, GenerateSettings, ReviewSeverity, new_id
from validator import validate_cloze_card

# ── Ollama structured output: single string field (value may contain { } freely) ──
CLOZE_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "cloze_text": {
            "type": "string",
            "minLength": 1,
            "description": "Full source text with only Anki cloze tags inserted",
        },
    },
    "required": ["cloze_text"],
    "additionalProperties": False,
}

# Gold-standard few-shot (syntax only; different wording than typical chunks avoids copy-paste).
_FEW_SHOT = r"""EXAMPLE (correct Anki cloze syntax only — two opening braces, c, digit, two colons, hidden text, two closing braces; multiline inside one cloze is allowed):

One of the most basic results in queuing theory is {{c1::Little's theorem}}. Let's state the theorem right away,
initially a bit informally. Consider a stable system over a long-enough period of time. Let {{c2::N}} be the
average {{c3::number of customers}} in the system. Let {{c4::T}} be the average time that {{c5::
a customer spends in the
system
}}. Let {{c6::λ}} be the {{c7::arrival rate}}.

Your task uses the SOURCE_TEXT below, not this example text."""

_SYSTEM_JSON = """You are an Anki cloze note assistant. You MUST respond with valid JSON only,
matching the requested schema. The cloze_text value must be the entire SOURCE_TEXT with Anki
cloze deletions inserted. Rules:
- Copy SOURCE_TEXT exactly: same characters, spaces, and newlines; only insert cloze markers.
- Each marker is EXACTLY: two { characters, then c, then a digit 1-9, then two : characters,
  then the exact substring from the source that will be hidden, then two } characters.
- Never use a single { or } for a cloze. Never use forms like {c: 1}, : 1:, {term: ...}, or LaTeX {τ}.
- Every cloze must hide a non-empty substring copied verbatim from the source (never {{c1::}} empty).
- Cloze numbers start at c1 and increase by 1 (c1, c2, c3, ...). At most MAX_CLOZES clozes.
- Cloze spans must not overlap.
- JSON string rules apply: escape double quotes and backslashes inside cloze_text."""

_USER_PLAIN_TEMPLATE = """{_FEW_SHOT}

Annotate SOURCE_TEXT with Anki clozes (same rules as above). Reply with ONLY the annotated
full text — no JSON, no preamble, first character must match the source.

MAX_CLOZES: {max_clozes}

---SOURCE_TEXT---
{text}
---END---"""


def _build_prompt_json(text: str, max_clozes: int) -> str:
    return f"""MAX_CLOZES: {max_clozes}

{_FEW_SHOT}

---SOURCE_TEXT (copy verbatim into JSON cloze_text; only add clozes)---
{text}
---END---

Respond with valid JSON only: one object with a single key "cloze_text" (string value).
No markdown fences."""


def _build_prompt_plain(text: str, max_clozes: int, strict: bool) -> str:
    base = _USER_PLAIN_TEMPLATE.format(_FEW_SHOT=_FEW_SHOT, text=text, max_clozes=max_clozes)
    if strict:
        base += (
            "\n\nSTRICT: If you output anything other than the annotated passage "
            "(no titles, no 'Here is', no colons on their own line before the text), it will be rejected."
        )
    return base


_RE_SPACED_C = re.compile(
    r"(?<!\{)\{\s*c\s*:\s*(\d+)\s*:\s*:\s*", re.IGNORECASE
)
_RE_SINGLE_BRACE = re.compile(r"(?<!\{)\{c(\d+)\s*::\s*", re.IGNORECASE)
_RE_LOOSE_OPEN = re.compile(r"\{\{\s*c\s*(\d+)\s*:\s*:\s*")


def _repair_marker_shapes(text: str) -> str:
    """Best-effort fixes for common malformed outputs (CSV-derived patterns)."""
    t = text
    t = _RE_SPACED_C.sub(r"{{c\1::", t)
    t = _RE_SINGLE_BRACE.sub(r"{{c\1::", t)
    t = _RE_LOOSE_OPEN.sub(r"{{c\1::", t)
    # Close tags written as single } at end of cloze: ...foo}  when should be }}
    # Only fix when preceded by typical cloze content end (risky) — skip global single }

    # Remove empty clozes {{cN::}} — they never validate; stripping may help text fidelity retry
    t = re.sub(r"\{\{\s*c(\d+)\s*::\s*\}\}", "", t)
    return t


def _fix_single_brace_clozes(text: str) -> str:
    """{cN::...}  →  {{cN::...}} when inner ... has no }."""
    pat = re.compile(r"(?<!\{)\{c(\d+)\s*::\s*([^}]*)\}(?!\})")
    prev, cur = None, text
    while prev != cur:
        prev = cur
        cur = pat.sub(r"{{c\1::\2}}", cur)
    return cur


def _fix_double_open_single_close(text: str) -> str:
    """{{cN::...}  →  {{cN::...}} (missing one closing brace)."""
    pat = re.compile(r"\{\{c(\d+)\s*::\s*([^}]*)\}(?!\})")
    prev, cur = None, text
    while prev != cur:
        prev = cur
        cur = pat.sub(r"{{c\1::\2}}", cur)
    return cur


def _strip_leading_garbage_lines(text: str) -> str:
    lines = text.split("\n")
    i = 0
    junk = (
        "cloze tag",
        "annotated text",
        "full text",
        "text with",
        "important term",
        "output",
        "here is",
        "here are",
        "below is",
        "following text",
        "sure!",
        "certainly",
        "```json",
    )
    cloze_hint = re.compile(r"\{\{?\s*c\d+\s*::", re.IGNORECASE)
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if cloze_hint.search(s):
            break
        low = s.lower()
        if any(k in low for k in junk):
            i += 1
            continue
        if len(s) < 140 and s.endswith(":") and not cloze_hint.search(s):
            i += 1
            continue
        break
    return "\n".join(lines[i:])


def _clean_plain_output(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"^```[^\n]*\n", "", text)
    text = re.sub(r"\n```$", "", text)
    text = text.strip()
    low = text.lower()
    for p in (
        "annotated text:",
        "output:",
        "result:",
        "answer:",
        "here is",
        "here are",
    ):
        if low.startswith(p):
            text = text[len(p) :].lstrip()
            low = text.lower()
            break
    text = _strip_leading_garbage_lines(text)
    text = _repair_marker_shapes(text)
    text = _fix_double_open_single_close(text)
    text = _fix_single_brace_clozes(text)
    return text.strip()


def _parse_json_response(raw: str) -> str | None:
    """Extract cloze_text from Ollama structured response; return None if invalid."""
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```\w*\n?", "", s)
        s = re.sub(r"\n```$", "", s).strip()
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        # Try largest {...} slice
        start, end = s.find("{"), s.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(s[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    ct = data.get("cloze_text")
    if not isinstance(ct, str) or not ct.strip():
        return None
    return _clean_plain_output(ct)


async def generate_cloze_for_chunk(
    chunk: Chunk,
    settings: GenerateSettings,
) -> ClozeCard:
    """
    Call Ollama to generate a cloze card for one chunk.
    Retries up to settings.max_retries times with stricter prompts.
    """
    card = ClozeCard(
        card_id=new_id(),
        chunk_id=chunk.chunk_id,
        original_text=chunk.normalized_text or chunk.text,
        cloze_text="",
        retry_count=0,
    )
    original = card.original_text
    last_output = ""

    max_clozes = settings.max_clozes
    n_predict = max(4096, min(len(original) * 4, 32000))

    async with httpx.AsyncClient(timeout=120.0) as client:
        for attempt in range(settings.max_retries + 1):
            card.retry_count = attempt
            use_json = attempt == 0 or attempt == 1
            strict_plain = attempt >= 2

            if use_json:
                prompt = _build_prompt_json(original, max_clozes)
                system = _SYSTEM_JSON
                fmt: dict | str | None = CLOZE_JSON_SCHEMA
            else:
                prompt = _build_prompt_plain(original, max_clozes, strict_plain)
                system = None
                fmt = None

            payload: dict = {
                "model": settings.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.05,
                    "top_p": 0.85,
                    "num_predict": n_predict,
                },
            }
            if system:
                payload["system"] = system
            if fmt is not None:
                payload["format"] = fmt

            fmt_sent = fmt is not None

            try:
                resp = await client.post(
                    f"{settings.ollama_url}/api/generate",
                    json=payload,
                )
                used_structured = fmt_sent
                if resp.status_code == 400 and fmt_sent:
                    p2 = {
                        "model": settings.model,
                        "prompt": _build_prompt_plain(original, max_clozes, strict=True),
                        "stream": False,
                        "options": payload["options"],
                    }
                    resp = await client.post(
                        f"{settings.ollama_url}/api/generate",
                        json=p2,
                    )
                    used_structured = False
                resp.raise_for_status()
                data = resp.json()
                raw_output = data.get("response", "") or ""

                if used_structured:
                    parsed = _parse_json_response(raw_output)
                    last_output = parsed if parsed is not None else _clean_plain_output(raw_output)
                else:
                    last_output = _clean_plain_output(raw_output)

            except httpx.HTTPError as exc:
                card.validation_result.valid = False
                card.validation_result.errors.append(f"Ollama request failed: {exc}")
                card.cloze_text = original
                card.review_severity = ReviewSeverity.HIGH
                return card

            card.cloze_text = last_output
            card = validate_cloze_card(card, settings.max_hidden_proportion)
            if card.validation_result.valid:
                return card

            if attempt < settings.max_retries:
                continue

    card.cloze_text = last_output if last_output else original
    card.validation_result.errors.append(
        f"Validation failed after {settings.max_retries} retries. Manual review required."
    )
    card.review_severity = ReviewSeverity.HIGH
    return card
