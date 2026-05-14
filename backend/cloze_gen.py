"""
LLM-assisted cloze generation via Ollama.

The LLM is given one chunk at a time and instructed only to insert
{{cN::...}} syntax — no paraphrasing, no rewriting.

Retry logic uses progressively stricter prompts.
"""
from __future__ import annotations
import re
import httpx
from models import Chunk, ClozeCard, GenerateSettings, new_id
from validator import validate_cloze_card

# ── Prompt templates ──────────────────────────────────────────────────────────

BASE_PROMPT = """\
TASK:
You are an Anki flashcard annotator. Insert cloze deletion tags into the important concepts in the text below.

STRICT RULES:
- Do NOT modify, paraphrase, reorder, or rewrite any word in the text.
- ONLY insert {{cN::...}} tags around important terms or concepts.
- Preserve all punctuation, capitalization, whitespace, and formatting.
- Use sequential numbers starting at c1.
- Maximum {max_clozes} cloze deletions.
- Do not create overlapping clozes.
- Do not hide trivial words (articles, prepositions).
- Output ONLY the annotated text. No explanation. No preamble.

TEXT:
{text}

ANNOTATED TEXT:"""

RETRY_PROMPT = """\
TASK:
Annotate the following text with Anki cloze deletion tags.

CRITICAL RULES — READ CAREFULLY:
- YOU MUST NOT CHANGE A SINGLE WORD OF THE SOURCE TEXT.
- YOU MUST NOT CHANGE A SINGLE PUNCTUATION MARK.
- YOU MUST NOT CHANGE WHITESPACE.
- ONLY insert {{cN::...}} tags. Nothing else.
- Maximum {max_clozes} cloze deletions.
- Sequential numbering starting at c1.
- No overlapping clozes.
- If you are unsure, insert fewer clozes, not more.
- Output ONLY the annotated text. Nothing else.

SOURCE TEXT (DO NOT MODIFY):
{text}

OUTPUT (annotated text only):"""

FINAL_RETRY_PROMPT = """\
Wrap at most {max_clozes} important terms in the following text with {{c1::term}}, {{c2::term}}, etc.
Do not change any other part of the text.
Output the full text with cloze tags only.

{text}"""


def _build_prompt(text: str, max_clozes: int, retry_count: int) -> str:
    if retry_count == 0:
        return BASE_PROMPT.format(text=text, max_clozes=max_clozes)
    elif retry_count == 1:
        return RETRY_PROMPT.format(text=text, max_clozes=max_clozes)
    else:
        return FINAL_RETRY_PROMPT.format(text=text, max_clozes=max_clozes)


def _clean_llm_output(raw: str, original: str) -> str:
    """
    Best-effort cleanup of LLM output.
    Remove common prefixes/suffixes, extract the annotated text.
    """
    text = raw.strip()

    # Remove markdown code blocks if the model wrapped its output
    text = re.sub(r'^```[^\n]*\n', '', text)
    text = re.sub(r'\n```$', '', text)
    text = text.strip()

    # If the model prefixed with "ANNOTATED TEXT:" or similar
    for prefix in ['ANNOTATED TEXT:', 'OUTPUT:', 'RESULT:', 'Answer:', 'Here is']:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    return text


async def generate_cloze_for_chunk(
    chunk: Chunk,
    settings: GenerateSettings,
) -> ClozeCard:
    """
    Call Ollama to generate a cloze card for one chunk.
    Retries up to settings.max_retries times with stricter prompts.
    Returns a ClozeCard (may be flagged for manual review if all retries fail).
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

    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(settings.max_retries + 1):
            card.retry_count = attempt
            prompt = _build_prompt(original, settings.max_clozes, attempt)

            try:
                resp = await client.post(
                    f"{settings.ollama_url}/api/generate",
                    json={
                        "model": settings.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,   # low temp for determinism
                            "top_p": 0.9,
                            "num_predict": max(256, len(original) * 2),
                        },
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                raw_output = data.get("response", "")
                last_output = _clean_llm_output(raw_output, original)

            except httpx.HTTPError as exc:
                card.validation_result.valid = False
                card.validation_result.errors.append(f"Ollama request failed: {exc}")
                card.cloze_text = original
                card.review_severity = "high"
                return card

            card.cloze_text = last_output
            card = validate_cloze_card(card, settings.max_hidden_proportion)

            if card.validation_result.valid:
                return card

            # If still invalid, retry with stricter prompt
            if attempt < settings.max_retries:
                continue

    # All retries exhausted — preserve last output and flag for manual review
    card.cloze_text = last_output if last_output else original
    card.validation_result.errors.append(
        f"Validation failed after {settings.max_retries} retries. Manual review required."
    )
    card.review_severity = "high"
    return card
