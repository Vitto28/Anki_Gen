"""
Deterministic validation pipeline for Cloze cards.

Pipeline:
  1. Parse cloze spans
  2. Remove cloze syntax → plain text
  3. Normalize plain text
  4. Compare against original normalized text
  5. Verify numbering (c1, c2, c3 ...)
  6. Verify syntax balance (all {{ }} matched)
  7. Verify overlap rules
  8. Verify readability constraints
"""
from __future__ import annotations
import re
from models import ValidationResult, ReviewSeverity, ClozeCard
from normalizer import normalize_text

# Matches {{cN::content}} — N is the cloze number, content is the hidden text
CLOZE_RE = re.compile(r'\{\{c(\d+)::([^}]*)\}\}')


def _parse_cloze_spans(cloze_text: str) -> list[tuple[int, int, int, str]]:
    """
    Parse cloze syntax and return list of (start, end, number, content).
    """
    spans = []
    for m in CLOZE_RE.finditer(cloze_text):
        spans.append((m.start(), m.end(), int(m.group(1)), m.group(2)))
    return spans


def _remove_cloze_syntax(cloze_text: str) -> str:
    """Strip {{cN::...}} tags to recover the plain text."""
    return CLOZE_RE.sub(lambda m: m.group(2), cloze_text)


def _normalize_for_comparison(text: str) -> str:
    """Aggressive normalization for text comparison."""
    norm, _ = normalize_text(text)
    # Strip all whitespace and punctuation differences
    norm = re.sub(r'\s+', ' ', norm).strip().lower()
    return norm


def validate_card(original_text: str, cloze_text: str,
                  max_hidden_proportion: float = 0.35) -> tuple[ValidationResult, ReviewSeverity]:
    """
    Full validation pipeline. Returns (ValidationResult, ReviewSeverity).
    """
    result = ValidationResult()

    # ── 1. Check for empty input ─────────────────────────────────────────────
    if not cloze_text.strip():
        result.valid = False
        result.errors.append("Cloze text is empty.")
        return result, ReviewSeverity.HIGH

    # ── 2. Syntax balance check ──────────────────────────────────────────────
    open_count = cloze_text.count('{{')
    close_count = cloze_text.count('}}')
    if open_count != close_count:
        result.valid = False
        result.errors.append(
            f"Unbalanced cloze delimiters: {open_count} opening, {close_count} closing."
        )

    # ── 3. Parse spans ───────────────────────────────────────────────────────
    spans = _parse_cloze_spans(cloze_text)

    if not spans:
        result.valid = False
        result.errors.append("No cloze deletions found.")
        return result, ReviewSeverity.HIGH

    # ── 4. Numbering verification ────────────────────────────────────────────
    numbers = sorted({s[2] for s in spans})
    expected = list(range(1, len(numbers) + 1))
    # Allow gaps — Anki doesn't require contiguous numbers, but warn
    if numbers != expected:
        result.warnings.append(
            f"Cloze numbers are not contiguous: {numbers}. Expected {expected}."
        )

    # ── 5. Overlap check ─────────────────────────────────────────────────────
    sorted_spans = sorted(spans, key=lambda s: s[0])
    for i in range(len(sorted_spans) - 1):
        cur_end = sorted_spans[i][1]
        nxt_start = sorted_spans[i + 1][0]
        if cur_end > nxt_start:
            result.valid = False
            result.errors.append("Overlapping cloze deletions detected.")
            break

    # ── 6. Text fidelity check ───────────────────────────────────────────────
    plain_from_cloze = _remove_cloze_syntax(cloze_text)
    orig_norm = _normalize_for_comparison(original_text)
    cloze_norm = _normalize_for_comparison(plain_from_cloze)

    if orig_norm != cloze_norm:
        result.valid = False
        result.errors.append(
            "Text mismatch: cloze output does not match original after stripping syntax. "
            "The model may have modified source text."
        )

    # ── 7. Hidden proportion check ───────────────────────────────────────────
    total_len = len(plain_from_cloze)
    hidden_len = sum(len(s[3]) for s in spans)
    if total_len > 0:
        proportion = hidden_len / total_len
        if proportion > max_hidden_proportion:
            result.warnings.append(
                f"Hidden content proportion is {proportion:.0%}, "
                f"exceeding the recommended {max_hidden_proportion:.0%} limit."
            )

    # ── 8. Individual span length check ─────────────────────────────────────
    for span in spans:
        if len(span[3]) > 80:
            result.warnings.append(
                f"Cloze c{span[2]} span is very long ({len(span[3])} chars). "
                "Consider shorter spans."
            )

    # ── 9. Trivial deletion check ────────────────────────────────────────────
    for span in spans:
        content = span[3].strip()
        if len(content) <= 2:
            result.warnings.append(
                f"Cloze c{span[2]} hides very short content: '{content}'. "
                "May be a trivial deletion."
            )

    # ── Determine severity ───────────────────────────────────────────────────
    if not result.valid or result.errors:
        severity = ReviewSeverity.HIGH
    elif len(result.warnings) >= 2 or len(spans) > 6:
        severity = ReviewSeverity.MEDIUM
    else:
        severity = ReviewSeverity.LOW

    return result, severity


def validate_cloze_card(card: ClozeCard, max_hidden_proportion: float = 0.35) -> ClozeCard:
    """Validate a ClozeCard in place and return it."""
    result, severity = validate_card(
        card.original_text, card.cloze_text, max_hidden_proportion
    )
    card.validation_result = result
    card.review_severity = severity
    return card
