"""
Deterministic chunk generation using boundary scoring.

Split signals:
  +5  paragraph break
  +4  markdown heading
  +3  sentence ending
  +3  blank line
  +2  list boundary

Do-not-split signals:
  -5  inline math continuation
  -4  very short sentence (< 30 chars)
  -3  connective continuation
  -3  same list item
"""
from __future__ import annotations
import re
from models import Chunk, ChunkSettings
from normalizer import normalize_text

# Connective words that indicate continuation
CONNECTIVES = {
    'however', 'therefore', 'furthermore', 'moreover', 'additionally',
    'consequently', 'nevertheless', 'meanwhile', 'similarly', 'likewise',
    'also', 'thus', 'hence', 'accordingly', 'subsequently', 'finally',
    'first', 'second', 'third', 'lastly', 'next', 'then', 'additionally',
    'in addition', 'on the other hand', 'in contrast', 'for example',
    'for instance', 'that is', 'in other words', 'as a result',
}

SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"])')
HEADING_RE = re.compile(r'^#{1,6} ', re.MULTILINE)
BLANK_LINE_RE = re.compile(r'\n\n+')
INLINE_MATH_RE = re.compile(r'\$[^$]+\$')
LIST_ITEM_RE = re.compile(r'^\s*[-*\d]', re.MULTILINE)


def _score_boundary(before: str, after: str) -> int:
    """Score the boundary between two adjacent text segments."""
    score = 0

    # Paragraph break (blank line between segments)
    if re.search(r'\n\n', before[-3:] + after[:3]):
        score += 5

    # Markdown heading in 'after'
    if HEADING_RE.match(after.lstrip()):
        score += 4

    # Sentence ending
    if re.search(r'[.!?]\s*$', before.rstrip()):
        score += 3

    # Blank line
    if '\n\n' in before[-5:]:
        score += 3

    # List boundary
    before_is_list = bool(LIST_ITEM_RE.match(before.lstrip()))
    after_is_list = bool(LIST_ITEM_RE.match(after.lstrip()))
    if before_is_list != after_is_list:
        score += 2

    # ── Do-not-split signals ──────────────────────────────────────────────

    # Inline math continuation (math straddles potential boundary)
    if INLINE_MATH_RE.search(before[-20:]) and not re.search(r'\$', before[-20:]):
        score -= 5

    # Very short sentence (after is short)
    stripped_after = after.strip()
    if stripped_after and len(stripped_after.split('.')[0]) < 30:
        score -= 4

    # Connective continuation
    first_word = stripped_after.lower().split()[0] if stripped_after.split() else ''
    first_phrase = ' '.join(stripped_after.lower().split()[:3])
    if first_word in CONNECTIVES or first_phrase in CONNECTIVES:
        score -= 3

    # Same list item (after is a continuation of the list)
    if before_is_list and after_is_list:
        score -= 3

    return score


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentence-like segments, respecting math and code."""
    # Protect inline math from splitting
    protected = INLINE_MATH_RE.sub(lambda m: m.group(0).replace('.', '⟨DOT⟩'), text)

    # Split on sentence boundaries
    parts = re.split(r'(?<=[.!?])\s+', protected)

    # Restore protected content
    parts = [p.replace('⟨DOT⟩', '.') for p in parts]

    # Also split on paragraph breaks
    result = []
    for part in parts:
        sub = re.split(r'\n\n+', part)
        result.extend(sub)

    return [s.strip() for s in result if s.strip()]


def generate_chunks(normalized_text: str, settings: ChunkSettings) -> list[Chunk]:
    """
    Generate chunks from normalized text using boundary scoring.
    Returns a list of Chunk objects.
    """
    sentences = _split_into_sentences(normalized_text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_start = 0
    pos = 0

    for i, sentence in enumerate(sentences):
        current_sentences.append(sentence)
        current_text = ' '.join(current_sentences)
        char_count = len(current_text)
        sent_count = len(current_sentences)

        # Check if we should flush a chunk
        should_split = False

        if sent_count >= settings.max_sentences:
            should_split = True
        elif char_count >= settings.max_chars:
            should_split = True
        elif sent_count >= settings.min_sentences and i + 1 < len(sentences):
            # Check boundary score with next sentence
            next_sentence = sentences[i + 1]
            score = _score_boundary(current_text, next_sentence)
            if score >= 3:
                should_split = True

        if should_split or i == len(sentences) - 1:
            # Find position in original text
            chunk_text = current_text.strip()
            if not chunk_text:
                current_sentences = []
                continue

            # Locate approximate position in normalized_text
            start = normalized_text.find(current_sentences[0], pos)
            if start == -1:
                start = pos
            end = start + len(chunk_text)
            pos = max(pos, end)

            norm_text, _ = normalize_text(chunk_text)
            chunk = Chunk(
                source_start=start,
                source_end=end,
                text=chunk_text,
                normalized_text=norm_text,
            )
            chunks.append(chunk)
            current_sentences = []

    return chunks
