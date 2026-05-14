"""
Deterministic normalization pipeline.
Handles text, markdown, and math normalization.
The canonical source text is never mutated; we work on a copy.
"""
from __future__ import annotations
import re
import unicodedata
from models import MathSpan


# ── Quote tables ──────────────────────────────────────────────────────────────

QUOTE_MAP = {
    "\u2018": "'",  # '
    "\u2019": "'",  # '
    "\u201a": "'",  # ‚
    "\u201b": "'",  # ‛
    "\u201c": '"',  # "
    "\u201d": '"',  # "
    "\u201e": '"',  # „
    "\u201f": '"',  # ‟
    "\u2032": "'",  # ′
    "\u2033": '"',  # ″
    "\u00ab": '"',  # «
    "\u00bb": '"',  # »
}

DASH_MAP = {
    "\u2013": "-",  # –
    "\u2014": "--", # —
    "\u2015": "--", # ―
}

# ── Math patterns ─────────────────────────────────────────────────────────────

INLINE_MATH_RE = re.compile(r'\$(?!\$)(.+?)\$', re.DOTALL)
DISPLAY_MATH_RE = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)
LATEX_ENV_RE = re.compile(r'\\begin\{(equation|align|math|gather)\*?\}(.+?)\\end\{\1\*?\}', re.DOTALL)

# Simple Greek / symbol substitutions for plain text
GREEK_MAP = {
    r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
    r'\epsilon': 'ε', r'\zeta': 'ζ', r'\eta': 'η', r'\theta': 'θ',
    r'\iota': 'ι', r'\kappa': 'κ', r'\lambda': 'λ', r'\mu': 'μ',
    r'\nu': 'ν', r'\xi': 'ξ', r'\pi': 'π', r'\rho': 'ρ',
    r'\sigma': 'σ', r'\tau': 'τ', r'\upsilon': 'υ', r'\phi': 'φ',
    r'\chi': 'χ', r'\psi': 'ψ', r'\omega': 'ω',
    r'\Delta': 'Δ', r'\Gamma': 'Γ', r'\Lambda': 'Λ', r'\Omega': 'Ω',
    r'\Phi': 'Φ', r'\Pi': 'Π', r'\Sigma': 'Σ', r'\Theta': 'Θ',
    r'\infty': '∞', r'\cdot': '·', r'\times': '×', r'\div': '÷',
    r'\pm': '±', r'\leq': '≤', r'\geq': '≥', r'\neq': '≠',
    r'\approx': '≈', r'\in': '∈', r'\notin': '∉', r'\subset': '⊂',
    r'\cup': '∪', r'\cap': '∩', r'\sqrt': '√', r'\partial': '∂',
    r'\nabla': '∇', r'\sum': 'Σ', r'\int': '∫', r'\prod': 'Π',
}


def normalize_text(raw: str) -> tuple[str, list[MathSpan]]:
    """
    Full normalization pipeline.
    Returns (normalized_text, math_spans).
    """
    text = raw

    # 1. Unicode normalization (NFC)
    text = unicodedata.normalize("NFC", text)

    # 2. Quote normalization
    for src, dst in QUOTE_MAP.items():
        text = text.replace(src, dst)
    for src, dst in DASH_MAP.items():
        text = text.replace(src, dst)

    # 3. Whitespace normalization – collapse runs of spaces (not newlines)
    text = re.sub(r'[ \t]+', ' ', text)

    # 4. Newline normalization – CRLF → LF, strip trailing spaces per line
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r' +\n', '\n', text)

    # 5. Collapse 3+ blank lines → 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 6. Strip leading/trailing whitespace
    text = text.strip()

    # 7. Markdown normalization
    text = _normalize_markdown(text)

    # 8. Math extraction & normalization
    text, math_spans = _extract_math(text)

    return text, math_spans


def _normalize_markdown(text: str) -> str:
    lines = text.split('\n')
    out = []
    for line in lines:
        # Headings: ensure single space after #
        m = re.match(r'^(#{1,6})\s*(.*)', line)
        if m:
            line = f"{m.group(1)} {m.group(2).strip()}"

        # List items: normalize to '- ' or '* '
        m = re.match(r'^(\s*)[-*+]\s+(.*)', line)
        if m:
            line = f"{m.group(1)}- {m.group(2)}"

        # Ordered lists: normalize spacing
        m = re.match(r'^(\s*)(\d+)[.)]\s+(.*)', line)
        if m:
            line = f"{m.group(1)}{m.group(2)}. {m.group(3)}"

        out.append(line)
    return '\n'.join(out)


def _normalize_latex_expr(expr: str) -> str:
    """Normalize a LaTeX math expression to a readable form."""
    result = expr.strip()
    for latex, symbol in GREEK_MAP.items():
        result = result.replace(latex, symbol)
    # Normalize superscripts: x^2 stays as-is for readability
    # Normalize fractions: \frac{a}{b} → a/b
    result = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', result)
    # Remove \left, \right
    result = re.sub(r'\\(?:left|right)[.()\[\]|]', '', result)
    # Clean up extra braces
    result = re.sub(r'\{([^{}]+)\}', r'\1', result)
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def _extract_math(text: str) -> tuple[str, list[MathSpan]]:
    """
    Find math spans, normalize them, and record positions.
    We replace display math before inline to avoid double-matching $$.
    """
    spans: list[MathSpan] = []
    offset = 0  # tracks how much the string has grown/shrunk

    def replace_math(m: re.Match, inline: bool) -> str:
        nonlocal offset
        original = m.group(0)
        inner = m.group(1)
        normalized = _normalize_latex_expr(inner)
        # We keep the original delimiter but store the span
        start = m.start() + offset
        end = start + len(original)
        spans.append(MathSpan(
            start=start,
            end=end,
            original=original,
            normalized=normalized,
            inline=inline,
        ))
        return original  # keep text unchanged; normalization is metadata

    # Display math first
    text = DISPLAY_MATH_RE.sub(lambda m: replace_math(m, False), text)
    text = INLINE_MATH_RE.sub(lambda m: replace_math(m, True), text)
    text = LATEX_ENV_RE.sub(lambda m: replace_math(m, False), text)

    return text, spans


def strip_markdown(text: str) -> str:
    """Return plain text suitable for comparison (remove markdown syntax)."""
    # Remove headings markers
    text = re.sub(r'^#{1,6} ', '', text, flags=re.MULTILINE)
    # Remove bold/italic
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}([^_]+)_{1,3}', r'\1', text)
    # Remove inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove links
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove list bullets
    text = re.sub(r'^\s*[-*] ', '', text, flags=re.MULTILINE)
    return text
