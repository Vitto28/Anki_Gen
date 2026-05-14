from __future__ import annotations
import uuid
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


def new_id() -> str:
    return str(uuid.uuid4())


# ── Enums ──────────────────────────────────────────────────────────────────

class InputMethod(str, Enum):
    FILE = "file"
    PASTE = "paste"


class ValidationState(str, Enum):
    PENDING = "pending"
    VALID = "valid"
    WARNING = "warning"
    FAILED = "failed"


class ReviewSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ── Validation ──────────────────────────────────────────────────────────────

class ValidationResult(BaseModel):
    valid: bool = True
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# ── Math span ───────────────────────────────────────────────────────────────

class MathSpan(BaseModel):
    start: int
    end: int
    original: str
    normalized: str
    inline: bool = True  # True = inline, False = display


# ── Chunk ───────────────────────────────────────────────────────────────────

class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=new_id)
    source_start: int = 0
    source_end: int = 0
    text: str = ""
    normalized_text: str = ""
    math_spans: list[MathSpan] = Field(default_factory=list)
    validation_state: ValidationState = ValidationState.PENDING
    selected: bool = True   # user may deselect chunks


# ── ClozeCard ───────────────────────────────────────────────────────────────

class ClozeCard(BaseModel):
    card_id: str = Field(default_factory=new_id)
    chunk_id: str = ""
    original_text: str = ""
    cloze_text: str = ""
    retry_count: int = 0
    validation_result: ValidationResult = Field(default_factory=ValidationResult)
    review_severity: ReviewSeverity = ReviewSeverity.LOW
    tags: list[str] = Field(default_factory=lambda: ["auto-generated"])


# ── Document ─────────────────────────────────────────────────────────────────

class Document(BaseModel):
    document_id: str = Field(default_factory=new_id)
    filename: str = ""
    input_method: InputMethod = InputMethod.PASTE
    raw_text: str = ""
    normalized_text: str = ""
    chunks: list[Chunk] = Field(default_factory=list)


# ── Request / Response shapes ────────────────────────────────────────────────

class IngestTextRequest(BaseModel):
    text: str
    filename: str = "pasted_text"


class ChunkSettings(BaseModel):
    max_sentences: int = 5
    max_chars: int = 500
    min_sentences: int = 2


class GenerateSettings(BaseModel):
    model: str = "llama3.2:3b"
    max_clozes: int = 5
    max_hidden_proportion: float = 0.35
    max_retries: int = 3
    ollama_url: str = "http://localhost:11434"


class UpdateChunkRequest(BaseModel):
    text: Optional[str] = None
    selected: Optional[bool] = None


class UpdateCardRequest(BaseModel):
    cloze_text: str


class AppState(BaseModel):
    """Singleton in-memory state for the prototype."""
    document: Optional[Document] = None
    cards: list[ClozeCard] = Field(default_factory=list)
    chunk_settings: ChunkSettings = Field(default_factory=ChunkSettings)
    generate_settings: GenerateSettings = Field(default_factory=GenerateSettings)
