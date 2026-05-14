"""
FastAPI backend for the Anki Cloze Card Generator.
All state is held in memory (prototype — no database required).
"""
from __future__ import annotations
import io
import csv
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import (
    AppState, Document, InputMethod, Chunk, ClozeCard,
    ChunkSettings, GenerateSettings,
    IngestTextRequest, UpdateChunkRequest, UpdateCardRequest,
    ValidationResult, ReviewSeverity, new_id,
)
from normalizer import normalize_text
from chunker import generate_chunks
from validator import validate_cloze_card
from cloze_gen import generate_cloze_for_chunk

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="Anki Cloze Generator", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton in-memory state
STATE = AppState()

# Generation progress tracking
GENERATION_STATUS: dict[str, dict] = {}  # session-level

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_document() -> Document:
    if STATE.document is None:
        raise HTTPException(status_code=404, detail="No document loaded. Ingest a document first.")
    return STATE.document


def _get_chunk(chunk_id: str) -> Chunk:
    doc = _get_document()
    for chunk in doc.chunks:
        if chunk.chunk_id == chunk_id:
            return chunk
    raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found.")


def _get_card(card_id: str) -> ClozeCard:
    for card in STATE.cards:
        if card.card_id == card_id:
            return card
    raise HTTPException(status_code=404, detail=f"Card {card_id} not found.")


# ── Ingest endpoints ──────────────────────────────────────────────────────────

@app.post("/api/ingest/text")
async def ingest_text(body: IngestTextRequest):
    """Accept pasted text and create a Document."""
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    norm_text, math_spans = normalize_text(body.text)
    doc = Document(
        filename=body.filename,
        input_method=InputMethod.PASTE,
        raw_text=body.text,
        normalized_text=norm_text,
    )
    STATE.document = doc
    STATE.cards = []
    return {
        "document_id": doc.document_id,
        "filename": doc.filename,
        "char_count": len(norm_text),
        "math_span_count": len(math_spans),
    }


@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """Accept a file upload (Markdown or PDF)."""
    filename = file.filename or "upload"
    content = await file.read()

    if filename.lower().endswith(".md") or filename.lower().endswith(".txt"):
        raw_text = content.decode("utf-8", errors="replace")
    elif filename.lower().endswith(".pdf"):
        raw_text = _extract_pdf_text(content)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use .md, .txt, or .pdf.")

    norm_text, math_spans = normalize_text(raw_text)
    doc = Document(
        filename=filename,
        input_method=InputMethod.FILE,
        raw_text=raw_text,
        normalized_text=norm_text,
    )
    STATE.document = doc
    STATE.cards = []
    return {
        "document_id": doc.document_id,
        "filename": doc.filename,
        "char_count": len(norm_text),
        "math_span_count": len(math_spans),
    }


def _extract_pdf_text(content: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=content, filetype="pdf")
        pages = []
        for page in doc:
            pages.append(page.get_text())
        return "\n\n".join(pages)
    except ImportError:
        # Fallback: pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n\n".join(pages)
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="PDF parsing requires PyMuPDF or pdfplumber. Install with: pip install PyMuPDF"
            )


# ── Document endpoints ────────────────────────────────────────────────────────

@app.get("/api/document")
async def get_document():
    doc = _get_document()
    return {
        "document_id": doc.document_id,
        "filename": doc.filename,
        "input_method": doc.input_method,
        "normalized_text": doc.normalized_text,
        "char_count": len(doc.normalized_text),
        "chunk_count": len(doc.chunks),
    }


# ── Chunk endpoints ───────────────────────────────────────────────────────────

@app.post("/api/chunks/generate")
async def generate_chunk_list(settings: Optional[ChunkSettings] = None):
    """Run the chunker and store chunks on the document."""
    doc = _get_document()
    if settings:
        STATE.chunk_settings = settings
    doc.chunks = generate_chunks(doc.normalized_text, STATE.chunk_settings)
    STATE.cards = []  # reset cards when re-chunking
    return {"chunk_count": len(doc.chunks), "chunks": [_chunk_to_dict(c) for c in doc.chunks]}


@app.get("/api/chunks")
async def list_chunks():
    doc = _get_document()
    return [_chunk_to_dict(c) for c in doc.chunks]


@app.put("/api/chunks/{chunk_id}")
async def update_chunk(chunk_id: str, body: UpdateChunkRequest):
    chunk = _get_chunk(chunk_id)
    if body.text is not None:
        chunk.text = body.text
        chunk.normalized_text, _ = normalize_text(body.text)
    if body.selected is not None:
        chunk.selected = body.selected
    return _chunk_to_dict(chunk)


@app.delete("/api/chunks/{chunk_id}")
async def delete_chunk(chunk_id: str):
    doc = _get_document()
    doc.chunks = [c for c in doc.chunks if c.chunk_id != chunk_id]
    return {"deleted": chunk_id}


@app.post("/api/chunks/split/{chunk_id}")
async def split_chunk(chunk_id: str, split_at: int):
    """Split a chunk at a character offset."""
    doc = _get_document()
    chunk = _get_chunk(chunk_id)
    text = chunk.text
    if split_at <= 0 or split_at >= len(text):
        raise HTTPException(status_code=400, detail="Invalid split position.")
    
    text_a = text[:split_at].strip()
    text_b = text[split_at:].strip()
    
    norm_a, _ = normalize_text(text_a)
    norm_b, _ = normalize_text(text_b)
    
    chunk_a = Chunk(
        source_start=chunk.source_start,
        source_end=chunk.source_start + split_at,
        text=text_a,
        normalized_text=norm_a,
    )
    chunk_b = Chunk(
        source_start=chunk.source_start + split_at,
        source_end=chunk.source_end,
        text=text_b,
        normalized_text=norm_b,
    )
    
    idx = next(i for i, c in enumerate(doc.chunks) if c.chunk_id == chunk_id)
    doc.chunks[idx:idx+1] = [chunk_a, chunk_b]
    return [_chunk_to_dict(chunk_a), _chunk_to_dict(chunk_b)]


@app.post("/api/chunks/merge")
async def merge_chunks(chunk_ids: list[str]):
    """Merge two or more chunks into one."""
    doc = _get_document()
    to_merge = [c for c in doc.chunks if c.chunk_id in chunk_ids]
    if len(to_merge) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 chunks to merge.")
    
    combined_text = " ".join(c.text for c in to_merge)
    norm_text, _ = normalize_text(combined_text)
    merged = Chunk(
        source_start=min(c.source_start for c in to_merge),
        source_end=max(c.source_end for c in to_merge),
        text=combined_text,
        normalized_text=norm_text,
    )
    
    # Replace first occurrence, remove rest
    first_idx = next(i for i, c in enumerate(doc.chunks) if c.chunk_id == chunk_ids[0])
    doc.chunks = [c for c in doc.chunks if c.chunk_id not in chunk_ids]
    doc.chunks.insert(first_idx, merged)
    return _chunk_to_dict(merged)


# ── Generation endpoints ──────────────────────────────────────────────────────

@app.post("/api/generate")
async def generate_cards(settings: Optional[GenerateSettings] = None):
    """
    Generate cloze cards for all selected chunks.
    Returns a streaming JSON response with progress updates.
    """
    doc = _get_document()
    if settings:
        STATE.generate_settings = settings

    selected = [c for c in doc.chunks if c.selected]
    if not selected:
        raise HTTPException(status_code=400, detail="No chunks selected for generation.")

    async def event_stream():
        STATE.cards = []
        total = len(selected)
        yield f'{{"type":"start","total":{total}}}\n'
        
        for i, chunk in enumerate(selected):
            try:
                card = await generate_cloze_for_chunk(chunk, STATE.generate_settings)
                STATE.cards.append(card)
                import json
                payload = json.dumps({
                    "type": "progress",
                    "index": i + 1,
                    "total": total,
                    "card": _card_to_dict(card),
                })
                yield f"{payload}\n"
            except Exception as exc:
                import json
                yield json.dumps({
                    "type": "error",
                    "index": i + 1,
                    "chunk_id": chunk.chunk_id,
                    "error": str(exc),
                }) + "\n"

        import json
        yield json.dumps({"type": "done", "card_count": len(STATE.cards)}) + "\n"

    return StreamingResponse(event_stream(), media_type="text/plain")


@app.post("/api/generate/single/{chunk_id}")
async def regenerate_chunk(chunk_id: str, settings: Optional[GenerateSettings] = None):
    """Regenerate the cloze card for a single chunk."""
    chunk = _get_chunk(chunk_id)
    if settings:
        STATE.generate_settings = settings
    card = await generate_cloze_for_chunk(chunk, STATE.generate_settings)
    # Replace existing card for this chunk
    STATE.cards = [c for c in STATE.cards if c.chunk_id != chunk_id]
    STATE.cards.append(card)
    return _card_to_dict(card)


# ── Card endpoints ────────────────────────────────────────────────────────────

@app.get("/api/cards")
async def list_cards():
    return [_card_to_dict(c) for c in STATE.cards]


@app.put("/api/cards/{card_id}")
async def update_card(card_id: str, body: UpdateCardRequest):
    """Update cloze text and re-validate."""
    card = _get_card(card_id)
    card.cloze_text = body.cloze_text
    card = validate_cloze_card(card, STATE.generate_settings.max_hidden_proportion)
    return _card_to_dict(card)


@app.delete("/api/cards/{card_id}")
async def delete_card(card_id: str):
    STATE.cards = [c for c in STATE.cards if c.card_id != card_id]
    return {"deleted": card_id}


# ── Settings endpoints ────────────────────────────────────────────────────────

@app.get("/api/settings")
async def get_settings():
    return {
        "chunk_settings": STATE.chunk_settings.model_dump(),
        "generate_settings": STATE.generate_settings.model_dump(),
    }


@app.put("/api/settings/chunk")
async def update_chunk_settings(settings: ChunkSettings):
    STATE.chunk_settings = settings
    return settings


@app.put("/api/settings/generate")
async def update_generate_settings(settings: GenerateSettings):
    STATE.generate_settings = settings
    return settings


# ── Export endpoints ──────────────────────────────────────────────────────────

@app.get("/api/export/csv")
async def export_csv():
    """Export all cards as Anki-compatible CSV."""
    if not STATE.cards:
        raise HTTPException(status_code=400, detail="No cards to export.")

    output = io.StringIO()
    writer = csv.writer(output)
    # Anki CSV format: Text, Tags (no header row)
    for card in STATE.cards:
        tags = " ".join(card.tags)
        writer.writerow([card.cloze_text, tags])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=anki_cloze_cards.csv"},
    )


# ── Ollama health check ───────────────────────────────────────────────────────

@app.get("/api/ollama/status")
async def check_ollama():
    """Check if Ollama is reachable and list available models."""
    import httpx
    url = STATE.generate_settings.ollama_url
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            return {"status": "ok", "models": models, "url": url}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "url": url}


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ── Serialization helpers ─────────────────────────────────────────────────────

def _chunk_to_dict(c: Chunk) -> dict:
    return {
        "chunk_id": c.chunk_id,
        "source_start": c.source_start,
        "source_end": c.source_end,
        "text": c.text,
        "normalized_text": c.normalized_text,
        "validation_state": c.validation_state,
        "selected": c.selected,
        "char_count": len(c.text),
        "sentence_count": len([s for s in c.text.split('.') if s.strip()]),
    }


def _card_to_dict(c: ClozeCard) -> dict:
    return {
        "card_id": c.card_id,
        "chunk_id": c.chunk_id,
        "original_text": c.original_text,
        "cloze_text": c.cloze_text,
        "retry_count": c.retry_count,
        "review_severity": c.review_severity,
        "tags": c.tags,
        "validation": {
            "valid": c.validation_result.valid,
            "warnings": c.validation_result.warnings,
            "errors": c.validation_result.errors,
        },
    }
