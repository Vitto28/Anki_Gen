# Anki Cloze Generator

A local-first Anki flashcard generation application that inserts Cloze deletions into your educational material using a lightweight local LLM. No cloud APIs required.

## Architecture

```
anki-cloze-gen/
├── backend/
│   ├── main.py          # FastAPI application & all endpoints
│   ├── models.py        # Pydantic data models
│   ├── normalizer.py    # Deterministic text/markdown/math normalization
│   ├── chunker.py       # Boundary-scoring chunker
│   ├── cloze_gen.py     # Ollama LLM integration with retry logic
│   ├── validator.py     # Deterministic validation pipeline
│   └── requirements.txt
└── frontend/
    └── index.html       # Single-file React app (no build step)
```

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.11+ | |
| pip | latest | |
| Ollama | latest | https://ollama.ai |
| A local LLM | — | Recommended: `llama3.2:3b` or `qwen2.5:3b` |

## Quick Start

### 1. Install Ollama and pull a model

```bash
# Install Ollama (macOS)
brew install ollama

# Or download from https://ollama.ai/download

# Start Ollama
ollama serve

# Pull a recommended model (in a new terminal)
ollama pull llama3.2:3b

# Other good options:
# ollama pull qwen2.5:3b
# ollama pull phi3:mini
# ollama pull gemma3:4b
```

### 2. Set up the Python backend

```bash
# Navigate to the backend directory
cd anki-cloze-gen/backend

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the backend server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.
You can explore the interactive docs at `http://localhost:8000/docs`.

### 3. Open the frontend

Simply open the frontend HTML file in your browser:

```bash
open anki-cloze-gen/frontend/index.html
```

Or serve it with Python's built-in server for a cleaner experience:

```bash
cd anki-cloze-gen/frontend
python3 -m http.server 3000
# Then open http://localhost:3000 in your browser
```

---

## Usage Walkthrough

### Stage 1 — Input
- **Paste text**: Copy your notes, textbook sections, or any educational content directly into the text area.
- **Upload file**: Drag & drop or browse for a `.md`, `.txt`, or `.pdf` file.
- Click **Continue** to ingest and normalize the document.

### Stage 2 — Chunk Review
The chunker uses deterministic boundary scoring to split your text into self-contained, contextually meaningful segments.

- Left panel shows the original source text with chunks highlighted.
- Right panel lists all generated chunks.
- Click any chunk to select it and reveal **Edit / Deselect / Delete** controls.
- Adjust chunk settings (max sentences, max chars) and click **Re-chunk** to regenerate.
- Deselect chunks you don't want to generate cards for.
- Click **Generate Cards** when ready.

### Stage 3 — Generation
- Select your Ollama model from the dropdown (auto-populated from your Ollama instance).
- Configure max clozes per card, hidden proportion, and retry count.
- Click **Start Generation** — cards stream in as they complete.
- The LLM only inserts `{{cN::...}}` syntax; it never modifies your source text.
- Failed cards are retried up to the configured limit with stricter prompts.

### Stage 4 — Card Review
Cards are displayed with severity indicators:

| Severity | Meaning |
|----------|---------|
| ✓ Low | Passed all validation checks |
| ⚠ Medium | Usable but worth reviewing (many clozes, dense content) |
| ✗ High | Validation failure — text mismatch, malformed syntax |

For each card you can:
- **Edit** the cloze text manually (re-validates on save)
- **Regen** to request a new LLM generation for that chunk
- **Delete** to remove the card
- Filter by severity using the tab bar

### Stage 5 — Export
Downloads an Anki-compatible CSV file.

**Import into Anki:**
1. Open Anki → File → Import
2. Select `anki_cloze_cards.csv`
3. Set Note Type to **Cloze**
4. Map Field 1 → Text, Field 2 → Tags
5. Click Import

---

## API Reference

All endpoints are available at `http://localhost:8000`. Interactive docs: `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/ingest/text` | Ingest pasted text |
| POST | `/api/ingest/file` | Ingest uploaded file |
| GET | `/api/document` | Get current document info |
| POST | `/api/chunks/generate` | Run chunker |
| GET | `/api/chunks` | List chunks |
| PUT | `/api/chunks/{id}` | Update chunk text or selection |
| DELETE | `/api/chunks/{id}` | Delete chunk |
| POST | `/api/chunks/split/{id}` | Split chunk at character offset |
| POST | `/api/chunks/merge` | Merge multiple chunks |
| POST | `/api/generate` | Generate all cards (streaming) |
| POST | `/api/generate/single/{chunk_id}` | Regenerate one card |
| GET | `/api/cards` | List all cards |
| PUT | `/api/cards/{id}` | Update card cloze text |
| DELETE | `/api/cards/{id}` | Delete card |
| GET | `/api/export/csv` | Download Anki CSV |
| GET | `/api/ollama/status` | Check Ollama connectivity |

---

## Configuration

### Chunk Settings
| Setting | Default | Description |
|---------|---------|-------------|
| `max_sentences` | 5 | Max sentences per chunk |
| `max_chars` | 500 | Max characters per chunk |
| `min_sentences` | 2 | Min sentences before considering a split |

### Generation Settings
| Setting | Default | Description |
|---------|---------|-------------|
| `model` | `llama3.2:3b` | Ollama model name |
| `max_clozes` | 5 | Max cloze deletions per card |
| `max_hidden_proportion` | 0.35 | Max fraction of text that can be hidden |
| `max_retries` | 3 | Retry attempts before flagging for manual review |
| `ollama_url` | `http://localhost:11434` | Ollama server URL |

---

## Core Design Principles

- **Deterministic First**: Chunking, normalization, validation, and export are all fully deterministic algorithms.
- **Minimal LLM Responsibility**: The model performs exactly one task — inserting `{{cN::...}}` syntax. It never rewrites, paraphrases, or generates new content.
- **Validation Over Trust**: Every generated card is validated against the original source text. Text mismatches, malformed syntax, and overlapping deletions are all caught and flagged.
- **Stateless Generation**: Each chunk is sent to the LLM independently with no prior context. This improves reproducibility and inference speed.
- **Human Reviewable**: Every stage exposes structured, editable output before proceeding.

---

## Troubleshooting

**"Ollama not reachable"**
- Ensure `ollama serve` is running in a terminal.
- Check that the port 11434 is not blocked by a firewall.
- Try `curl http://localhost:11434/api/tags` — you should see a JSON response.

**"No models available"**
- Run `ollama pull llama3.2:3b` to download a model.
- After pulling, click **Refresh** in the Generation stage.

**CORS errors in the browser**
- Make sure you're accessing the frontend via `http://` not `file://` when the backend is running.
- Serve the frontend with `python3 -m http.server 3000`.

**PDF extraction not working**
- Install PyMuPDF: `pip install PyMuPDF`
- Or install pdfplumber as a fallback: `pip install pdfplumber`
- Only PDFs with selectable text are supported (not scanned images).

**All cards flagged as High severity**
- Your model may be modifying the source text. Try a larger model (`qwen2.5:3b`, `llama3.2:3b`).
- Reduce `max_clozes` to give the model less to do.
- The text mismatch validator catches any word-level edits by the LLM.

---

## Development

To add new features:

- **New input types**: Add a parser in `main.py` → `ingest_file`
- **Custom chunking logic**: Modify `chunker.py` → `_score_boundary`
- **Different prompt strategies**: Edit `cloze_gen.py` → `_build_prompt`
- **Additional validation rules**: Extend `validator.py` → `validate_card`
- **New export formats**: Add endpoint in `main.py` alongside `/api/export/csv`

---

## License

MIT
