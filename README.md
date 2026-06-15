# OpenGraphKB

A vendor-agnostic knowledge base service combining **Neo4j** (graph) and **Qdrant** (vector) for hybrid RAG/GraphRAG retrieval.

## Features

- **Multi-format document ingestion** — PDF, DOCX, Markdown, plain text (extensible)
- **Configurable chunking** — fixed-size or sentence-based strategies
- **Vector search** — Qdrant-powered semantic similarity retrieval
- **Graph knowledge** — Neo4j-powered entity extraction and relationship mapping
- **Hybrid retrieval** — combine vector and graph search with reranking
- **Schema-configurable** — entity types and relationship types are env-driven
- **LLM-agnostic** — OpenAI now, Anthropic-ready architecture
- **Fully dockerized** — `docker compose up` starts everything

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env with your LLM API key

# 2. Start the full stack
docker compose up --build

# 3. Access the API
open http://localhost:8000/docs       # OpenAPI docs
open http://localhost:7474            # Neo4j Browser
open http://localhost:6333/dashboard  # Qdrant Dashboard
```

## Ingesting Documents

```bash
# Upload a PDF
curl -X POST http://localhost:8000/api/documents \
  -F "file=@report.pdf" \
  -F "doc_id=quarterly_report" \
  -F "title=Q1 Report" \
  -F "source_type=article" \
  -F "company=Acme Corp"

# Check processing status
curl http://localhost:8000/api/documents/quarterly_report/status
```

## Searching

```bash
# RAG (vector only)
curl -X POST http://localhost:8000/api/search/rag \
  -H "Content-Type: application/json" \
  -d '{"queries": ["revenue growth"], "top_k": 5}'

# Hybrid (vector + graph expansion + reranking)
curl -X POST http://localhost:8000/api/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["AI developments"],
    "top_k": 10,
    "graph_options": {"depth": 2, "include_relationships": true},
    "debug": true
  }'
```

## API Endpoints

### Documents
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/documents` | Upload file + trigger ingestion |
| `GET` | `/api/documents` | List all documents |
| `GET` | `/api/documents/{id}` | Get document metadata |
| `GET` | `/api/documents/{id}/status` | Processing status |
| `DELETE` | `/api/documents/{id}` | Delete from all stores |
| `POST` | `/api/documents/{id}/reprocess` | Re-run full pipeline |
| `POST` | `/api/documents/{id}/reprocess/graph` | Re-run graph only |

### Search
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/search` | Unified search (profile-driven) |
| `POST` | `/api/search/rag` | Vector-only search |
| `POST` | `/api/search/graph-rag` | Vector + graph metadata |
| `POST` | `/api/search/hybrid` | Full hybrid with reranking |

### Graph
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/graph/stats` | Node/edge counts |
| `GET` | `/api/graph/entities` | List entities |
| `GET` | `/api/graph/documents` | List graph documents |
| `GET` | `/api/graph/documents/{id}` | Document graph detail |

### Operations
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/health/detailed` | Detailed stats |

## Configuration

All settings are driven by environment variables (see `.env.example`):

- **LLM_PROVIDER** — `openai` (Anthropic coming soon)
- **CHUNKING_STRATEGY** — `fixed_size` or `sentence`
- **ENTITY_TYPES** — Comma-separated list (default: Person, Organization, Project, Topic, Event, Technology, Location)
- **RELATIONSHIP_TYPES** — Comma-separated list (default: MENTIONS, BELONGS_TO, RELATES_TO, LEADS, PRODUCES, LOCATED_IN, PART_OF)

## Architecture

```
src/
├── parsers/       # Interface + implementations (PDF, DOCX, MD, TXT)
├── chunking/      # Interface + implementations (fixed-size, sentence)
├── llm/           # Interface + implementations (OpenAI, Anthropic-ready)
├── services/      # Neo4j, Qdrant, storage, extraction, pipeline, search
├── api/           # FastAPI route handlers
├── models/        # Pydantic schemas
├── core/          # Config, logging, exceptions
└── main.py        # App entry point
```

## Extending

### Add a new file format
1. Create `src/parsers/your_format.py` implementing `DocumentParser`
2. Register it in `ParserRegistry.default()`

### Add a new chunking strategy
1. Create `src/chunking/your_strategy.py` implementing `ChunkingStrategy`
2. Add a case in `chunking/factory.py`

### Add a new LLM provider
1. Create `src/llm/your_provider.py` implementing `LLMProvider`
2. Add a case in `llm/factory.py`

### Add new entity/relationship types
Update `ENTITY_TYPES` and `RELATIONSHIP_TYPES` in your `.env` — no code changes needed.

## Development

```bash
# Install dependencies
uv sync

# Run locally (needs Neo4j + Qdrant running)
uv run python -m main

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/
```
