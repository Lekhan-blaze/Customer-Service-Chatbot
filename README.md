# Customer Service Chatbot (Extended)

This project extends my training project, a customer service chatbot (Streamlit + LangChain + FAISS + Gemini, FAQ dataset in dataset/dataset.csv), with six new capabilities.
It adds production support workflows including sentiment escalation, ticketing, versioned knowledge bases, secure retrieval, file processing, and multilingual sessions.

## Training Project

- Streamlit web interface with a question-answering input box and knowledgebase creation trigger.
- LangChain RetrievalQA chain backed by a local FAISS vector store and Google Gemini (gemini-2.0-flash).
- Static FAQ dataset of 76 question-and-answer pairs in dataset/dataset.csv.
- Prompt template enforcing answers based strictly on retrieved FAQ context, falling back to "I don't know."

## What I Added

| Module | What it does in one line | Main folder |
|---|---|---|
| Sentiment and escalation | Multi-label sentiment scoring, tone adjustment, 15-minute negativity timer, and immediate risk escalation | support_bot/sentiment, support_bot/escalation |
| Tickets and SLA | Structured ticket creation, multi-factor priority, skill and after-hours routing, duplicate linking, and SLA tracking | support_bot/tickets |
| Knowledge-base pipeline | Policy document ingestion, automated quality gate, semantic chunking, maintenance window scheduling, and admin rollback | support_bot/knowledge |
| Secure RAG | Pre-retrieval RBAC filtering, temporal policy routing by date, citation grounding, and prompt-injection defense | support_bot/rag |
| Files and OCR | Document upload validation, OCR field extraction, discrepancy detection against claims, and automated retention purge | support_bot/files |
| Multilingual sessions | Language and intent detection across 4 languages, entity preservation, context tracking, and session restore/expiry | support_bot/conversation |

## Quick Start

```bash
# 1. Create and activate a virtual environment (Python 3.11 required)
python3.11 -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY to your Gemini API key

# 4. Run the app
streamlit run main.py

# 5. Run the test suite
PYTHONPATH=. pytest
```

## Configuration

Copy `.env.example` to `.env` and fill in the required value:

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Google Gemini API key used by the LLM, sentiment analyzer, OCR, and conversation modules |

Runtime behaviour is controlled via `config/defaults.yaml`.
Any key in that file can be overridden at runtime through the admin panel in the Streamlit sidebar without restarting the app.
Key sections: `conversation`, `session`, `tickets`, `files`, `knowledge`, `rag`, `sentiment`, `escalation`.

## Demo Guide

The sidebar demo panel lets you switch user roles (customer, internal, admin), inspect active version metrics, view sentiment scores and escalation records, trigger background jobs, and reset sessions. The simulated clock allows freezing and advancing time in hours or minutes to test business hours, 15-minute negativity timeouts, and SLA deadlines without waiting.

1. **Calm Duplicate Payment**: Type "I was double charged on order ORD-1234. Please refund the duplicate payment." Expect immediate high-risk escalation despite calm tone.
2. **Consecutive Negative Escalation**: Send 3 consecutive frustrated messages. Expect empathetic tone adjustment and escalation to the human queue.
3. **Immediate High-Risk Bypass**: Type "I will take legal action against your company for data breach." Expect immediate escalation bypassing normal conversation flow.
4. **Historical Policy Query**: Type "What was the refund policy on 15 March 2025?" Expect retrieval of expired policy v1 active on that date.
5. **Current Policy Query**: Type "What is the current refund policy?" Expect retrieval of currently active policy v2 with source citation.
6. **Restricted Internal Guide**: Query "Show me the internal supervisor escalation guide" as customer (refused) vs internal (answered with citation).
7. **Ticket Creation**: Click "Create Ticket from Conversation" after discussing an order. Expect structured ticket extraction with assigned team and SLA deadline.
8. **Discrepancy Detection**: Enter claimed amount 14999 in the upload widget and upload a receipt showing 9999. Expect a clarification conflict prompt.

## Architecture

The codebase extends the baseline with a modular architecture centered on support_bot/. All modules rely on a shared foundation providing an injectable clock (support_bot/clock.py), runtime configuration with SQLite overrides (support_bot/config.py), a background job queue (support_bot/jobs.py), structured audit logging (support_bot/audit.py), PII masking (support_bot/masking.py), prompt-injection detection (support_bot/injection.py), role-based access control (support_bot/rbac.py), and forward-only SQLite migrations (support_bot/db.py).

```text
support_bot/
├── conversation/   # Multilingual intent analysis, entity preservation, session lifecycle
├── sentiment/      # Multi-label sentiment analysis, Platt scaling calibration
├── escalation/     # Escalation rule evaluation, 15-minute timer, resolution detection
├── tickets/        # Ticket model, multi-factor priority, team routing, SLA deadlines
├── knowledge/      # Knowledge pipeline, document ingestion, quality gates, versioning
├── rag/            # Secure RAG service, pre-retrieval RBAC filters, citation builder
├── files/          # File upload validation, OCR engine, discrepancy checks, purge jobs
└── migrations/     # Versioned SQLite schema migrations (001 through 005)
```

## Testing

Run the full suite from the project root:

```bash
PYTHONPATH=. pytest
```

Current results from a clean run: **188 passed, 1 warning** in ~14 s.

Individual module tests can be run in isolation:

```bash
PYTHONPATH=. pytest tests/test_sentiment*.py -v      # Module 1
PYTHONPATH=. pytest tests/test_tickets.py -v         # Module 2
PYTHONPATH=. pytest tests/test_knowledge*.py -v      # Module 3
PYTHONPATH=. pytest tests/test_rag*.py -v            # Module 4
PYTHONPATH=. pytest tests/test_files_ocr.py -v       # Module 5
PYTHONPATH=. pytest tests/test_conversation*.py -v   # Module 6
```

## Limitations

- **Live Gemini calls**: In network-restricted or quota-exhausted environments, model calls fall back to deterministic heuristics and FakeOCR without crashing.
- **Golden QA evaluation**: The quality gate uses a small synthetic set of 2 golden questions in tests/data/golden_qa.json rather than a large production evaluation set.
- **Sentiment calibration**: Platt scaling weights were calibrated on a synthetic dataset of 120 labeled samples in dataset/calibration_data.csv.
- **OCR extraction**: Uses simple regex and mock vision extraction for receipts; handwritten, skewed, or degraded documents are not evaluated.
- **File storage**: Uploaded documents are stored on the local filesystem rather than a cloud object store.
- **Live sentiment check**: not tested live — the live Gemini sentiment check script requires a valid API key and network access; it is skipped in CI.

For detailed module specifications, configuration keys, and requirement-to-test mapping tables, see [docs/MODULES.md](docs/MODULES.md).
