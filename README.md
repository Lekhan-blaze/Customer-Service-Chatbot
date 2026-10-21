# Support Chatbot Extended

A customer service question-answering chatbot repository initialized for internship project extensions.

## Original Features (Training Project)

The baseline project provides an interactive FAQ customer service chatbot designed to answer customer queries based on an educational program FAQ dataset:

- **Web Interface (`main.py`)**: A Streamlit application allowing users to trigger knowledgebase creation and enter customer service questions to receive context-grounded answers.
- **LLM**: Google Gemini (`gemini-3.8-flash` via `langchain-google-genai`) with low temperature (0.1) for deterministic responses.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` via `HuggingFaceEmbeddings` from `langchain-huggingface`.
  > **Note**: The original project used `hkunlp/instructor-large` via the deprecated `HuggingFaceInstructEmbeddings` class. That model is incompatible with current versions of `sentence-transformers` (`AttributeError: 'INSTRUCTOR' object has no attribute '_text_length'`). The embedding model was replaced with `all-MiniLM-L6-v2`, which is lightweight, fully supported, and produces high-quality retrieval results for FAQ-style workloads.
- **Knowledge Base & Vector Store (`langchain_helper.py`)**: Uses LangChain's `CSVLoader` to parse `dataset/dataset.csv` (using the `prompt` column as document source) and builds/loads a local FAISS vector database stored in `faiss_index/`.
- **Question-Answering Chain**: Implements a LangChain `RetrievalQA` pipeline ("stuff" chain type) with a custom prompt template ensuring answers strictly adhere to retrieved FAQ context, falling back to *"I don't know."* when information is absent.
- **Dataset (`dataset/dataset.csv`)**: 76 question-and-answer pairs covering course prerequisites, enrollment, job assistance, and program details across two columns (`prompt` and `response`).
  > **Note**: The original CSV file was encoded in Windows-1252 (cp1252). It was converted once to UTF-8 so that `CSVLoader` can read it reliably with `encoding="utf-8"`. Row count and content were verified unchanged (76 rows).
- **Embedding model name** is configurable via the `EMBEDDING_MODEL_NAME` environment variable (defaults to `sentence-transformers/all-MiniLM-L6-v2`).

> **Note on Future Work**: New internship modules, architectures, and expanded features will be added below as project milestones are developed.

---

## Setup & Running Locally

1. **Python Environment**:
   Ensure Python 3.11 is used. Create and activate a virtual environment:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Copy `.env.example` to `.env` and set your API key:
   ```bash
   cp .env.example .env
   ```
   Add your `GOOGLE_API_KEY` in `.env`.
   Optionally override the embedding model:
   ```bash
   EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
   ```

3. **Run Application**:
   ```bash
   streamlit run main.py
   ```

4. **Run Tests**:
   ```bash
   pip install pytest
   PYTHONPATH=. pytest tests/
   ```

---

## Internship Additions

A modular, shared foundation package (`support_bot/`) was added to support upcoming features (multilingual support, human escalation, SLA tracking, audit logging, admin dashboard):

### Foundation Package (`support_bot/`)
- **Dependency Injection (`Container`)**: Simple service container supplying `clock`, `config`, `db`/`conn`, `job_queue`/`jobs`, and `audit` to modules without global state.
- **Clock Abstraction (`Clock`, `SystemClock`, `FakeClock`)**: Deterministic time mocking in tests, plus business-hours calculations (`is_business_hours`, `next_working_day`, `add_business_time`) honoring holidays and working days.
- **Runtime Configuration (`config/defaults.yaml` & `config_overrides`)**: Version-controlled defaults merged with runtime SQLite overrides for zero-downtime updates.
- **Job Queue (`JobQueue`, `jobs.py`)**: SQLite-backed asynchronous queue supporting delayed, scheduled, retryable jobs with exponential/configurable backoff (`[15, 30, 60]` minutes), and after-hours execution.
- **Structured Audit Logging (`AuditLogger`)**: JSON-structured event logging with decisions, rules, references, and metadata written to SQLite and Python logging.
- **PII & Payment Data Masking (`mask_pii`, `MaskingFilter`)**: Sanitizes emails, phone numbers, credit cards, bank accounts, IFSC codes, UPI IDs, and API keys.
- **Prompt-Injection Detector (`detect_injection`)**: Heuristic risk scoring for override patterns, jailbreak keywords, role hijacking, and data exfiltration.
- **Role-Based Access Control (`rbac.py`)**: Pre-seeded roles (`viewer`, `ingestor`, `approver`, `activator`, `admin`) with a `@require_permission` decorator.
- **Database Migrations (`db.py`, `migrations/`)**: Forward-only SQLite migrations with WAL mode and foreign key enforcement.

### Configuration (`config/defaults.yaml`)
Key configurations include:
- `company_timezone`: `Asia/Kolkata`
- `business_hours`: `09:00` - `18:00` (Mon–Fri)
- `next_working_day_schedule_time`: `09:00`
- `holidays`: Configurable list of ISO date strings
- `sla`: First response (30 min), resolution (24 hours)
- `retention`: Audit logs (90 days), sessions (30 days)
- `session`: Inactivity timeout (30 min), restore window (24 hours)
- `supported_languages`: English, Hindi, Tamil, Kannada
- `maintenance_window`: `02:00` - `04:00`
- `injection_detection`: Enabled with threshold `0.6`
- `job_retry_backoff_minutes`: `[15, 30, 60]`

### How to Run Tests
```bash
PYTHONPATH=. pytest tests/ -v
```
The test suite includes 57 automated tests covering the baseline Q&A flow (mocked LLM), FakeClock timing, configuration overrides, audit logging, PII masking, injection detection, RBAC permissions, and job retries.
