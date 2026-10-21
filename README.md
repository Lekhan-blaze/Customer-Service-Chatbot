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
The test suite includes 75 automated tests covering the foundation package, multilingual conversation pipeline, entity preservation, session lifecycle, and chat flows.

---

## Module 6: Multilingual Sessions

The `support_bot/conversation/` module extends the customer service chatbot with model-based multilingual conversation analysis, entity preservation, and session lifecycle management while keeping the English FAQ knowledge base (`dataset/dataset.csv`) and FAISS retriever intact.

### What It Does
1. **Model-Based Language & Intent Detection**:
   - Detects customer languages (`English`, `Hindi`, `Tamil`, `Kannada`), code-switching (e.g. `Hinglish`), scripts (`native`, `transliterated` / romanized), and multiple intents per message in priority order.
   - Normalizes non-English or code-switched questions into clean English for FAISS retrieval, and translates answers back into the customer's language.
   - Built behind the `MessageAnalyzer` interface (`GeminiAnalyzer` for production with structured JSON outputs, `FakeAnalyzer` for deterministic offline testing).
2. **Strict Entity Preservation**:
   - Protects domain entities (order/payment IDs e.g. `ORD-4521`, `PAY_9988`, product/course codes e.g. `PY-101`, dates, currency amounts e.g. `₹5000`, and names) with unique placeholders prior to normalization.
   - Restores protected entities verbatim into the translated response so IDs, amounts, and dates are never altered or translated.
3. **Low-Confidence Clarification**:
   - If language or intent confidence falls below configured thresholds (`0.60`), the system asks a short clarification question in the customer's most likely language rather than guessing.
4. **Customer Fact Corrections**:
   - Detects customer corrections (e.g. *"Sorry, I meant order 4521, not 4512"*), updates active session facts in SQLite, and logs an audit record (`customer_correction`).
5. **Session Management & Isolation**:
   - SQLite-backed persistence (`conversation_sessions`, `conversation_messages`) isolated by session ID with zero global state.
   - Retains the latest 10 messages (configurable) plus a durable summary generated by the analyzer with deterministic fallback.
   - **Inactivity Expiry**: Sessions expire after 30 minutes of inactivity (driven by injectable `Clock`).
   - **Restore Window**: If a customer returns within 24 hours, the prior summary and facts are restored into an active session; returning after 24 hours starts a fresh session while retaining historical records.
   - **Security**: Treats all user messages as untrusted by running prompt-injection detection on every turn, and automatically sanitizes PII/payment data in stored message records.
6. **Streamlit UI Integration**:
   - Integrated into `main.py` via `st.session_state["session_id"]`.
   - Supports zero-downtime disabling via `conversation.enabled: false` to fall back to the original direct QA flow.

### Configuration Keys (`config/defaults.yaml`)
| Key | Default | Description |
|---|---|---|
| `conversation.enabled` | `true` | Enables or disables the multilingual conversation pipeline |
| `conversation.confidence_threshold` | `0.65` | General confidence threshold |
| `conversation.language_confidence_threshold` | `0.60` | Minimum confidence required before asking clarification |
| `conversation.intent_confidence_threshold` | `0.60` | Minimum intent confidence required before asking clarification |
| `conversation.max_recent_messages` | `10` | Maximum recent messages retained in prompt context |
| `session.inactivity_timeout_minutes` | `30` | Inactivity period after which a session expires |
| `session.restore_window_hours` | `24` | Window during which returning customers restore prior summary |
| `supported_languages` | `[English, Hindi, Tamil, Kannada]` | Supported customer languages |

### How to Test
```bash
PYTHONPATH=. pytest tests/test_conversation_*.py -v
```

### Requirement-to-Test Mapping Table
| Requirement | Test Module | Test Case |
|---|---|---|
| Language switching in one session | `tests/test_conversation_service.py` | `test_language_switching_inside_one_conversation` |
| Hinglish code-switching | `tests/test_conversation_service.py` | `test_code_switching_and_transliteration_hinglish` |
| Transliterated input (Kannada, Tamil) | `tests/test_conversation_service.py` | `test_transliterated_kannada_and_tamil` |
| Multiple intents in one message | `tests/test_conversation_service.py` | `test_multiple_intents_in_one_message` |
| Customer corrections & audit logging | `tests/test_conversation_service.py` | `test_customer_correction_handling_and_audit` |
| Ambiguous follow-ups using context | `tests/test_conversation_service.py` | `test_ambiguous_follow_up_with_context` |
| Low-confidence clarification question | `tests/test_conversation_service.py` | `test_low_confidence_clarification` |
| Prompt-injection detection & refusal | `tests/test_conversation_service.py` | `test_prompt_injection_refusal_and_audit` |
| End-to-end entity preservation | `tests/test_conversation_service.py` | `test_entity_preservation_end_to_end` |
| Entity preservation (IDs, dates, amounts, codes) | `tests/test_conversation_entities.py` | `test_entity_preservation_order_and_payment_ids`<br>`test_entity_preservation_product_codes_currency_dates`<br>`test_contextual_order_id_patterns`<br>`test_extra_entities_from_facts` |
| Concurrent session isolation | `tests/test_conversation_session.py` | `test_concurrent_sessions_isolation` |
| 10-message history window | `tests/test_conversation_session.py` | `test_session_message_history_limit` |
| 30-min inactivity expiry | `tests/test_conversation_session.py` | `test_inactivity_expiry_30_minutes` |
| 24-hr restore window vs fresh session | `tests/test_conversation_session.py` | `test_24_hour_restore_window_vs_fresh_session` |
| PII masking in session message storage | `tests/test_conversation_session.py` | `test_pii_masking_in_session_storage` |

