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
The test suite includes 136 automated tests covering the foundation package, multilingual conversation pipeline, entity preservation, session lifecycle, sentiment analysis, Platt calibration, and escalation rules.

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
4. **Customer Fact Corrections & Spelling Tolerance**:
   - Detects customer corrections (e.g. *"Sorry, I meant order 4521, not 4512"*) via model `is_correction` flag with regex fallback, updates active session facts in SQLite, and logs an audit record (`customer_correction`).
   - Handles spelling mistakes gracefully while extracting correct intents and entities.
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
| Spelling mistakes tolerance | `tests/test_conversation_service.py` | `test_spelling_mistakes_intent_and_entity_preservation` |
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

---

## Module 1: Sentiment and escalation

The `support_bot/sentiment/` and `support_bot/escalation/` modules provide multilingual sentiment analysis, confidence calibration via Platt scaling, dynamic response tone adaptation, and deterministic business-hours-aware escalation routing.

### What It Does
1. **Multi-Label Sentiment & Sarcasm Guard**:
   - Evaluates messages across six sentiment labels: `positive`, `neutral`, `negative`, `frustrated`, `urgent`, and `sarcastic`.
   - **Sarcasm Guard**: When `sarcastic >= 0.60`, positive scores are suppressed, ensuring sarcastic comments (e.g. *"Oh great service, my course is still locked"*) are treated with urgent/negative attention rather than positive sentiment.
2. **Platt Confidence Calibration**:
   - Fits logistic regression per label ($P = \sigma(a \cdot s + b)$) to calibrate raw scores, reducing Expected Calibration Error (ECE) and improving probability reliability.
3. **Dynamic Response Tone Adaptation**:
   - Adjusts retriever generation tone instructions:
     - `frustrated` / `negative` $\rightarrow$ `empathetic` (acknowledging feelings without adding unverified facts)
     - `urgent` $\rightarrow$ `concise` (direct, brief answers without filler)
     - `neutral` / `positive` $\rightarrow$ `neutral` (standard helpful FAQ response)
   - Guarded anti-hallucination constraint strictly enforces preserving the `"I don't know."` fallback.
4. **Deterministic Escalation Routing**:
   - **Rule 1: High-Risk Category**: If `risk_category` $\in$ {`account_compromise`, `duplicate_payment`, `legal_threat`, `data_breach`} and `risk_score >= 0.70`, immediately routes to `human_queue` (in business hours) or `on_call` (outside business hours).
   - **Rule 2: Urgent Outside Hours**: If `urgent >= 0.65` outside business hours, routes to `on_call`.
   - **Rule 3: Repeated Negativity**: If customer has $\ge 2$ consecutive negative/frustrated messages, routes to `human_queue` (in hours) or `scheduled_next_working_day` (outside hours).
   - **Rule 4: Negative Outside Hours**: Single negative message outside business hours routes to `scheduled_next_working_day`.
   - **Rule 5: 15-Minute Unresolved-Negativity Timer**:
     - The timer job starts at the **FIRST** negative or frustrated message in a conversation.
     - If the conversation remains unresolved for 15 minutes, the background job escalates the session to `human_queue` (or `scheduled_next_working_day`).
     - **Idempotency**: Escalation records are strictly idempotent; no duplicate escalation is generated if `repeated_negative` also fires.
5. **Safe Customer-Facing Escalation Message**:
   - Recorded in `escalation_records.customer_message`.
   - Makes **no promises** about financial outcomes, refunds, or arbitrary turnaround times beyond configured working days.

### Definition of "Resolved"
A conversation is considered **resolved** when any of the following conditions occurs:
1. **Later Positive or Neutral Sentiment**: The customer sends a subsequent message whose dominant sentiment label is `positive` or `neutral`.
2. **Explicit Customer Confirmation**: The customer message matches confirmation keywords (e.g. *"thank you"*, *"thanks"*, *"got it"*, *"perfect"*, *"resolved"*, *"okay"*, *"ok"*, *"understood"*, *"sorted"*).
3. **Agent / System Resolution**: An agent or administrative workflow marks the escalation resolved via `service.resolve_escalation(session_id)`.

When a conversation is resolved, the pending `unresolved_negativity_timeout` job is immediately cancelled, and any open negativity escalations are marked `status = 'resolved'`.

### Configuration Keys (`config/defaults.yaml`)
| Key | Default | Description |
|---|---|---|
| `sentiment.sarcasm_threshold` | `0.60` | Threshold above which sarcasm suppresses positive score |
| `sentiment.low_confidence_threshold` | `0.45` | Score below which clarification is requested without escalation |
| `sentiment.recent_messages_count` | `5` | History window for conversation-level sentiment aggregation |
| `sentiment.calibration.enabled` | `true` | Enables Platt scaling transformation of raw sentiment scores |
| `sentiment.calibration.weights` | `...` | Per-label sigmoid parameters `a` and `b` |
| `escalation.enabled` | `true` | Master flag to enable/disable automated escalations |
| `escalation.repeated_negative_threshold` | `2` | Number of consecutive negative messages triggering escalation |
| `escalation.risk_threshold` | `0.70` | Confidence threshold for high-risk category escalation |
| `escalation.urgent_threshold` | `0.65` | Confidence threshold for urgent out-of-hours escalation |
| `escalation.unresolved_negative_timeout_minutes` | `15` | Timeout period for unresolved negative sentiment timer |
| `escalation.high_risk_categories` | `[...]` | List of severe categories requiring immediate escalation |
| `demo_panel_enabled` | `true` | Toggles the sentiment and escalation inspect panel in Streamlit UI |

### How to Run Calibration
```bash
python scripts/calibrate_sentiment.py
```
This fits Platt scaling parameters on `tests/data/sentiment_validation.json`, prints per-label accuracy and ECE, and updates `config/defaults.yaml`.

*Note on dataset size*: The validation dataset (`tests/data/sentiment_validation.json`) contains 65 labeled synthetic examples across 5 languages (English, Hindi, Tamil, Kannada, Hinglish). While sufficient for unit tests and relative calibration benchmarks, production deployments should fit calibration on $\ge 500$ human-annotated transcripts.

### How to Run Live Model Check
```bash
python scripts/live_sentiment_check.py
```
Runs the real `GeminiSentimentAnalyzer` on 8 representative sample messages (sarcasm, duplicate payment, urgent, Hindi, Tamil, Kannada, Hinglish, neutral) without exposing API keys.

### How to Run Tests
```bash
pytest tests/test_sentiment_escalation.py -v
```

### Requirement-to-Test Mapping Table
| Requirement | Test Module | Test Case |
|---|---|---|
| 15-minute timer escalates after 16 min | `tests/test_sentiment_escalation.py` | `test_one_negative_message_then_16_minutes_escalates` |
| Resolved at 10 min cancels timer (no escalation) | `tests/test_sentiment_escalation.py` | `test_resolved_at_10_minutes_no_escalation` |
| Idempotency (no duplicate with repeated-negative) | `tests/test_sentiment_escalation.py` | `test_no_duplicate_escalation_if_repeated_negative_also_fires` |
| Repeated negative messages in business hours | `tests/test_sentiment_escalation.py` | `test_repeated_negative_in_hours` |
| Repeated negative messages outside business hours | `tests/test_sentiment_escalation.py` | `test_repeated_negative_outside_hours_schedules` |
| Negative sentiment outside business hours | `tests/test_sentiment_escalation.py` | `test_negative_outside_hours_schedules_next_day` |
| High-risk category detection (in & outside hours) | `tests/test_sentiment_escalation.py` | `test_high_risk_in_hours_routes_to_human_queue`<br>`test_high_risk_outside_hours_routes_to_on_call` |
| Urgent message outside business hours | `tests/test_sentiment_escalation.py` | `test_urgent_outside_hours_routes_to_on_call` |
| Escalation table schema & stored fields verification | `tests/test_sentiment_escalation.py` | `test_escalation_record_stored_fields` |
| Safe customer escalation message (no outcome promises) | `tests/test_sentiment_escalation.py` | `test_customer_facing_message_makes_no_unsupported_promises` |
| Low-confidence sentiment clarification | `tests/test_sentiment_escalation.py` | `test_low_confidence_returns_clarification_no_escalation` |
| Manual escalation resolution | `tests/test_sentiment_escalation.py` | `test_resolve_escalation`<br>`test_resolve_specific_rule` |
| Platt scaling formula & overflow protection | `tests/test_sentiment_escalation.py` | `test_platt_scale_midpoint`<br>`test_platt_scale_overflow_protection` |
| Expected Calibration Error (ECE) metric calculation | `tests/test_sentiment_escalation.py` | `test_ece_perfect_calibration`<br>`test_ece_worst_calibration` |
| Multi-label sentiment analysis heuristics | `tests/test_sentiment_escalation.py` | `test_angry_message`<br>`test_urgent_message`<br>`test_positive_message`<br>`test_legal_threat` |
| Sarcasm guard suppresses positive sentiment | `tests/test_sentiment_escalation.py` | `test_sarcasm_suppresses_positive`<br>`test_sarcasm_guard_in_gemini` |
| Multilingual sentiment (Hindi, Hinglish) | `tests/test_sentiment_escalation.py` | `test_hindi_frustrated`<br>`test_hinglish_sarcastic` |
| Tone instructions & anti-hallucination guard | `tests/test_sentiment_escalation.py` | `test_empathetic_tone_phrase`<br>`test_concise_tone_phrase`<br>`test_neutral_tone_no_hint`<br>`test_tone_hint_has_no_added_commitments` |
| PII masking in stored sentiment | `tests/test_sentiment_escalation.py` | `test_pii_masked_in_stored_sentiment` |




---

## Module 3: Knowledge-Base Pipeline

### Overview
A lean, secure knowledge-base ingestion and activation pipeline supporting immutable index versioning, quarantine for unsafe or malicious uploads, quality gates, scheduled activation inside maintenance windows, automatic health-check rollbacks, and role-based access control.

### Architecture & Lifecycle
1. **Inbox Discovery & Metadata Validation**: Processes `.txt`, `.md`, and `.csv` files alongside `.meta.json` sidecars in `knowledge_inbox/`. Duplicates (same content SHA256) are skipped idempotently.
2. **Quarantine & Prompt Injection Defense**: Files failing type allowlists, exceeding size bounds (>1MB), or containing prompt injections (e.g. system override instructions) are rejected and stored in `kb_quarantine` with risk scores and auditable reasons.
3. **Immutable Version Building**: Each build creates an isolated index directory under `faiss_index/versions/<id>/`. The active version pointer is never overwritten in place.
4. **Automated Quality Gate**: Evaluates candidate indices against `tests/data/golden_qa.json`. Requires minimum retrieval hit rate and grounding scores; rejects updates that degrade below active scores.
5. **Scheduled Maintenance Activation**: Activations only take effect inside the configured maintenance window (`maintenance_window: 02:00-04:00`). Updates requested outside this window are automatically enqueued via `JobQueue` for activation during the window.
6. **Post-Activation Health Checks & Auto-Rollback**: After activation, health checks verify index search integrity. If a check fails, the pipeline immediately rolls back to the previous stable version.
7. **Failed Update Retries**: Failed updates trigger retries at 15, 30, and 60 minutes via the job queue, recording each attempt in the audit log.
8. **RBAC & Manual Rollback**: Only `admin` role may execute manual rollbacks. `ingestor` builds versions, `approver` approves, and `activator` triggers activation.

### Configuration Keys (`config/defaults.yaml`)
- `maintenance_window.start`: Start time for index updates (default `"02:00"`).
- `maintenance_window.end`: End time for index updates (default `"04:00"`).
- `knowledge.min_hit_rate`: Minimum retrieval hit rate threshold for quality gate (default `0.80`).
- `knowledge.min_grounding_score`: Minimum grounding score threshold (default `0.75`).
- `knowledge.auto_rollback_window_seconds`: Time window for post-activation health checks (default `300`).

### Synthetic Policy Documents (`knowledge_inbox/`)
- `refund_policy_v1.txt` + `.meta.json`: Expired 7-day refund policy (active 2025, expired 2026-01-01).
- `refund_policy_v2.txt` + `.meta.json`: Current 14-day refund policy with immediate duplicate charge refunds.
- `refund_policy_v3.txt` + `.meta.json`: Future-dated 30-day money-back guarantee (effective 2026-12-01).
- `internal_escalation_guide.txt` + `.meta.json`: Internal-only escalation policy for Tier 2 supervisors.
- `india_fee_policy.txt` + `.meta.json`: Region-specific GST and processing fee policy.
- `malicious_prompt_injection.txt` + `.meta.json`: Synthetic malicious document containing injection triggers to verify quarantine.

### Requirement-to-Test Mapping Table
| Requirement | Test Module | Test Case |
|---|---|---|
| Legal & illegal state transitions | `tests/test_knowledge_pipeline.py` | `test_state_machine_legal_and_illegal_transitions` |
| Valid file ingestion & chunking | `tests/test_knowledge_pipeline.py` | `test_ingest_document_success_and_chunking` |
| Idempotent duplicate skipping (content hash) | `tests/test_knowledge_pipeline.py` | `test_duplicate_document_skipped` |
| Invalid file types & oversized files quarantined | `tests/test_knowledge_pipeline.py` | `test_invalid_extension_and_oversized_file_quarantined` |
| Malicious prompt injections quarantined | `tests/test_knowledge_pipeline.py` | `test_malicious_prompt_injection_quarantined` |
| RBAC on ingest, approve, activate, rollback | `tests/test_knowledge_pipeline.py` | `test_rbac_enforcement_on_pipeline_actions` |
| Maintenance window scheduling outside window | `tests/test_knowledge_pipeline.py` | `test_maintenance_window_activation_scheduling` |
| Degraded quality gate rejection | `tests/test_knowledge_pipeline.py` | `test_degraded_quality_gate_rejected` |
| Manual rollback by admin with audit logging | `tests/test_knowledge_pipeline.py` | `test_manual_rollback_by_admin` |
| Failed updates retry at 15, 30, 60 minutes | `tests/test_knowledge_pipeline.py` | `test_failed_update_schedules_retries_15_30_60` |
| Health-check failure triggers automatic rollback | `tests/test_knowledge_pipeline.py` | `test_health_check_auto_rollback_within_5_minutes` |

### What is Simplified
- Embedded vector store utilizes cosine similarity with normalized TF-IDF downweighting in test suites for fast, deterministic unit test execution without requiring API keys or model downloads.
- Maintenance window schedule executes via the injectable `JobQueue` and `FakeClock`.

---

## Module 4: Secure RAG Layer

### Overview
Enterprise RAG retrieval layer enforcing strict pre-retrieval authorization guards, temporal policy selection, citation generation, prompt-injection neutralization on retrieved passages, and claim-to-source validation.

### Key Capabilities
1. **Pre-Retrieval Metadata Guards**: Authorization (`customer` vs `internal`), region (`IN` vs `GLOBAL`), and policy dates are filtered BEFORE vector ranking, ensuring non-qualifying passages never reach LLM context or candidates.
2. **Temporal Policy Handling**:
   - Current queries automatically ignore future-dated policies and expired policies.
   - Historical queries (e.g., *"What was the refund policy on 5 March 2025?"*) extract the target date and retrieve the policy active on that date.
   - Conflict resolution: When multiple versions match, the latest version takes precedence.
3. **Mandatory Citations**: Every factual answer includes structured citations: document title, version number, and source locator.
4. **Untrusted Passage Defense**: Passages undergo prompt injection inspection; instructions attempting system overrides or credential leaks are neutralized.
5. **Claim-to-Source Validation**: Verifies generated answers against retrieved passages before returning to the user, stripping unsupported or hallucinated claims.
6. **Fallback Behavior**: If evidence is missing, conflicting, or below retrieval thresholds, securely falls back to `"I don't know."`.

### Requirement-to-Test Mapping Table
| Requirement | Test Module | Test Case |
|---|---|---|
| Query date extraction (historical vs current) | `tests/test_secure_rag.py` | `test_parse_date_from_query` |
| Current questions use active policy v2 (ignoring v1, v3) | `tests/test_secure_rag.py` | `test_current_question_uses_active_policy_v2` |
| Historical date queries return policy active on that date | `tests/test_secure_rag.py` | `test_historical_date_question_returns_policy_active_then` |
| Future-dated policies ignored for current questions | `tests/test_secure_rag.py` | `test_future_dated_policy_ignored_for_current_questions` |
| Restricted documents hidden from customers | `tests/test_secure_rag.py` | `test_restricted_document_hidden_from_customer_visible_to_internal` |
| Missing evidence triggers safe refusal ("I don't know.") | `tests/test_secure_rag.py` | `test_missing_evidence_refuses` |
| Mandatory citations on factual answers | `tests/test_secure_rag.py` | `test_citation_present_on_every_factual_answer` |
| Unsupported claim removed by validator | `tests/test_secure_rag.py` | `test_unsupported_claim_removed_by_validator` |
| Latest policy version wins in conflicts | `tests/test_secure_rag.py` | `test_conflicting_policies_latest_wins` |
| Malicious instructions in documents suppressed | `tests/test_secure_rag.py` | `test_malicious_instruction_in_retrieved_document_suppressed` |

### What is Simplified
- Claim-to-source validation uses an interface (`BaseClaimValidator`) backed by token-overlap claim verification in tests and LLM factual verification in production.
- Natural date parsing supports ISO-8601, day-month-year, and month-day-year patterns.
