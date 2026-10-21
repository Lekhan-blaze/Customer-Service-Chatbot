# Implementation Plan — Shared Foundation (`support_bot/`)

## Goal
Add a reusable foundation layer that future internship modules (multi-language,
escalation, SLA, admin dashboard, etc.) will build on, without changing the
existing chatbot behaviour.

## Architecture Decisions

| Decision | Rationale |
|---|---|
| **Pure-Python package** (`support_bot/`) | No framework coupling; importable from Streamlit, CLI, tests, or future FastAPI |
| **SQLite for persistence** | Zero-ops, ships with Python, sufficient for single-node deployment |
| **Dependency injection via `Container`** | Modules get clock/config/db/logger from a container, never import globals — makes testing trivial with `FakeClock` and in-memory SQLite |
| **YAML defaults + SQLite overrides** | `config/defaults.yaml` is version-controlled; runtime overrides live in a `config_overrides` table so ops can change values without redeploying |
| **Versioned migrations** | Forward-only SQL scripts in `support_bot/migrations/` applied in order |

## Package Layout

```
support_bot/
├── __init__.py          # Package marker
├── clock.py             # SystemClock, FakeClock, business-hours helpers
├── config.py            # load defaults.yaml, merge SQLite overrides
├── container.py         # DI container / factory
├── db.py                # get_connection, run_migrations
├── audit.py             # Structured JSON audit logger
├── masking.py           # PII / payment data masking
├── injection.py         # Prompt-injection risk scorer
├── rbac.py              # Roles & permission checks
├── jobs.py              # SQLite-backed job queue + worker
├── migrations/
│   └── 001_initial.sql  # config_overrides, audit_log, jobs, roles tables
config/
└── defaults.yaml        # All default configuration keys
tests/
├── test_baseline.py     # (existing)
├── test_clock.py
├── test_config.py
├── test_audit.py
├── test_masking.py
├── test_injection.py
├── test_rbac.py
├── test_jobs.py
├── test_chat_smoke.py   # Existing chat flow with mocked LLM
└── conftest.py          # Shared fixtures (tmp db, container, FakeClock)
```

## Configuration Keys (defaults.yaml)

- `company_timezone`: Asia/Kolkata
- `business_hours.start / end`: 09:00 / 18:00
- `business_hours.days`: Mon–Fri
- `next_working_day_schedule_time`: 09:00
- `holidays`: []
- `sla.first_response_minutes`: 30
- `sla.resolution_hours`: 24
- `retention.audit_days`: 90
- `retention.session_days`: 30
- `session.inactivity_timeout_minutes`: 30
- `session.restore_window_hours`: 24
- `supported_languages`: [English, Hindi, Tamil, Kannada]
- `maintenance_window.start / end`: 02:00 / 04:00
- `injection_detection.enabled`: true
- `injection_detection.threshold`: 0.6
- `job_retry_backoff_minutes`: [15, 30, 60]

## Commit Plan

1. `docs/PLAN.md` + `config/defaults.yaml`
2. `support_bot/` core: clock, config, db, container, migrations
3. `support_bot/` features: audit, masking, injection, rbac, jobs
4. Tests + `conftest.py`
5. README update + final push
