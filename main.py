"""Customer Service Chatbot — Streamlit UI.

Extends the base FAQ chatbot with:
- Multilingual conversation & session management (ConversationService)
- Sentiment analysis, tone adjustment, and escalation (SentimentEscalationService)
- Knowledge-base pipeline & Secure RAG layer with pre-retrieval filtering (Modules 3 & 4)
- Optional Debug / Demo sidebar panel (enabled via config.demo_panel_enabled)
"""

import uuid
import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db
from support_bot.config import get_config
from support_bot.db import get_connection, run_migrations
from support_bot.clock import FakeClock, SystemClock
from support_bot.container import Container
from support_bot.conversation.service import ConversationService
from support_bot.escalation.service import SentimentEscalationService
from support_bot.rbac import Role

st.title("CUSTOMER SERVICE CHATBOT 🤖")

# ── Session bootstrap ──────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
if "fake_clock" not in st.session_state:
    st.session_state["fake_clock"] = None   # None → use real SystemClock
if "user_role" not in st.session_state:
    st.session_state["user_role"] = "customer"

# ── Shared resources ───────────────────────────────────────────────────────────
conn = get_connection()
run_migrations(conn)
cfg = get_config(conn)

demo_enabled = cfg.get("demo_panel_enabled", False)
clock = st.session_state["fake_clock"] or SystemClock()

container = Container(conn=conn, clock=clock, auto_migrate=False)
kb_pipeline = container.get_knowledge_pipeline()
rag_svc = container.get_rag_service(pipeline=kb_pipeline)

# Service instance for due jobs and escalations
esc_svc = SentimentEscalationService(conn=conn, clock=clock, config=cfg, auto_migrate=False)
esc_svc.process_due_jobs()
container.job_queue.run_due()

# ── Knowledgebase builder ──────────────────────────────────────────────────────
btn = st.button("Create Knowledgebase")
if btn:
    with st.spinner("Creating knowledgebase..."):
        create_vector_db()
    st.success("Knowledgebase created successfully!")

# ── User input ─────────────────────────────────────────────────────────────────
question = st.text_input("Question:")

if question:
    conv_enabled = cfg.get("conversation", {}).get("enabled", True)
    sent_enabled = cfg.get("escalation", {}).get("enabled", True)

    with st.spinner("Finding answer..."):
        sentiment_result = None
        escalation_record = None
        clarification = None
        tone = "neutral"

        if conv_enabled:
            service = ConversationService(conn=conn, clock=clock, config=cfg, auto_migrate=False)
            conv_resp = service.handle_message(st.session_state["session_id"], question)

            # ── Sentiment & escalation ─────────────────────────────────────────
            if sent_enabled:
                sentiment_result, escalation_record, clarification = esc_svc.process_message(
                    session_id=st.session_state["session_id"],
                    text=question,
                    context=conv_resp.context,
                )
                if sentiment_result:
                    tone = sentiment_result.tone_instruction or "neutral"

            if conv_resp.needs_clarification:
                answer = conv_resp.clarification_question
            elif clarification:
                answer = clarification
            else:
                query_to_run = conv_resp.normalized_query or question
                active_ver = kb_pipeline.get_active_version()
                if active_ver:
                    rag_res = rag_svc.answer(
                        question=query_to_run,
                        user_role=st.session_state["user_role"],
                        session_id=st.session_state["session_id"],
                    )
                    answer = service.format_reply(conv_resp, rag_res.answer)
                else:
                    chain = get_qa_chain(tone_instruction=tone)
                    raw_response = chain(query_to_run)
                    answer = service.format_reply(conv_resp, raw_response["result"])

        else:
            # Direct QA behavior
            active_ver = kb_pipeline.get_active_version()
            if active_ver:
                rag_res = rag_svc.answer(
                    question=question,
                    user_role=st.session_state["user_role"],
                    session_id=st.session_state["session_id"],
                )
                answer = rag_res.answer
            else:
                chain = get_qa_chain(tone_instruction=tone)
                raw_response = chain(question)
                answer = raw_response["result"]

    st.header("Answer")
    st.write(answer)

    # Escalation banner
    if escalation_record:
        routing_labels = {
            "on_call": "🚨 Escalated to on-call agent",
            "human_queue": "👤 Escalated to human support",
            "scheduled_next_working_day": "📅 Scheduled for next working day",
        }
        label = routing_labels.get(escalation_record.routing_outcome, "⚠️ Escalated")
        st.warning(f"{label} — {escalation_record.reason}")

# ── Debug / Demo Sidebar Panel ─────────────────────────────────────────────────
if demo_enabled:
    with st.sidebar:
        st.header("🛠 Debug / Demo Panel")

        # Requester Role Selector
        st.subheader("👤 Requester Authorization")
        current_role = st.session_state.get("user_role", "customer")
        role_options = ["customer", "internal", "admin"]
        role_idx = role_options.index(current_role) if current_role in role_options else 0
        selected_role = st.selectbox("Role / Access Level:", role_options, index=role_idx)
        if selected_role != current_role:
            st.session_state["user_role"] = selected_role
            st.rerun()

        # Simulated clock controls
        st.subheader("🕐 Simulated Clock")
        use_fake = st.checkbox("Use simulated clock", value=st.session_state["fake_clock"] is not None)
        if use_fake:
            if st.session_state["fake_clock"] is None:
                import datetime as _dt
                from zoneinfo import ZoneInfo
                st.session_state["fake_clock"] = FakeClock(
                    frozen=_dt.datetime.now(ZoneInfo("Asia/Kolkata"))
                )
                st.rerun()
            fake_clk = st.session_state["fake_clock"]
            current_fake = fake_clk.now()
            st.write(f"**Current simulated time:** `{current_fake.strftime('%Y-%m-%d %H:%M %Z')}`")
            advance_h = st.number_input("Advance hours", min_value=0, max_value=48, value=0, step=1)
            advance_m = st.number_input("Advance minutes", min_value=0, max_value=120, value=0, step=1)
            if st.button("⏩ Advance Clock") and (advance_h > 0 or advance_m > 0):
                fake_clk.advance(hours=int(advance_h), minutes=int(advance_m))
                st.success(f"Clock advanced by {advance_h}h {advance_m}m → {fake_clk.now().strftime('%H:%M %Z')}")
                st.rerun()
        else:
            if st.session_state["fake_clock"] is not None:
                st.session_state["fake_clock"] = None
                st.rerun()

        # Knowledge Base & RAG Pipeline Controls
        st.subheader("📚 Knowledge Base Pipeline")
        active_version = kb_pipeline.get_active_version()
        if active_version:
            st.success(f"Active Version: `{active_version.id}`")
            st.caption(f"Docs: {active_version.document_count} | Chunks: {active_version.chunk_count} | Quality: {active_version.quality_score:.2f}")
        else:
            st.info("Active Version: None (using dataset.csv fallback)")

        if st.button("📥 Ingest Inbox & Update Index"):
            with st.spinner("Ingesting knowledge inbox..."):
                ingest_res = kb_pipeline.ingest_inbox_and_update(user_role=st.session_state["user_role"])
            if ingest_res["status"] == "active":
                st.success(f"Activated version `{ingest_res['version_id']}`")
            elif ingest_res["status"] == "scheduled":
                st.warning(f"Scheduled activation for `{ingest_res['version_id']}` (outside maintenance window)")
            else:
                st.error(f"Update failed quality gate: `{ingest_res.get('error')}`")
            st.rerun()

        # Admin Rollback
        if st.session_state["user_role"] == "admin":
            st.markdown("**Admin Version Rollback:**")
            all_versions = kb_pipeline.list_versions()
            if all_versions:
                v_ids = [v.id for v in all_versions if not v.is_active]
                if v_ids:
                    target_ver = st.selectbox("Rollback to:", v_ids)
                    if st.button("⏪ Rollback"):
                        kb_pipeline.rollback_to(target_ver, actor="admin_ui", user_role=Role.ADMIN)
                        st.success(f"Rolled back to `{target_ver}`")
                        st.rerun()

        # Background Jobs / Due Escalations Trigger
        st.subheader("⚙️ Background Jobs")
        if st.button("▶️ Run due jobs"):
            due_esc = esc_svc.process_due_jobs()
            due_kb = container.job_queue.run_due()
            st.success(f"Processed {len(due_esc)} escalation job(s) and {len(due_kb)} background job(s).")
            st.rerun()

        # Sentiment display
        if question and "sentiment_result" in dir() and sentiment_result:
            st.subheader("😊 Sentiment")
            scores = sentiment_result.calibrated_scores
            for label, score in sorted(scores.items(), key=lambda x: -x[1]):
                st.progress(score, text=f"{label}: {score:.2f}")
            st.write(f"**Dominant:** `{sentiment_result.dominant_label}` | **Tone:** `{tone}`")
            if sentiment_result.risk_category:
                st.error(f"⚠️ Risk: `{sentiment_result.risk_category}` ({sentiment_result.risk_score:.2f})")

        # Escalation records display
        st.subheader("📋 Escalation Records")
        records = esc_svc.get_escalations(st.session_state["session_id"])
        if records:
            for r in records:
                status_icon = {"pending": "🟡", "in_progress": "🔵", "resolved": "✅"}.get(r.status, "⚪")
                st.write(f"{status_icon} **{r.rule_name}** → `{r.routing_outcome}` ({r.status})")
                st.caption(r.reason)
        else:
            st.write("No escalations for this session.")

        # Session ID
        st.subheader("🔑 Session")
        st.code(st.session_state["session_id"])
        if st.button("🔄 New Session"):
            st.session_state["session_id"] = str(uuid.uuid4())
            st.rerun()
