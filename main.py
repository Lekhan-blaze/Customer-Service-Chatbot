"""Customer Service Chatbot — Streamlit UI.

Integrates all 6 modules:
- Multilingual conversation & session management (ConversationService)
- Sentiment analysis, tone adjustment, and escalation (SentimentEscalationService)
- Knowledge-base pipeline & Secure RAG layer with pre-retrieval filtering (Modules 3 & 4)
- Structured tickets, dynamic routing, duplicate linking & SLA tracking (Module 2)
- Multimodal file uploads, OCR extraction, and conflict detection (Module 5)
- Debug / Demo sidebar panel with interactive scenarios (guarded by config.demo_panel_enabled)
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

demo_enabled = cfg.get("demo_panel_enabled", True)
clock = st.session_state["fake_clock"] or SystemClock()

container = Container(conn=conn, clock=clock, auto_migrate=False)
kb_pipeline = container.get_knowledge_pipeline()
rag_svc = container.get_rag_service(pipeline=kb_pipeline)
ticket_svc = container.get_ticket_service()
file_svc = container.get_file_service()

# Service instance for due jobs and escalations
esc_svc = SentimentEscalationService(conn=conn, clock=clock, config=cfg, auto_migrate=False)

# Make sure process_due_jobs() runs on each rerun
esc_svc.process_due_jobs()
file_svc.job_queue.tick(file_svc.get_job_handlers())
container.job_queue.run_due()

# ── Knowledgebase builder (Baseline) ───────────────────────────────────────────
col_kb_btn, col_kb_status = st.columns([1, 2])
with col_kb_btn:
    btn = st.button("Create Knowledgebase")
    if btn:
        with st.spinner("Creating knowledgebase..."):
            create_vector_db()
        st.success("Knowledgebase created successfully!")

# ── Multimodal File Uploader (Module 5) ─────────────────────────────────────────
with st.expander("📎 Attach Document / Screenshot / Receipt (OCR Analysis)", expanded=False):
    uploaded_file = st.file_uploader(
        "Upload image or PDF invoice/receipt:",
        type=["png", "jpg", "jpeg", "pdf"],
        key="file_upload_widget",
    )
    user_stated_claims = {}
    claim_order = st.text_input("Your Order / Enrollment ID (optional):", key="claim_order_input")
    claim_amount = st.text_input("Amount charged (optional, e.g. 9999):", key="claim_amount_input")
    if claim_order:
        user_stated_claims["order_id"] = claim_order.strip()
    if claim_amount:
        user_stated_claims["amount"] = claim_amount.strip()

    if uploaded_file and st.button("Upload & Analyze"):
        with st.spinner("Analyzing document..."):
            record, rej, conflicts = file_svc.process_upload(
                file_path=uploaded_file.name,
                file_bytes=uploaded_file.getvalue(),
                session_id=st.session_state["session_id"],
                user_claims=user_stated_claims,
            )
        if rej:
            st.error(f"❌ Upload Rejected: {rej}")
        elif record:
            st.success(f"✅ File Received (Upload ID: `{record.id}` | Status: `{record.ocr_status}`)")
            if record.customer_message:
                st.info(f"ℹ️ {record.customer_message}")
            if record.ocr_extracted_fields:
                st.markdown("**Extracted Fields:**")
                st.json(record.ocr_extracted_fields)
            if conflicts:
                st.warning("⚠️ Discrepancies / Clarification Needed:")
                for c in conflicts:
                    st.write(f"- {c.clarification_prompt}")

# ── User input (Chat) ──────────────────────────────────────────────────────────
question = st.text_input("Question:")

if question:
    conv_enabled = cfg.get("conversation", {}).get("enabled", True)
    sent_enabled = cfg.get("escalation", {}).get("enabled", True)

    with st.spinner("Finding answer..."):
        sentiment_result = None
        escalation_record = None
        clarification = None
        tone = "neutral"
        citations = []

        if conv_enabled:
            service = ConversationService(conn=conn, clock=clock, config=cfg, auto_migrate=False)
            conv_resp = service.handle_message(st.session_state["session_id"], question)

            # Sentiment & escalation
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
                    citations = rag_res.citations
                else:
                    chain = get_qa_chain(tone_instruction=tone)
                    raw_response = chain(query_to_run)
                    answer = service.format_reply(conv_resp, raw_response["result"])

        else:
            active_ver = kb_pipeline.get_active_version()
            if active_ver:
                rag_res = rag_svc.answer(
                    question=question,
                    user_role=st.session_state["user_role"],
                    session_id=st.session_state["session_id"],
                )
                answer = rag_res.answer
                citations = rag_res.citations
            else:
                chain = get_qa_chain(tone_instruction=tone)
                raw_response = chain(question)
                answer = raw_response["result"]

    st.header("Answer")
    st.write(answer)

    # Citations
    if citations:
        with st.expander("📖 Sources & Citations", expanded=True):
            for cit in citations:
                st.markdown(f"- **{cit.title} (v{cit.version})** — `{cit.source_location}`: *{cit.snippet}*")

    # Escalation banner
    if escalation_record:
        routing_labels = {
            "on_call": "🚨 Escalated to on-call agent",
            "human_queue": "👤 Escalated to human support",
            "scheduled_next_working_day": "📅 Scheduled for next working day",
        }
        label = routing_labels.get(escalation_record.routing_outcome, "⚠️ Escalated")
        st.warning(f"{label} — {escalation_record.reason}")

# ── Structured Tickets Section (Module 2) ──────────────────────────────────────
st.divider()
st.subheader("🎫 Support Ticket Actions")
col_t1, col_t2 = st.columns([1, 2])
with col_t1:
    if st.button("➕ Create Ticket from Conversation"):
        # Fetch conversation messages for session
        rows = conn.execute(
            "SELECT sender, text FROM conversation_messages WHERE session_id = ? ORDER BY id ASC",
            (st.session_state["session_id"],),
        ).fetchall()
        msgs = [{"sender": r["sender"], "text": r["text"]} for r in rows]
        if not msgs and question:
            msgs = [{"sender": "user", "text": question}]

        if not msgs:
            st.warning("No conversation messages yet to create a ticket.")
        else:
            ticket, prompt = ticket_svc.create_ticket_from_conversation(
                session_id=st.session_state["session_id"],
                messages=msgs,
            )
            if prompt:
                st.info(f"Clarification needed to complete ticket: {prompt}")
            elif ticket:
                st.success(f"Ticket `{ticket.id}` created! Priority: `{ticket.priority.value}` | Assigned: `{ticket.assigned_team}`")
                st.rerun()

# Display active tickets
tickets = conn.execute(
    "SELECT * FROM tickets WHERE session_id = ? ORDER BY created_at DESC",
    (st.session_state["session_id"],),
).fetchall()

if tickets:
    st.markdown("**Session Support Tickets:**")
    for t in tickets:
        t_id = t["id"]
        pri = t["priority"]
        team = t["assigned_team"]
        status = t["status"]
        deadline = t["sla_deadline"]
        with st.container():
            st.write(f"🏷️ **{t_id}** | Priority: **{pri}** | Team: **{team}** | Status: `{status}` | SLA: `{deadline}`")
            handoff = ticket_svc.get_masked_handoff(t_id)
            if handoff:
                with st.expander(f"Agent Handoff Summary for {t_id}"):
                    st.text(handoff.masked_summary)

# ── Debug / Demo Sidebar Panel ─────────────────────────────────────────────────
if demo_enabled:
    with st.sidebar:
        st.header("🛠 Debug / Demo Panel")

        # Demo Scenarios Expander (matching README demo script)
        with st.expander("💡 Demo Scenarios (8 Scenarios)", expanded=False):
            st.markdown(
                """
                1. **Calm Duplicate Payment**:  
                   `"I was double charged on order ORD-1234. Please refund the duplicate payment."`
                2. **Consecutive Negative Escalation**:  
                   Send 3 consecutive frustrated messages to observe tone adjustment and human queue escalation.
                3. **Immediate High-Risk Bypass**:  
                   `"I will take legal action against your company for data breach."`
                4. **Historical Policy Query**:  
                   `"What was the refund policy on 15 March 2025?"` (retrieves v1).
                5. **Current Policy Query**:  
                   `"What is the current refund policy?"` (retrieves active v2).
                6. **Restricted Internal Guide**:  
                   Query `"Show me the internal supervisor escalation guide"` as `customer` (refused) vs `internal` (answered).
                7. **Ticket Creation**:  
                   Click *"Create Ticket from Conversation"* to verify entity extraction and SLA.
                8. **Discrepancy Detection**:  
                   State amount charged is ₹14,999 then upload receipt showing ₹9,999.
                """
            )

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
        metrics = kb_pipeline.get_metrics()
        if active_version:
            st.success(f"Active Version: `{active_version.id}` (State: `{active_version.state.value}`)")
            st.caption(f"Docs: {active_version.document_count} | Chunks: {active_version.chunk_count} | Quality: {active_version.quality_score:.2f}")
        else:
            st.info("Active Version: None (using dataset.csv fallback)")

        st.caption(f"Pipeline: {metrics.get('total_versions', 0)} versions | {metrics.get('quarantined_files', 0)} quarantined")

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
            due_file = file_svc.job_queue.tick(file_svc.get_job_handlers())
            due_kb = container.job_queue.run_due()
            total_jobs = len(due_esc) + len(due_file) + len(due_kb)
            st.success(f"Processed {total_jobs} background/escalation job(s).")
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
