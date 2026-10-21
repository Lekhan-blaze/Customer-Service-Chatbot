import uuid
import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db
from support_bot.config import get_config
from support_bot.db import get_connection
from support_bot.conversation.service import ConversationService

# App title
st.title("CUSTOMER SERVICE CHATBOT 🤖")

# Initialize session ID in st.session_state
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

# Build vector knowledgebase
btn = st.button("Create Knowledgebase")
if btn:
    with st.spinner("Creating knowledgebase..."):
        create_vector_db()
    st.success("Knowledgebase created successfully!")

# User query input
question = st.text_input("Question:")

# Generate and display answer
if question:
    # Check if conversation module is enabled in configuration
    conn = get_connection()
    cfg = get_config(conn)
    conv_enabled = cfg.get("conversation", {}).get("enabled", True)

    with st.spinner("Finding answer..."):
        if conv_enabled:
            service = ConversationService(conn=conn, config=cfg)
            conv_resp = service.handle_message(st.session_state["session_id"], question)

            if conv_resp.needs_clarification:
                answer = conv_resp.clarification_question
            else:
                query_to_run = conv_resp.normalized_query or question
                chain = get_qa_chain()
                raw_response = chain(query_to_run)
                answer = service.format_reply(conv_resp, raw_response["result"])
        else:
            # Original direct QA behavior
            chain = get_qa_chain()
            raw_response = chain(question)
            answer = raw_response["result"]

    st.header("Answer")
    st.write(answer)