import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

# App title
st.title("CUSTOMER SERVICE CHATBOT 🤖")

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
    with st.spinner("Finding answer..."):
        chain = get_qa_chain()
        response = chain(question)

    st.header("Answer")
    st.write(response["result"])