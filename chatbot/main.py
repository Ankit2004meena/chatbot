import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.set_page_config(page_title="Customer Service Chatbot", page_icon="🤖")

st.title("CUSTOMER SERVICE CHATBOT 🤖")

# Button to create or refresh the knowledge base
if st.button("🧠 Create Knowledge Base"):
    with st.spinner("Creating knowledge base..."):
        create_vector_db()
    st.success("✅ Knowledge base created successfully!")

# Input box for user question
question = st.text_input("💬 Ask a question:")

# When a question is entered
if question:
    with st.spinner("Generating answer..."):
        chain = get_qa_chain()
        response = chain(question)

    st.subheader("🧾 Answer")
    st.write(response["result"])

