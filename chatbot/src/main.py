import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.title("CUSTOMER SERVICE CHATBOT 🤖")

if st.button("Create Knowledgebase"):
    create_vector_db()

question = st.text_input("Question:")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])






# langchain-community==0.2.3
# langchain==0.2.3
# langchain-google-genai==1.0.10
# streamlit==1.35.0
# sentence-transformers==2.2.2
# huggingface-hub==0.16.4
# transformers==4.29.2
# faiss-cpu==1.7.4
# python-dotenv==1.0.1
# dotenv

# langchain-community==0.2.1
# streamlit
# sentence-transformers==2.2.2
# huggingface-hub==0.16.4
# transformers==4.29.2
# faiss-cpu
# google-generativeai==0.3.2
# python-dotenv
# langchain









# import streamlit as st
# # from langchain_helper import get_qa_chain, create_vector_db
# from langchain_helper import get_qa_chain, create_vector_db


# st.title(" CUSTOMER SERVICE CHATBOT 🤖")
# btn = st.button("Create Knowledgebase")
# if btn:
#     create_vector_db()

# question = st.text_input("Question: ")

# if question:
#     chain = get_qa_chain()
#     response = chain(question)

#     st.header("Answer")
#     st.write(response["result"])
