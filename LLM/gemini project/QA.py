from dotenv import load_dotenv
load_dotenv()  # Take environment variables from .env

import streamlit as st
import os
import google.generativeai as genai

from IPython.display import display, Markdown

# Load and configure API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBA6ddXZhK5j3Nbmv4_AdrUQcHqcoGJMLk"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize the Gemini model
model = genai.GenerativeModel(model_name="models/gemma-3-4b-it")
chat = model.start_chat(history=[])

# Function to send a prompt and receive streamed response
def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

# Streamlit UI setup
st.set_page_config(page_title="GEMINI CHATBOT DEMO")
st.header("Gemini Application")

input = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")

if submit:
    response = get_gemini_response(input)
    st.subheader("The Response is")

    full_response = ""  # Collect all parts

    for chunk in response:
        try:
            if chunk.candidates and chunk.candidates[0].content.parts:
                part = chunk.candidates[0].content.parts[0].text
                if part.strip():
                    full_response += part
        except Exception as e:
            st.error(f"Chunk error: {e}")

    if full_response.strip():
        st.write(full_response)
    else:
        st.warning("The model did not return any content.")

    st.subheader("Conversation History:")
for message in chat.history:
    role = message.role
    content = message.parts[0].text if message.parts else ""
    if role == "user":
        st.markdown(f"🧑 **You:** {content}")
    elif role == "model":
        st.markdown(f"🤖 **Gemini:** {content}")
    # response = get_gemini_response(input)
    # st.subheader("The Response is")
    # for chunk in response:
    #     print(st.write(chunk.text))
    #     print("_" * 80)

    # st.write(chat.history)
