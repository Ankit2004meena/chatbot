import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
import google.generativeai as genai

load_dotenv()

csv_path = "dataset/dataset.csv"
vectordb_file_path = "faiss_index"

embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-small")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash")  

def create_vector_db():
    
    loader = CSVLoader(
        file_path=csv_path,
        source_column="prompt",  
        encoding="ISO-8859-1"
    )
    data = loader.load()
    vectordb = FAISS.from_documents(data, embedding=embedding_model)
    vectordb.save_local(folder_path=vectordb_file_path)
    print("✅ Vector DB created and saved successfully!")

def load_retriever():

    if not os.path.exists(os.path.join(vectordb_file_path, "index.faiss")):
        print("Vector DB not found. Creating a new one...")
        create_vector_db()

    vectordb = FAISS.load_local(
        folder_path=vectordb_file_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3}, score_threshold=0.7)
    return retriever

def get_gemini_response(question: str):
 
    retriever = load_retriever()
    docs = retriever.get_relevant_documents(question)

    if not docs:
        return {"result": "I don't know.", "source_documents": []}

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an intelligent AI assistant.
Use the following context to answer the question.
If the answer is not found in the context, reply with "I don't know."

Context:
{context}

Question:
{question}
"""

    try:
        response = gemini_model.generate_content(prompt)
        return {
            "result": response.text.strip(),
            "source_documents": docs
        }
    except Exception as e:
        print("⚠️ Gemini API Error:", e)
        return {
            "result": "There was an error getting a response from Gemini.",
            "source_documents": docs
        }

def get_qa_chain():
    def chain(question):
        return get_gemini_response(question)
    return chain