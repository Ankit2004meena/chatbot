import os
from dotenv import load_dotenv
#from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
#from langchain.chains import RetrievalQA
from langchain.llms import GooglePalm
from google.ai.generativelanguage_v1beta.types import content
print(dir(content))


from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API keys
load_dotenv()

# Paths
csv_path = r"C:\Users\Ankur\Desktop\Nullclass 2nd project\mine\chatbot\dataset\dataset.csv"
vectordb_file_path = "faiss_index"

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-small")


# LLM
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key from .env
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1
)


# 📌 Create vector database
def create_vector_db():
    loader = CSVLoader(file_path=csv_path, source_column="prompt")
    data = loader.load()

    # Create new FAISS vector DB
    vectordb = FAISS.from_documents(data, embedding=embedding_model)
    vectordb.save_local(folder_path=vectordb_file_path)
    print("✅ Vector DB created and saved.")

# 📌 Load vector DB and build RetrievalQA chain
def get_qa_chain():
    vectordb = FAISS.load_local(
        folder_path=vectordb_file_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True  # Needed for .pkl safety
    )

    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

CONTEXT: {context}

QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain

# 📌 Test block (optional)
if __name__ == "__main__":
    create_vector_db()
    qa = get_qa_chain()
    result = qa("Hello?")
    print("Answer:", result["result"])
