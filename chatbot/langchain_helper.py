import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import CSVLoader

# ✅ Load environment variables
load_dotenv()

# ✅ Paths

csv_path = "dataset/dataset.csv"

vectordb_file_path = "faiss_index"

# ✅ Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-small")

# ✅ LLM setup
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1
)

# ✅ Create FAISS vector DB
def create_vector_db():
    loader = CSVLoader(file_path=csv_path, source_column="prompt")
    data = loader.load()
    vectordb = FAISS.from_documents(data, embedding=embedding_model)
    vectordb.save_local(folder_path=vectordb_file_path)
    print("✅ Vector DB created and saved.")

# ✅ Load FAISS and build QA chain
def get_qa_chain():
    if not os.path.exists(os.path.join(vectordb_file_path, "index.faiss")):
        print("Vector DB not found. Creating new one...")
        create_vector_db()

    vectordb = FAISS.load_local(
        folder_path=vectordb_file_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
In the answer try to provide as much text as possible from the "response" section in the source document context without making many changes.
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

# ✅ Optional: quick test
if __name__ == "__main__":
    qa = get_qa_chain()
    result = qa("Hello?")
    print("Answer:", result["result"])
