import os
from dotenv import load_dotenv

# Document loading and vector storage
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

# LLM and prompts
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Retrieval QA chain
try:
    from langchain_classic.chains import RetrievalQA
except ImportError:
    # pyrefly: ignore [missing-import]
    from langchain.chains import RetrievalQA

load_dotenv()  # Load environment variables (.env)

# Initialize Gemini LLM
llm = GoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.1,
)

# Initialize instructor embeddings
instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large"
)
vectordb_file_path = "faiss_index"


def create_vector_db():
    # Load FAQ dataset
    csv_path = "dataset/dataset.csv" if os.path.exists("dataset/dataset.csv") else "dataset.csv"
    loader = CSVLoader(file_path=csv_path, source_column="prompt", encoding="utf-8")
    data = loader.load()

    # Build and save FAISS vector database locally
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    # Automatically build vector DB if it hasn't been created yet
    if not os.path.exists(vectordb_file_path):
        create_vector_db()

    # Load local vector database
    vectordb = FAISS.load_local(
        vectordb_file_path,
        instructor_embeddings,
        allow_dangerous_deserialization=True,
    )

    # Create retriever
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Build QA retrieval chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    return chain


if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("hello?"))