import os
from dotenv import load_dotenv

# Document loading and vector storage
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS

# Embeddings — migrated from deprecated HuggingFaceInstructEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

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

# ---------------------------------------------------------------------------
# Config constants — change the embedding model here, not in multiple places
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
VECTORDB_FILE_PATH = "faiss_index"

# ---------------------------------------------------------------------------
# Initialize Gemini LLM
# ---------------------------------------------------------------------------
llm = GoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "fake_key_for_offline_testing",
    temperature=0.1,
)

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def create_vector_db():
    """Load FAQ dataset and build a local FAISS vector database."""
    csv_path = (
        "dataset/dataset.csv"
        if os.path.exists("dataset/dataset.csv")
        else "dataset.csv"
    )
    # Dataset was converted to UTF-8 — keep encoding explicit
    loader = CSVLoader(file_path=csv_path, source_column="prompt", encoding="utf-8")
    data = loader.load()

    # Build and persist FAISS index
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local(VECTORDB_FILE_PATH)


def get_qa_chain(tone_instruction: str = "neutral"):
    """Return a RetrievalQA chain backed by the local FAISS index.

    Args:
        tone_instruction: One of "empathetic", "concise", or "neutral" (default).
            Adjusts reply phrasing only — the factual content always comes from
            the retrieved FAQ context. Never adds commitments not in the context.

    Automatically creates the index if it has not been built yet.
    """
    # Automatically build vector DB if it hasn't been created yet
    if not os.path.exists(VECTORDB_FILE_PATH):
        create_vector_db()

    # Load local vector database
    vectordb = FAISS.load_local(
        VECTORDB_FILE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # Create retriever
    retriever = vectordb.as_retriever(score_threshold=0.7)

    # ── Tone phrasing instructions (wording only; never adds un-retrieved facts) ──
    _tone_hints = {
        "empathetic": (
            "The customer seems upset. Acknowledge their concern warmly before "
            "providing the answer. Use a compassionate, understanding tone. "
            "Do NOT add any promises or information not found in the context."
        ),
        "concise": (
            "The customer needs a quick resolution. Be brief and direct. "
            "Avoid filler phrases. Do NOT add any information not found in the context."
        ),
        "neutral": "",
    }
    tone_hint = _tone_hints.get(tone_instruction, "")
    tone_section = f"\n    Tone guidance: {tone_hint}" if tone_hint else ""

    prompt_template = (
        "Given the following context and a question, generate an answer based on this context only.\n"
        "    In the answer try to provide as much text as possible from \"response\" section in the source document context without making much changes.\n"
        "    If the answer is not found in the context, kindly state \"I don't know.\" Don't try to make up an answer."
        + tone_section
        + "\n\n    CONTEXT: {context}\n\n    QUESTION: {question}"
    )

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