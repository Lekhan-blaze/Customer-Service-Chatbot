# Support Chatbot Extended

A customer service question-answering chatbot repository initialized for internship project extensions.

## Original Features (Training Project)

The baseline project provides an interactive FAQ customer service chatbot designed to answer customer queries based on an educational program FAQ dataset:

- **Web Interface (`main.py`)**: A Streamlit application allowing users to trigger knowledgebase creation and enter customer service questions to receive context-grounded answers.
- **LLM**: Google Gemini (`gemini-3.8-flash` via `langchain-google-genai`) with low temperature (0.1) for deterministic responses.
- **Embeddings**: `hkunlp/instructor-large` embeddings via `HuggingFaceInstructEmbeddings`.
- **Knowledge Base & Vector Store (`langchain_helper.py`)**: Uses LangChain's `CSVLoader` to parse `dataset/dataset.csv` (using the `prompt` column as document source) and builds/loads a local FAISS vector database stored in `faiss_index/`.
- **Question-Answering Chain**: Implements a LangChain `RetrievalQA` pipeline ("stuff" chain type) with a custom prompt template ensuring answers strictly adhere to retrieved FAQ context, falling back to *"I don't know."* when information is absent.
- **Dataset (`dataset/dataset.csv`)**: 76 question-and-answer pairs covering course prerequisites, enrollment, job assistance, and program details across two columns (`prompt` and `response`).

> **Note on Future Work**: New internship modules, architectures, and expanded features will be added below as project milestones are developed.

---

## Setup & Running Locally

1. **Python Environment**:
   Ensure Python 3.11 is used. Create and activate a virtual environment:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Copy `.env.example` to `.env` and set your API key:
   ```bash
   cp .env.example .env
   ```
   Add your `GOOGLE_API_KEY` in `.env`.

3. **Run Application**:
   ```bash
   streamlit run main.py
   ```
