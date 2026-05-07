# AI Interview Coach – RAG-Based LLM System

An AI Interview Coach prototype built with OpenAI, RAG, ChromaDB, Sentence Transformers, and Streamlit.

## Features

- Generate personalized interview questions
- Evaluate interview answers with AI feedback
- Retrieve relevant context from local knowledge base
- Use ChromaDB as vector database
- Use Sentence Transformers for embeddings
- Test retrieval quality with separate scripts

## Technologies

- Python
- Streamlit
- OpenAI API
- ChromaDB
- sentence-transformers
- pypdf

## Project Structure

```text
app.py                  # Main Streamlit app
rag_utils.py            # RAG, embeddings, retrieval logic
build_knowledge_base.py # Builds ChromaDB knowledge base
external_knowledge/     # CV, job descriptions, sample answers, STAR guidelines
test_retrieval.py       # General retrieval test
test_retrieval_2.py     # CV-specific retrieval test
```

# How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/sehab1611251/ai-interview-coach-rag.git
cd ai-interview-coach-rag
```

### 2. Create and activate a virtual environment

Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install required packages

```bash
pip install streamlit openai chromadb sentence-transformers pypdf
```

### 4. Build the vector database

```bash
python build_knowledge_base.py
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

The app will open in your browser automatically.

### 6. Enter your OpenAI API key

Paste your OpenAI API key in the sidebar to use the application.
