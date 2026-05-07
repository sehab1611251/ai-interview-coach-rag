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