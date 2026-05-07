import os
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# -----------------------------
# Constants
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "external_knowledge")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "ai_interview_knowledge"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# -----------------------------
# Embedding function wrapper
# -----------------------------
class SentenceTransformerEmbeddingFunction:
    """
    Custom embedding function for ChromaDB using sentence-transformers.
    Compatible with newer Chroma versions.
    """
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Fallback callable interface.
        """
        return self.embed_documents(input)

    def name(self) -> str:
        """
        Chroma expects embedding functions to expose a name.
        """
        return f"sentence_transformer_{self.model_name}"

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        """
        embeddings = self.model.encode(input, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, input: str) -> List[float]:
        """
        Embed a single query string.
        """
        embedding = self.model.encode(input, normalize_embeddings=True)
        return embedding.tolist()


# -----------------------------
# Chroma client / collection
# -----------------------------
def get_chroma_client():
    """
    Returns a persistent local ChromaDB client.
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_or_create_collection():
    """
    Returns the knowledge collection. Uses cosine space for retrieval.
    """
    client = get_chroma_client()
    embedding_function = SentenceTransformerEmbeddingFunction()

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
    except Exception:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

    return collection


def load_collection():
    """
    Loads the existing collection.
    """
    client = get_chroma_client()
    embedding_function = SentenceTransformerEmbeddingFunction()

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )
    return collection


# -----------------------------
# Chunking utilities
# -----------------------------
def chunk_text(
    text: str,
    chunk_size: int = 700,
    chunk_overlap: int = 120
) -> List[str]:
    """
    Splits text into overlapping chunks.
    Character-based chunking is enough for a simple prototype.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - chunk_overlap

    return chunks


# -----------------------------
# Retrieval
# -----------------------------
def retrieve_relevant_chunks(
    query: str,
    top_k: int = 4,
    where: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k relevant chunks from the Chroma collection.
    """
    collection = load_collection()

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    retrieved = []
    for doc_id, doc, meta, distance in zip(ids, documents, metadatas, distances):
        retrieved.append({
            "id": doc_id,
            "text": doc,
            "metadata": meta,
            "distance": distance
        })

    return retrieved


def format_retrieved_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Formats retrieved chunks into a prompt-friendly context block.
    """
    if not chunks:
        return "No relevant external knowledge found."

    formatted_parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk["metadata"].get("source_file", "unknown_source")
        section = chunk["metadata"].get("section", "general")
        text = chunk["text"]

        formatted_parts.append(
            f"[Context {i}] Source: {source} | Section: {section}\n{text}"
        )

    return "\n\n".join(formatted_parts)


# -----------------------------
# Query builders
# -----------------------------
def build_question_generation_query(
    target_role: str,
    skills: str,
    job_description: str
) -> str:
    """
    Query for retrieving context useful for generating interview questions.
    """
    return (
        f"Generate interview questions for role: {target_role}. "
        f"Relevant skills: {skills}. "
        f"Job description context: {job_description}. "
        f"Find candidate profile, project experience, and relevant job expectations."
    )


def build_answer_evaluation_query(
    target_role: str,
    questions: List[str],
    answers: List[str]
) -> str:
    """
    Query for retrieving context useful for evaluating answers.
    """
    joined_questions = " ".join(questions)
    joined_answers = " ".join(answers)

    return (
        f"Evaluate interview answers for role: {target_role}. "
        f"Questions: {joined_questions}. "
        f"Answers: {joined_answers}. "
        f"Find STAR guidelines, sample answers, candidate experience, and useful evaluation criteria."
    )