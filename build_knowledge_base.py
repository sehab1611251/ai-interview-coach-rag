import os
import re
from typing import List, Dict

from pypdf import PdfReader

from rag_utils import (
    KNOWLEDGE_DIR,
    get_or_create_collection,
    chunk_text
)


# --------------------------------------------------
# File readers
# --------------------------------------------------
def read_pdf_text(pdf_path: str) -> str:
    """
    Extract text from a PDF file using pypdf.
    """
    reader = PdfReader(pdf_path)
    pages_text = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)

    return "\n".join(pages_text).strip()


def read_txt_text(txt_path: str) -> str:
    """
    Read a plain text file.
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


# --------------------------------------------------
# Smart block splitters
# --------------------------------------------------
def split_job_descriptions(text: str) -> List[str]:
    """
    Split the job descriptions file by each 'JOB DESCRIPTION' block.

    This is better than raw character chunking because each job description
    remains semantically grouped.
    """
    parts = re.split(r"(?=JOB DESCRIPTION \d+)", text)
    return [part.strip() for part in parts if part.strip()]


def split_sample_answers(text: str) -> List[str]:
    """
    Split the sample answers file by each 'SAMPLE ANSWER' block.

    This keeps each sample answer together, which is useful for retrieval.
    """
    parts = re.split(r"(?=SAMPLE ANSWER \d+)", text)
    return [part.strip() for part in parts if part.strip()]


def split_star_guidelines(text: str) -> List[str]:
    """
    Split the STAR guidelines file into larger semantic sections.

    This is done by splitting around long separator lines and major section titles.
    The goal is to preserve meaningful instructional blocks.
    """
    # First split by long separator lines if present
    rough_parts = re.split(r"\n[-]{10,}\n", text)

    # Clean empty pieces
    rough_parts = [part.strip() for part in rough_parts if part.strip()]

    # If the split was too weak, fall back to splitting by major headings
    if len(rough_parts) <= 2:
        rough_parts = re.split(r"(?=\d+\.\s|\bSTAR EXAMPLE\b|\bHOW TO USE STAR WELL\b|\bHOW THIS FILE CAN HELP THE APP\b)", text)
        rough_parts = [part.strip() for part in rough_parts if part.strip()]

    return rough_parts


# --------------------------------------------------
# Document loading
# --------------------------------------------------
def load_knowledge_sources() -> List[Dict]:
    """
    Load all knowledge files from the external_knowledge directory.

    Returns a list of dictionaries with:
    - filename
    - section
    - text
    """
    files = [
        ("CV.pdf", "cv"),
        ("job_descriptions.txt", "job_description"),
        ("sample_answers.txt", "sample_answer"),
        ("star_guidelines.txt", "star_guideline"),
    ]

    sources = []

    for filename, section in files:
        file_path = os.path.join(KNOWLEDGE_DIR, filename)

        if not os.path.exists(file_path):
            print(f"[WARNING] File not found: {file_path}")
            continue

        if filename.lower().endswith(".pdf"):
            text = read_pdf_text(file_path)
        else:
            text = read_txt_text(file_path)

        sources.append({
            "filename": filename,
            "section": section,
            "text": text
        })

    return sources


# --------------------------------------------------
# Chunk preparation
# --------------------------------------------------
def prepare_chunks() -> List[Dict]:
    """
    Convert all knowledge sources into chunked records with metadata.

    Strategy:
    - CV.pdf -> normal chunking
    - job_descriptions.txt -> split by JOB DESCRIPTION, then chunk if needed
    - sample_answers.txt -> split by SAMPLE ANSWER, then chunk if needed
    - star_guidelines.txt -> split by semantic section, then chunk if needed
    """
    sources = load_knowledge_sources()
    records = []

    for source in sources:
        filename = source["filename"]
        section = source["section"]
        text = source["text"]

        block_list: List[str] = []

        # ------------------------------------------
        # Choose splitting strategy by source type
        # ------------------------------------------
        if filename == "CV.pdf":
            # CV is handled with regular chunking
            block_list = [text]

        elif filename == "job_descriptions.txt":
            block_list = split_job_descriptions(text)

        elif filename == "sample_answers.txt":
            block_list = split_sample_answers(text)

        elif filename == "star_guidelines.txt":
            block_list = split_star_guidelines(text)

        else:
            # Fallback
            block_list = [text]

        # ------------------------------------------
        # Chunk each block if needed
        # ------------------------------------------
        chunk_counter = 0

        for block_index, block in enumerate(block_list):
            block_chunks = chunk_text(
                block,
                chunk_size=700,
                chunk_overlap=120
            )

            for local_chunk_index, chunk in enumerate(block_chunks):
                record = {
                    "id": f"{section}_{block_index}_{local_chunk_index}",
                    "document": chunk,
                    "metadata": {
                        "source_file": filename,
                        "section": section,
                        "block_index": block_index,
                        "chunk_index": chunk_counter
                    }
                }
                records.append(record)
                chunk_counter += 1

    return records


# --------------------------------------------------
# Build knowledge base
# --------------------------------------------------
def build_knowledge_base():
    """
    Build or rebuild the local Chroma knowledge collection.

    Steps:
    1. Open or create the collection
    2. Remove old records
    3. Prepare new chunks
    4. Insert them into Chroma
    5. Print summary
    """
    collection = get_or_create_collection()

    # Remove old content if it exists
    existing = collection.get()
    existing_ids = existing.get("ids", [])

    if existing_ids:
        collection.delete(ids=existing_ids)
        print(f"[INFO] Deleted {len(existing_ids)} old chunks from collection.")

    records = prepare_chunks()

    if not records:
        print("[ERROR] No knowledge records prepared. Check your files.")
        return

    ids = [record["id"] for record in records]
    documents = [record["document"] for record in records]
    metadatas = [record["metadata"] for record in records]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    print("[SUCCESS] Knowledge base built successfully.")
    print(f"[INFO] Total chunks stored: {len(records)}")

    # Print chunk summary by source
    by_source = {}
    for record in records:
        source_file = record["metadata"]["source_file"]
        by_source[source_file] = by_source.get(source_file, 0) + 1

    print("[INFO] Chunk counts by source:")
    for source_file, count in by_source.items():
        print(f"  - {source_file}: {count} chunks")


# --------------------------------------------------
# Main entry point
# --------------------------------------------------
if __name__ == "__main__":
    build_knowledge_base()