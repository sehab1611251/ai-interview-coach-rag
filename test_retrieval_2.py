from rag_utils import retrieve_relevant_chunks


def print_cv_results(chunks):
    print("\n" + "=" * 80)
    print("TEST - CV RETRIEVAL ONLY")
    print("=" * 80)

    cv_chunks = [
        chunk for chunk in chunks
        if chunk["metadata"].get("source_file") == "CV.pdf"
    ]

    if not cv_chunks:
        print("❌ No CV chunks retrieved.")
        return

    for i, chunk in enumerate(cv_chunks, start=1):
        print(f"\n--- CV Result {i} ---")
        print(f"ID       : {chunk['id']}")
        print(f"Distance : {chunk['distance']:.4f}")
        print("Text:")
        print(chunk["text"][:700])
        print()


def main():
    # ----------------------------------------
    # Query specifically targeting CV content
    # ----------------------------------------
    cv_query = (
    #"Detailed CV including education, work experience, projects, technical skills, and achievements in AI"
    "Candidate projects in AI, machine learning, NLP and computer vision"
)

    chunks = retrieve_relevant_chunks(
    query=cv_query,
    top_k=5,
    where={"source_file": "CV.pdf"}   
    )

    print_cv_results(chunks)


if __name__ == "__main__":
    main()