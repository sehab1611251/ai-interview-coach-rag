from rag_utils import (
    retrieve_relevant_chunks,
    format_retrieved_context,
    build_question_generation_query,
    build_answer_evaluation_query
)


def print_results(title: str, chunks):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if not chunks:
        print("No chunks retrieved.")
        return

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Result {i} ---")
        print(f"ID       : {chunk['id']}")
        print(f"Source   : {chunk['metadata'].get('source_file')}")
        print(f"Section  : {chunk['metadata'].get('section')}")
        print(f"Distance : {chunk['distance']:.4f}")
        print("Text:")
        print(chunk["text"][:700])
        print()


def main():
    # ----------------------------------------
    # Test 1: question generation retrieval
    # ----------------------------------------
    question_query = build_question_generation_query(
        target_role="Junior AI Engineer",
        skills="Python, machine learning, computer vision, NLP, Azure",
        job_description=(
            "Build AI-powered applications, support machine learning pipelines, "
            "work with LLM APIs, evaluate AI prototypes, and collaborate with product teams."
        )
    )

    question_chunks = retrieve_relevant_chunks(
        query=question_query,
        top_k=4
    )
    print_results("TEST 1 - QUESTION GENERATION RETRIEVAL", question_chunks)

    print("\nFormatted context for prompt:\n")
    print(format_retrieved_context(question_chunks))

    # ----------------------------------------
    # Test 2: answer evaluation retrieval
    # ----------------------------------------
    evaluation_query = build_answer_evaluation_query(
        target_role="Junior AI Engineer",
        questions=[
            "Tell me about yourself.",
            "Describe a technical project you are proud of.",
            "Why are you interested in this role?"
        ],
        answers=[
            "I recently completed my Master’s in AI and Data Engineering and worked on computer vision and NLP projects.",
            "My thesis involved 3D group detection using DBSCAN on LiDAR-based pedestrian detections.",
            "I like roles where I can apply AI in real products and continue growing technically."
        ]
    )

    evaluation_chunks = retrieve_relevant_chunks(
        query=evaluation_query,
        top_k=5
    )
    print_results("TEST 2 - ANSWER EVALUATION RETRIEVAL", evaluation_chunks)

    print("\nFormatted context for prompt:\n")
    print(format_retrieved_context(evaluation_chunks))


if __name__ == "__main__":
    main()