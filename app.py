import re
import streamlit as st
from openai import OpenAI

# ------------------------------------------------------------
# NEW IMPORTS FOR RAG INTEGRATION
# ------------------------------------------------------------
from rag_utils import (
    build_question_generation_query,
    build_answer_evaluation_query,
    retrieve_relevant_chunks,
    format_retrieved_context
)

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🎯",
    layout="wide"
)

# ------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------
# ------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------
st.markdown("""
<style>
/* Global app styling */
html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
    color: #0f172a;
}

/* Main page spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Hero section */
.hero-card {
    background: linear-gradient(135deg, #2563eb 0%, #4f46e5 60%, #7c3aed 100%);
    padding: 2rem 2rem 1.5rem 2rem;
    border-radius: 24px;
    color: white;
    box-shadow: 0 18px 40px rgba(37, 99, 235, 0.18);
    margin-bottom: 1.2rem;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
    line-height: 1.15;
}

.hero-subtitle {
    font-size: 1.02rem;
    line-height: 1.6;
    opacity: 0.96;
    margin-bottom: 1rem;
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.badge {
    background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.22);
    color: white;
    padding: 0.42rem 0.78rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
}

/* General section cards */
.section-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 22px;
    padding: 1.2rem;
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.05);
    margin-bottom: 1rem;
}

.section-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0.35rem;
}

.section-subtitle {
    color: #475569;
    font-size: 0.95rem;
    margin-bottom: 0.8rem;
    line-height: 1.5;
}

/* Step label */
.step-chip {
    display: inline-block;
    background: #e0e7ff;
    color: #3730a3;
    font-weight: 700;
    font-size: 0.82rem;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    margin-bottom: 0.6rem;
}

/* Question card */
.question-card {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    border: 1px solid #dbeafe;
    border-radius: 18px;
    padding: 1rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
}

.question-label {
    color: #2563eb;
    font-size: 0.8rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.question-text {
    color: #0f172a;
    font-size: 1rem;
    line-height: 1.55;
    font-weight: 600;
}

/* Mini cards */
.mini-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 1rem;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
    height: 100%;
}

.mini-card h4 {
    margin: 0 0 0.35rem 0;
    font-size: 1rem;
    color: #0f172a;
}

.mini-card p {
    margin: 0;
    color: #475569;
    line-height: 1.5;
    font-size: 0.92rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e3a8a 100%);
}

[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #ffffff;
}

[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stTextArea label,
[data-testid="stSidebar"] .stCheckbox label {
    color: #e2e8f0 !important;
}

/* Sidebar input wrappers */
[data-testid="stSidebar"] div[data-baseweb="input"] > div,
[data-testid="stSidebar"] div[data-baseweb="base-input"] > div,
[data-testid="stSidebar"] .stTextInput > div > div,
[data-testid="stSidebar"] .stTextArea textarea,
[data-testid="stSidebar"] textarea {
    background: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
    border-radius: 12px !important;
}

/* Sidebar actual input text */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #ffffff !important;
    opacity: 1 !important;
}

/* Sidebar placeholder */
[data-testid="stSidebar"] input::placeholder,
[data-testid="stSidebar"] textarea::placeholder {
    color: rgba(226,232,240,0.75) !important;
    -webkit-text-fill-color: rgba(226,232,240,0.75) !important;
    opacity: 1 !important;
}

/* Sidebar focus */
[data-testid="stSidebar"] input:focus,
[data-testid="stSidebar"] textarea:focus {
    outline: none !important;
    box-shadow: none !important;
}

/* Main input wrappers */
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] > div {
    background: #ffffff !important;
    border: 1px solid #dbe4f0 !important;
    border-radius: 14px !important;
}

/* Main area input fields */
.stTextInput input,
.stTextArea textarea,
input,
textarea {
    border-radius: 14px !important;
}

/* Main area text visibility */
.main input,
.main textarea,
.stTextInput input,
.stTextArea textarea {
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    caret-color: #0f172a !important;
    background-color: transparent !important;
    opacity: 1 !important;
}

/* Main area placeholder */
.main input::placeholder,
.main textarea::placeholder,
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #94a3b8 !important;
    -webkit-text-fill-color: #94a3b8 !important;
    opacity: 1 !important;
}

/* Buttons */
.stButton > button {
    border-radius: 14px;
    padding: 0.72rem 1rem;
    font-weight: 700;
    border: none;
    background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
    color: white !important;
    box-shadow: 0 10px 20px rgba(37, 99, 235, 0.18);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    filter: brightness(1.03);
}

/* Metrics */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 0.8rem;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
}

/* Expander */
.streamlit-expanderHeader {
    font-weight: 700;
    color: #1e293b;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 700;
    border-radius: 12px 12px 0 0;
}

/* Evaluation box */
.evaluation-wrap {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 1.25rem;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
}

/* Helper note */
.soft-note {
    color: #64748b;
    font-size: 0.9rem;
    margin-top: -0.2rem;
    margin-bottom: 0.6rem;
    line-height: 1.5;
}

/* Make markdown lists and paragraphs cleaner */
.main p, .main li {
    font-size: 0.97rem;
    line-height: 1.65;
    color: #334155;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def clean_questions(raw_text: str) -> list[str]:
    """
    Convert the model output into a clean list of questions.

    The model may return:
    - numbered lines
    - bullet points
    - extra empty lines

    This function removes numbering/bullets and keeps up to 3 questions.
    """
    lines = raw_text.splitlines()
    questions = []

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Remove numbering like "1. ", "2) ", "- ", "* "
        line = re.sub(r"^\s*(\d+[\.\)]\s*|[-*]\s*)", "", line).strip()

        # Keep only meaningful lines
        if line:
            questions.append(line)

    # Return only the first 3 questions
    return questions[:3]


def configure_openai(api_key: str):
    """
    Configure OpenAI with the user-provided API key
    and return the client instance.
    """
    return OpenAI(api_key=api_key)


def render_section_header(step_text: str, title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="section-card">
            <div class="step-chip">{step_text}</div>
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# UPDATED: QUESTION GENERATION NOW USES RAG CONTEXT
# ------------------------------------------------------------
def generate_interview_questions(
    client,
    target_role: str,
    skills: str,
    job_description: str,
    degree: str,
    english_level: str,
    italian_level: str
) -> tuple[list[str], str]:
    """
    Use OpenAI to generate 3 interview questions.

    NEW:
    - Build a retrieval query
    - Retrieve relevant chunks from local knowledge base
    - Inject retrieved context into the prompt
    - Return both questions and retrieved context
    """

    retrieval_query = build_question_generation_query(
        target_role=target_role,
        skills=skills,
        job_description=job_description
    )

    retrieved_chunks = retrieve_relevant_chunks(
        query=retrieval_query,
        top_k=4
    )

    retrieved_context = format_retrieved_context(retrieved_chunks)

    prompt = f"""
You are an interview coach helping international students prepare for jobs in Italy.

Use the external retrieved context below to make your questions more personalized, relevant, and grounded.

Retrieved context:
{retrieved_context}

Candidate information:
- Degree: {degree}
- Target role: {target_role}
- English level: {english_level}
- Italian level: {italian_level}
- Skills: {skills}

Job description:
{job_description}

Task:
Generate exactly 3 realistic interview questions for this candidate.

Rules:
- Questions should fit a junior or entry-level candidate.
- Use the retrieved context when relevant.
- Questions should be specific, practical, and role-aware.
- Return only the 3 questions, one per line.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an interview coach helping international students "
                    "prepare for job interviews."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    text = response.choices[0].message.content.strip() if response.choices else ""
    questions = clean_questions(text)

    return questions, retrieved_context


# ------------------------------------------------------------
# UPDATED: ANSWER EVALUATION NOW USES RAG CONTEXT
# ------------------------------------------------------------
def evaluate_answers_with_openai(
    client,
    student_name: str,
    degree: str,
    target_role: str,
    skills: str,
    job_description: str,
    questions: list[str],
    answers: list[str]
) -> tuple[str, str]:
    """
    Use OpenAI to evaluate the user's interview answers.

    NEW:
    - Build a retrieval query for evaluation
    - Retrieve relevant sample answers / STAR guidelines / CV context
    - Inject retrieved context into the evaluation prompt
    - Return both evaluation text and retrieved context
    """

    retrieval_query = build_answer_evaluation_query(
        target_role=target_role,
        questions=questions,
        answers=answers
    )

    retrieved_chunks = retrieve_relevant_chunks(
        query=retrieval_query,
        top_k=5
    )

    retrieved_context = format_retrieved_context(retrieved_chunks)

    prompt = f"""
You are an interview evaluator and coach.

Use the retrieved external context below to make the evaluation more grounded and helpful.
The context may include:
- sample good answers
- job expectations
- candidate background
- STAR structure guidance

Retrieved context:
{retrieved_context}

Candidate:
- Name: {student_name}
- Degree: {degree}
- Target role: {target_role}
- Skills: {skills}

Job description:
{job_description}

Interview questions and answers:

Question 1: {questions[0] if len(questions) > 0 else ""}
Answer 1: {answers[0] if len(answers) > 0 else ""}

Question 2: {questions[1] if len(questions) > 1 else ""}
Answer 2: {answers[1] if len(answers) > 1 else ""}

Question 3: {questions[2] if len(questions) > 2 else ""}
Answer 3: {answers[2] if len(answers) > 2 else ""}

Task:
Evaluate each answer separately.

For each answer, provide:
1. Score out of 10
2. Strengths
3. Weaknesses
4. Improved version of the answer

Then provide:
5. Overall feedback
6. Final improvement advice

If useful, use STAR-style feedback when the answer is vague or weakly structured.

Format the output clearly with headings.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an interview evaluator and coach. "
                    "Provide clear, practical, and structured feedback."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    evaluation_text = (
        response.choices[0].message.content.strip()
        if response.choices
        else "No evaluation received from OpenAI."
    )

    return evaluation_text, retrieved_context


# ------------------------------------------------------------
# Session state initialization
# ------------------------------------------------------------
if "questions" not in st.session_state:
    st.session_state.questions = []

if "generated" not in st.session_state:
    st.session_state.generated = False

if "evaluated" not in st.session_state:
    st.session_state.evaluated = False

if "evaluation_text" not in st.session_state:
    st.session_state.evaluation_text = ""

if "question_rag_context" not in st.session_state:
    st.session_state.question_rag_context = ""

if "evaluation_rag_context" not in st.session_state:
    st.session_state.evaluation_rag_context = ""


# ------------------------------------------------------------
# HERO SECTION
# ------------------------------------------------------------
st.markdown("""
<div class="hero-card">
    <div class="hero-title">AI Interview Coach</div>
    <div class="hero-subtitle">
        Practice smarter, build stronger answers, and get clearer interview feedback with a
        RAG-enhanced coaching experience for international students.
    </div>
    <div class="badge-row">
        <span class="badge">OpenAI-Powered</span>
        <span class="badge">RAG-Enhanced</span>
        <span class="badge">Question Generation</span>
        <span class="badge">Answer Evaluation</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# TOP INFO CARDS
# ------------------------------------------------------------
info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.markdown("""
    <div class="mini-card">
        <h4>Personalized Questions</h4>
        <p>Generate interview questions tailored to the candidate’s role, skills, education, and job description.</p>
    </div>
    """, unsafe_allow_html=True)

with info_col2:
    st.markdown("""
    <div class="mini-card">
        <h4>Clear Answer Practice</h4>
        <p>Write your answers in a simple way and get support to make them clearer, stronger, and more professional.</p>
    </div>
    """, unsafe_allow_html=True)

with info_col3:
    st.markdown("""
    <div class="mini-card">
        <h4>Practical AI Feedback</h4>
        <p>Receive useful feedback on your strengths, weaknesses, and how to improve each answer step by step.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# SIDEBAR: API key + candidate profile
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## OpenAI Settings")
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Paste your OpenAI API key here."
    )

    st.markdown("## Candidate Profile")
    student_name = st.text_input("Student Name")
    degree = st.text_input("Degree")
    target_role = st.text_input("Target Role")

    col_lang_1, col_lang_2 = st.columns(2)
    with col_lang_1:
        english_level = st.text_input("English Level")
    with col_lang_2:
        italian_level = st.text_input("Italian Level")

    skills = st.text_area("Skills", height=120)
    job_description = st.text_area("Job Description", height=220)

    st.markdown("---")
    show_rag_context = st.checkbox("Show retrieved RAG context")

    st.markdown("""
    <div class="soft-note">
        Fill in the candidate profile, generate 3 interview questions, then write your answers and get AI feedback.
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# MAIN LAYOUT
# ------------------------------------------------------------
left_col, right_col = st.columns([1.45, 1], gap="large")

# ------------------------------------------------------------
# LEFT: QUESTIONS + ANSWERS + EVALUATION
# ------------------------------------------------------------
with left_col:
    render_section_header(
        "Step 1",
        "Generate Interview Questions",
        "Create three realistic interview questions tailored to the candidate profile and job description."
    )

    generate_col1, generate_col2 = st.columns([1, 1.2])
    with generate_col1:
        if st.button("Generate Questions", use_container_width=True):
            if not openai_api_key.strip():
                st.error("Please enter your OpenAI API key in the sidebar.")
            elif not target_role.strip():
                st.error("Please enter the target role.")
            elif not job_description.strip():
                st.error("Please enter the job description.")
            else:
                try:
                    client = configure_openai(openai_api_key)

                    questions, question_rag_context = generate_interview_questions(
                        client=client,
                        target_role=target_role,
                        skills=skills,
                        job_description=job_description,
                        degree=degree,
                        english_level=english_level,
                        italian_level=italian_level
                    )

                    if len(questions) < 3:
                        st.warning(
                            "The model returned fewer than 3 clean questions. "
                            "You can try again or improve the job description."
                        )

                    st.session_state.questions = questions
                    st.session_state.generated = True
                    st.session_state.evaluated = False
                    st.session_state.evaluation_text = ""
                    st.session_state.question_rag_context = question_rag_context
                    st.session_state.evaluation_rag_context = ""

                except Exception as e:
                    st.error(f"Error while generating questions: {e}")

    with generate_col2:
        st.info("Tip: Better job descriptions and clearer skills usually produce stronger questions.")

    # ------------------------------------------------------------
    # Show generated questions + collect answers
    # ------------------------------------------------------------
    if st.session_state.generated and st.session_state.questions:
        st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
        render_section_header(
            "Step 2",
            "Generated Questions",
            "Review the generated questions and answer each one as clearly and specifically as possible."
        )

        for i, q in enumerate(st.session_state.questions, start=1):
            st.markdown(
                f"""
                <div class="question-card">
                    <div class="question-label">Question {i}</div>
                    <div class="question-text">{q}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        if show_rag_context and st.session_state.question_rag_context:
            with st.expander("Retrieved context used for question generation"):
                st.text(st.session_state.question_rag_context)

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        render_section_header(
            "Step 3",
            "Write Your Answers",
            "Draft your responses below. Aim for clear structure, relevant examples, and concise communication."
        )

        answer_1 = st.text_area("Answer 1", key="answer_1", height=170, placeholder="Write your answer to Question 1 here...")
        answer_2 = st.text_area("Answer 2", key="answer_2", height=170, placeholder="Write your answer to Question 2 here...")
        answer_3 = st.text_area("Answer 3", key="answer_3", height=170, placeholder="Write your answer to Question 3 here...")

        eval_col1, eval_col2 = st.columns([1, 1.3])
        with eval_col1:
            if st.button("Evaluate Answers", use_container_width=True):
                if not openai_api_key.strip():
                    st.error("Please enter your OpenAI API key in the sidebar.")
                elif not answer_1.strip() or not answer_2.strip() or not answer_3.strip():
                    st.error("Please write all 3 answers before evaluation.")
                else:
                    try:
                        client = configure_openai(openai_api_key)

                        evaluation, evaluation_rag_context = evaluate_answers_with_openai(
                            client=client,
                            student_name=student_name,
                            degree=degree,
                            target_role=target_role,
                            skills=skills,
                            job_description=job_description,
                            questions=st.session_state.questions,
                            answers=[answer_1, answer_2, answer_3]
                        )

                        st.session_state.evaluation_text = evaluation
                        st.session_state.evaluated = True
                        st.session_state.evaluation_rag_context = evaluation_rag_context

                    except Exception as e:
                        st.error(f"Error while evaluating answers: {e}")

        with eval_col2:
            st.success("You will receive grounded feedback, strengths, weaknesses, and improved answer suggestions.")

    # ------------------------------------------------------------
    # Display evaluation
    # ------------------------------------------------------------
    if st.session_state.evaluated and st.session_state.evaluation_text:
        st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
        render_section_header(
            "Final Step",
            "AI Evaluation",
            "Review the structured feedback below and use it to improve your next interview performance."
        )

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Questions Evaluated", len(st.session_state.questions))
        metric_col2.metric("Answers Submitted", 3)
        metric_col3.metric("Evaluation Status", "Complete")

        st.markdown('<div class="evaluation-wrap">', unsafe_allow_html=True)
        st.markdown(st.session_state.evaluation_text)
        st.markdown('</div>', unsafe_allow_html=True)

        if show_rag_context and st.session_state.evaluation_rag_context:
            with st.expander("Retrieved context used for answer evaluation"):
                st.text(st.session_state.evaluation_rag_context)

# ------------------------------------------------------------
# RIGHT: QUICK GUIDE / STATUS / RESET
# ------------------------------------------------------------
with right_col:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">How to use this app</div>
        <div class="section-subtitle">
            Follow the workflow from candidate setup to AI evaluation. This panel helps keep the process simple and focused.
        </div>
    </div>
    """, unsafe_allow_html=True)

    guide1, guide2, guide3 = st.tabs(["Workflow", "Tips", "Status"])

    with guide1:
        st.markdown("""
        1. Enter your OpenAI API key in the sidebar.  
        2. Fill in the candidate profile.  
        3. Generate three interview questions.  
        4. Write all three answers.  
        5. Evaluate answers to receive AI feedback.  
        """)

    with guide2:
        st.markdown("""
        - Use a specific target role.  
        - Add relevant technical and soft skills.  
        - Paste a realistic job description.  
        - Write concrete answers with examples.  
        - Use STAR-style structure when possible.  
        """)

    with guide3:
        st.markdown(f"""
        **Questions generated:** {'Yes' if st.session_state.generated else 'No'}  
        **Answers evaluated:** {'Yes' if st.session_state.evaluated else 'No'}  
        **RAG debug enabled:** {'Yes' if show_rag_context else 'No'}  
        """)

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)


    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    if st.button("Reset App", use_container_width=True):
        st.session_state.questions = []
        st.session_state.generated = False
        st.session_state.evaluated = False
        st.session_state.evaluation_text = ""
        st.session_state.question_rag_context = ""
        st.session_state.evaluation_rag_context = ""
        st.rerun()