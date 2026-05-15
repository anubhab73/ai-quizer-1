# app.py
import streamlit as st
from agents.pdf_agent import process_pdf
from agents.question_agent import generate_questions
from agents.evaluation_agent import evaluate_answers
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()

# --- Constants (directly here) ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
PERSIST_DIR = "./vector_db"

# --- UI Styling ---
st.markdown("""
<style>
.main .block-container {padding-top: 2rem;}
.stButton > button {background-color:#8B5CF6;color:white;border:none;border-radius:8px;padding:0.5rem 1rem;font-weight:500;}
.stButton > button:hover {background-color:#7C3AED;color:white;}
.step-circle {background:#A78BFA;color:white;border-radius:50%;width:40px;height:40px;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:16px;z-index:2;}
.step-circle-active {background:#8B5CF6;box-shadow:0 0 0 3px #C4B5FD;}
.step-circle-completed {background:#10B981;}
.question-box {background:#F3F4F6;border-left:4px solid #8B5CF6;padding:1rem;margin:0.5rem 0;border-radius:4px;}
.study-plan-box {background:#FEF3C7;border:2px solid #F59E0B;border-radius:8px;padding:1.5rem;margin:1rem 0;}
</style>
""", unsafe_allow_html=True)

st.title("AI Quiz Taker System")
st.markdown("### Intelligent Document-Based Quiz Generator & Evaluator")
st.markdown("*Powered by Groq API & HuggingFace*")

# --- Session State ---
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'topic' not in st.session_state:
    st.session_state.topic = ""
if 'questions' not in st.session_state:
    st.session_state.questions = None
if 'student_answers' not in st.session_state:
    st.session_state.student_answers = {}
if 'mcq_index' not in st.session_state:
    st.session_state.mcq_index = {}
if 'results' not in st.session_state:
    st.session_state.results = None
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = ""
if 'question_config' not in st.session_state:
    st.session_state.question_config = {
        'mcq': {'enabled': True, 'count': 3},
        'short': {'enabled': True, 'count': 2},
        'long': {'enabled': True, 'count': 1}
    }

# --- Progress Steps ---
steps = ["Upload PDF", "Select Topic", "Configure Questions", "Get Questions", "Enter Answers", "Get Feedback"]
cols = st.columns(6)
for i, step in enumerate(steps):
    with cols[i]:
        circle_class = "step-circle"
        if st.session_state.step == i + 1:
            circle_class += " step-circle-active"
        elif st.session_state.step > i + 1:
            circle_class += " step-circle-completed"
        st.markdown(f'<div class="{circle_class}">{i+1}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="text-align:center;margin-top:.5rem;font-size:12px;color:#6B7280;">{step}</div>', unsafe_allow_html=True)
st.divider()

# --- Clean MCQ options ---
def clean_options(opts):
    if not opts:
        return []
    return [o.strip() for o in opts if isinstance(o, str) and o.strip()]

# ========== STEP 1: UPLOAD PDF ==========
if st.session_state.step == 1:
    st.subheader("Upload Your Document")
    pdf_upload = st.file_uploader("Choose PDF", type="pdf")

    if pdf_upload:
        st.session_state.pdf_name = pdf_upload.name.replace(".pdf", "")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_upload.read())
            pdf_path = tmp.name

        with st.spinner("Processing PDF..."):
            st.session_state.vectorstore = process_pdf(pdf_path, st.session_state.pdf_name)
        os.unlink(pdf_path)

        if st.session_state.vectorstore:
            st.success("PDF processed!")

    c1, c2 = st.columns(2)
    with c1:
        st.button("Back", key="step1_back")
    with c2:
        if st.session_state.vectorstore and st.button("Next Step", key="step1_next"):
            st.session_state.step = 2
            st.rerun()

# ========== STEP 2: TOPIC ==========
elif st.session_state.step == 2:
    st.subheader("Choose Topic")
    topic = st.text_input("Topic", value=st.session_state.topic)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", key="step2_back"):
            st.session_state.step = 1
            st.rerun()
    with c2:
        if topic.strip() and st.button("Next Step", key="step2_next"):
            st.session_state.topic = topic.strip()
            st.session_state.step = 3
            st.rerun()

# ========== STEP 3: QUESTION CONFIG ==========
elif st.session_state.step == 3:
    st.subheader("Configure Question Types")
    cfg = st.session_state.question_config

    # MCQ
    c1, c2 = st.columns([3,1])
    with c1:
        mcq_enabled = st.checkbox("Multiple Choice", value=cfg['mcq']['enabled'])
    with c2:
        mcq_count = st.number_input("Count", 1, 10, cfg['mcq']['count']) if mcq_enabled else 0

    # Short
    c1, c2 = st.columns([3,1])
    with c1:
        short_enabled = st.checkbox("Short Questions", value=cfg['short']['enabled'])
    with c2:
        short_count = st.number_input("Count", 1, 10, cfg['short']['count']) if short_enabled else 0

    # Long
    c1, c2 = st.columns([3,1])
    with c1:
        long_enabled = st.checkbox("Long Questions", value=cfg['long']['enabled'])
    with c2:
        long_count = st.number_input("Count", 1, 5, cfg['long']['count']) if long_enabled else 0

    st.session_state.question_config = {
        "mcq": {"enabled": mcq_enabled, "count": mcq_count},
        "short": {"enabled": short_enabled, "count": short_count},
        "long": {"enabled": long_enabled, "count": long_count},
    }

    total = mcq_count + short_count + long_count
    st.info(f"Total Questions: {total}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", key="step3_back"):
            st.session_state.step = 2
            st.rerun()
    with c2:
        if total > 0 and st.button("Next Step", key="step3_next"):
            st.session_state.step = 4
            st.rerun()

# ========== STEP 4: GENERATE QUESTIONS ==========
elif st.session_state.step == 4:
    st.subheader("Generate Questions")
    cfg = st.session_state.question_config
    total = sum(cfg[t]['count'] if cfg[t]['enabled'] else 0 for t in ('mcq', 'short', 'long'))
    st.info(f"**Topic:** {st.session_state.topic}\n\n**Total Questions:** {total}")

    if st.button("Generate Questions", type="primary"):
        with st.spinner("Generating..."):
            st.session_state.questions = generate_questions(
                st.session_state.topic,
                st.session_state.vectorstore,
                st.session_state.question_config
            )
        if st.session_state.questions and 'error' not in st.session_state.questions:
            st.success("Questions generated!")
            qdata = st.session_state.questions['questions']
            question_number = 1

            # MCQs plain
            if 'mcqs' in qdata:
                st.markdown("#### Multiple Choice Questions")
                for mcq in qdata['mcqs']:
                    st.markdown(f"**Q{question_number}. {mcq.get('question', 'N/A')}**")
                    options = mcq.get('options', [])
                    for j, option in enumerate(options):
                        st.markdown(f"   {chr(65+j)}. {option}")
                    question_number += 1

            # Shorts plain
            if 'shorts' in qdata:
                st.markdown("#### Short Answer Questions")
                for short in qdata['shorts']:
                    st.markdown(f"**Q{question_number}. {short.get('question', 'N/A')}**")
                    question_number += 1

            # Longs plain
            if 'longs' in qdata:
                st.markdown("#### Long Answer Questions")
                for long_q in qdata['longs']:
                    st.markdown(f"**Q{question_number}. {long_q.get('question', 'N/A')}**")
                    question_number += 1
        else:
            st.error(st.session_state.questions.get('error', 'Generation failed'))

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", key="step4_back"):
            st.session_state.step = 3
            st.rerun()
    with c2:
        if st.session_state.questions and 'error' not in st.session_state.questions:
            if st.button("Next Step", key="step4_next"):
                st.session_state.step = 5
                st.rerun()


# ========== STEP 5: ENTER ANSWERS ==========
elif st.session_state.step == 5:
    st.subheader("Enter Your Answers")

    qdata = st.session_state.questions["questions"]
    qno = 1

    # MCQs
    if qdata.get("mcqs"):
        st.markdown("### MCQs")
        for i, mcq in enumerate(qdata["mcqs"]):
            key = f"mcq_{i}"
            st.write(f"**Q{qno}.** {mcq['question']}")
            opts = clean_options(mcq.get("options", []))

            if opts:
                options = [f"{chr(65+j)}. {opt}" for j,opt in enumerate(opts)]
                selected = st.radio("Choose:", options, key=key)
                st.session_state.student_answers[key] = selected[0]
            else:
                st.warning("Options missing — enter manually.")
                ans = st.text_input("Answer (A/B/C/D):", key=key+"_t")
                st.session_state.student_answers[key] = ans.upper().strip()

            qno += 1

    # SHORT
    if qdata.get("shorts"):
        st.markdown("### Short Questions")
        for i, sq in enumerate(qdata["shorts"]):
            key = f"short_{i}"
            st.write(f"**Q{qno}.** {sq['question']}")
            ans = st.text_area("Your Answer:", key=key)
            st.session_state.student_answers[key] = ans.strip()
            qno += 1

    # LONG
    if qdata.get("longs"):
        st.markdown("### Long Questions")
        for i, lq in enumerate(qdata["longs"]):
            key = f"long_{i}"
            st.write(f"**Q{qno}.** {lq['question']}")
            ans = st.text_area("Your Answer:", key=key)
            st.session_state.student_answers[key] = ans.strip()
            qno += 1

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", key="step5_back"):
            st.session_state.step = 4
            st.rerun()
    with c2:
        if st.button("Submit Answers", type="primary"):
            ordered = []
            for prefix in ["mcq", "short", "long"]:
                idx = 0
                while f"{prefix}_{idx}" in st.session_state.student_answers:
                    ordered.append(st.session_state.student_answers[f"{prefix}_{idx}"])
                    idx += 1

            st.session_state.results = evaluate_answers(st.session_state.questions, {"individual_answers": ordered})
            st.session_state.step = 6
            st.rerun()

# ========== STEP 6: VIEW RESULTS ==========
elif st.session_state.step == 6:
    st.subheader("Results")

    res = st.session_state.results
    st.write(f"**Average Score:** {res.get('avg_score',0)}/10")

    for r in res.get("results", []):
        st.markdown(f"### Q{r['question_number']} ({r['question_type']})")
        st.write(r["question"])
        st.write("**Your Answer:**", r["student_answer"])
        st.write("**Model Answer:**", r["model_answer"])
        st.write("**Score:**", r["evaluation"]["score"])
        st.write("**Feedback:**", r["evaluation"]["feedback"])

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start Over"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.session_state.step = 1
            st.rerun()
    with c2:
        if st.button("New Quiz"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
