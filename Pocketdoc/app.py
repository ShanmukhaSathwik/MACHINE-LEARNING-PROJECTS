import streamlit as st
import os
import numpy as np
import json
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PocketDoc",
    page_icon="🏥",
    layout="centered"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Clean medical UI
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 780px; }

/* Header */
.pd-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid #e8edf2;
    margin-bottom: 2rem;
}
.pd-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #0f3460;
    margin: 0;
    letter-spacing: -0.5px;
}
.pd-header p {
    color: #64748b;
    font-size: 1rem;
    margin: 0.4rem 0 0;
    font-weight: 300;
}
.pd-tagline {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    margin-top: 0.8rem;
}
.pd-tagline span {
    font-size: 0.78rem;
    font-weight: 500;
    color: #0ea5e9;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Risk badges */
.risk-critical {
    background: #fff1f2;
    border-left: 4px solid #ef4444;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.4rem;
    margin: 1rem 0;
}
.risk-critical .risk-label { color: #ef4444; font-weight: 600; font-size: 1.1rem; }

.risk-moderate {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.4rem;
    margin: 1rem 0;
}
.risk-moderate .risk-label { color: #d97706; font-weight: 600; font-size: 1.1rem; }

.risk-mild {
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.4rem;
    margin: 1rem 0;
}
.risk-mild .risk-label { color: #16a34a; font-weight: 600; font-size: 1.1rem; }

.risk-desc { color: #475569; font-size: 0.9rem; margin-top: 0.2rem; }

/* Report sections */
.report-section {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.report-section h4 {
    color: #0f3460;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 0 0 0.6rem;
}
.report-section p {
    color: #374151;
    font-size: 0.93rem;
    line-height: 1.65;
    margin: 0;
}

/* Question card */
.question-card {
    background: #f0f7ff;
    border: 1px solid #bfdbfe;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1rem;
}
.question-card p {
    color: #1e40af;
    font-size: 0.95rem;
    margin: 0;
}

/* Risk update pill */
.risk-update {
    display: inline-block;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.82rem;
    color: #15803d;
    font-weight: 500;
    margin: 0.5rem 0;
}

/* Step indicator */
.step-badge {
    background: #0f3460;
    color: white;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    display: inline-block;
    margin-bottom: 0.5rem;
}

/* Disclaimer */
.disclaimer {
    background: #f8fafc;
    border: 1px dashed #cbd5e1;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    color: #94a3b8;
    font-size: 0.8rem;
    text-align: center;
    margin-top: 1.5rem;
}

/* Warning box */
.warning-box {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    color: #9a3412;
    font-size: 0.92rem;
}

/* Statemanager label */
.stTextInput label, .stTextArea label {
    font-weight: 500 !important;
    color: #374151 !important;
}

/* Button style */
.stButton > button {
    background: #0f3460 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 0.5rem 2rem !important;
    font-size: 0.95rem !important;
}
.stButton > button:hover {
    background: #1e40af !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="pd-header">
    <h1>🏥 PocketDoc</h1>
    <p>Your AI-powered second opinion, anytime.</p>
    <div class="pd-tagline">
        <span>Fast</span>
        <span>·</span>
        <span>Clear</span>
        <span>·</span>
        <span>Reliable</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD ENV + CLIENT
# ─────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found. Please add it to your .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────
# LOAD EMBEDDINGS + FAISS (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_resources():
    emb_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = np.load("pocketdoc_data/embeddings/embeddings.npy")
    with open("pocketdoc_data/embeddings/chunks.json", "r") as f:
        all_chunks = json.load(f)
    faiss_index = faiss.read_index("pocketdoc_data/embeddings/faiss_index.index")
    return emb_model, all_chunks, faiss_index

try:
    emb_model, all_chunks, faiss_index = load_resources()
except Exception as e:
    st.error(f"⚠️ Could not load embeddings: {e}\nMake sure pocketdoc_data/embeddings/ folder exists.")
    st.stop()

# ─────────────────────────────────────────────
# CORE FUNCTIONS (same logic as your notebook)
# ─────────────────────────────────────────────
def embed_query(text):
    return emb_model.encode([text])

def search_chunks(query):
    query_embedding = embed_query(query)
    distances, indices = faiss_index.search(query_embedding, 5)
    return [all_chunks[i] for i in indices[0]]

def classify_severity(symptoms):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content":
                f"Classify these symptoms as exactly one of: CRITICAL, MODERATE, MILD. Return only that one word, nothing else. Symptoms: {symptoms}"}]
        )
        return response.choices[0].message.content.strip()
    except:
        return "MODERATE"

def re_evaluate_severity(symptoms, answers):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content":
                f"Original symptoms: {symptoms}\nPatient answers: {answers}\nBased on ALL this information, classify severity as CRITICAL, MODERATE, or MILD.\nReturn only one word."}]
        )
        return response.choices[0].message.content.strip()
    except:
        return "MODERATE"

def get_next_question(symptoms, severity, answers):
    try:
        conversation = f"Symptoms: {symptoms}\nSeverity: {severity}\nAnswers so far: {answers}"
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content":
                f"You are a triage nurse.\n{conversation}\nAsk only about:\n- Pain characteristics\n- Associated symptoms\n- Duration and progression\n- Relevant medical history\nAsk ONE question or say ENOUGH_INFO if you have enough.\nReturn only the question or ENOUGH_INFO."}]
        )
        return response.choices[0].message.content.strip()
    except:
        return "ENOUGH_INFO"

def generate_final_report(symptoms, severity, answers, chunks):
    try:
        context = "\n\n".join(chunks)
        answers_text = "\n".join(answers)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": f"""You are a clinical triage assistant.

Medical References: {context}

Patient Symptoms: {symptoms}
Severity: {severity}
Follow up answers: {answers_text}

Generate a structured clinical report with these exact sections:

1. SYMPTOMS SUMMARY
2. RED FLAGS
3. TRIAGE LEVEL: {severity}
4. POSSIBLE CONDITIONS
5. RECOMMENDED TESTS
6. FOLLOW UP CARE
"""}]
        )
        return response.choices[0].message.content
    except:
        return None

def validate_medical(symptoms):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content":
                f"Is this a medical symptom or health complaint? Answer only YES or NO. Input: {symptoms}"}]
        )
        return "NO" not in response.choices[0].message.content.strip().upper()
    except:
        return True

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
defaults = {
    "stage": "input",           # input → questioning → report
    "symptoms": "",
    "severity": "",
    "answers": [],
    "question_count": 0,
    "current_question": "",
    "report": "",
    "risk_updates": [],
    "chunks": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def risk_badge(severity):
    badges = {
        "CRITICAL": ("risk-critical", "🔴 CRITICAL", "Seek emergency care immediately"),
        "MODERATE": ("risk-moderate", "🟡 MODERATE", "See a doctor within 24–48 hours"),
        "MILD":     ("risk-mild",     "🟢 MILD",     "Monitor at home, rest and hydrate"),
    }
    cls, label, desc = badges.get(severity, badges["MODERATE"])
    return f"""<div class="{cls}">
        <div class="risk-label">{label}</div>
        <div class="risk-desc">{desc}</div>
    </div>"""

def max_questions(severity):
    return {"CRITICAL": 4, "MODERATE": 3, "MILD": 2}.get(severity, 3)

def parse_report_sections(report_text):
    """Split the report into named sections for pretty rendering."""
    sections = {}
    current = None
    buffer = []
    for line in report_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        found = False
        for name in ["SYMPTOMS SUMMARY", "RED FLAGS", "TRIAGE LEVEL", "POSSIBLE CONDITIONS", "RECOMMENDED TESTS", "FOLLOW UP CARE"]:
            if name in line.upper():
                if current:
                    sections[current] = "\n".join(buffer).strip()
                current = name
                buffer = []
                found = True
                break
        if not found and current:
            buffer.append(line)
    if current:
        sections[current] = "\n".join(buffer).strip()
    return sections

def reset():
    for k, v in defaults.items():
        st.session_state[k] = v

# ─────────────────────────────────────────────
# STAGE 1 — SYMPTOM INPUT
# ─────────────────────────────────────────────
if st.session_state.stage == "input":
    st.markdown('<div class="step-badge">STEP 1 OF 3 — DESCRIBE YOUR SYMPTOMS</div>', unsafe_allow_html=True)
    symptoms_input = st.text_area(
        "What are you experiencing?",
        placeholder="e.g. I have chest pain and fever for the past 4 days...",
        height=120
    )

    if st.button("Analyze Symptoms →"):
        if not symptoms_input.strip():
            st.warning("Please describe your symptoms first.")
        else:
            with st.spinner("Validating input..."):
                is_medical = validate_medical(symptoms_input)

            if not is_medical:
                st.markdown("""<div class="warning-box">
                    ⚠️ <strong>PocketDoc is designed for medical symptoms only.</strong><br>
                    Please describe a health concern or physical symptom.
                </div>""", unsafe_allow_html=True)
            else:
                with st.spinner("Analyzing symptoms..."):
                    severity = classify_severity(symptoms_input)
                    question = get_next_question(symptoms_input, severity, [])

                st.session_state.symptoms = symptoms_input
                st.session_state.severity = severity
                st.session_state.current_question = question
                st.session_state.stage = "questioning"
                st.rerun()

# ─────────────────────────────────────────────
# STAGE 2 — ADAPTIVE QUESTIONS
# ─────────────────────────────────────────────
elif st.session_state.stage == "questioning":
    st.markdown('<div class="step-badge">STEP 2 OF 3 — FOLLOW-UP QUESTIONS</div>', unsafe_allow_html=True)

    # Show symptoms summary
    st.markdown(f"**Symptoms:** {st.session_state.symptoms}")

    # Show current risk
    st.markdown(risk_badge(st.session_state.severity), unsafe_allow_html=True)

    # Show risk update history
    for update in st.session_state.risk_updates:
        st.markdown(f'<span class="risk-update">↻ Risk updated: {update}</span>', unsafe_allow_html=True)

    # Show previous Q&A
    if st.session_state.answers:
        with st.expander(f"Previous answers ({len(st.session_state.answers)})"):
            for qa in st.session_state.answers:
                st.markdown(f"• {qa}")

    # Current question
    q = st.session_state.current_question
    mq = max_questions(st.session_state.severity)
    qnum = st.session_state.question_count + 1

    if "ENOUGH_INFO" not in q and st.session_state.question_count < mq:
        st.markdown(f'<div class="question-card"><p>🏥 <strong>Question {qnum}:</strong> {q}</p></div>',
                    unsafe_allow_html=True)

        answer = st.text_input("Your answer:", key=f"answer_{qnum}")

        if st.button("Submit Answer →"):
            if not answer.strip():
                st.warning("Please type your answer.")
            else:
                st.session_state.answers.append(f"Q: {q}  A: {answer}")
                st.session_state.question_count += 1

                # Re-evaluate severity
                with st.spinner("Updating risk assessment..."):
                    new_severity = re_evaluate_severity(
                        st.session_state.symptoms,
                        st.session_state.answers
                    )

                if new_severity != st.session_state.severity:
                    st.session_state.risk_updates.append(
                        f"{st.session_state.severity} → {new_severity}"
                    )
                    st.session_state.severity = new_severity

                # Get next question
                with st.spinner("Preparing next question..."):
                    next_q = get_next_question(
                        st.session_state.symptoms,
                        st.session_state.severity,
                        st.session_state.answers
                    )
                st.session_state.current_question = next_q
                st.rerun()
    else:
        # Done with questions — generate report
        with st.spinner("Searching medical database..."):
            chunks = search_chunks(st.session_state.symptoms)
            st.session_state.chunks = chunks

        with st.spinner("Generating your clinical report..."):
            report = generate_final_report(
                st.session_state.symptoms,
                st.session_state.severity,
                st.session_state.answers,
                chunks
            )
            st.session_state.report = report
            st.session_state.stage = "report"
            st.rerun()

# ─────────────────────────────────────────────
# STAGE 3 — REPORT
# ─────────────────────────────────────────────
elif st.session_state.stage == "report":
    st.markdown('<div class="step-badge">STEP 3 OF 3 — YOUR POCKETDOC REPORT</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Risk banner
    st.markdown(risk_badge(st.session_state.severity), unsafe_allow_html=True)

    # Parse and render report sections
    sections = parse_report_sections(st.session_state.report)

    section_labels = {
        "SYMPTOMS SUMMARY":    "📋 Symptoms Summary",
        "RED FLAGS":           "🚩 Red Flags",
        "TRIAGE LEVEL":        "⚕️ Triage Level",
        "POSSIBLE CONDITIONS": "🔍 Possible Conditions",
        "RECOMMENDED TESTS":   "🧪 Recommended Tests",
        "FOLLOW UP CARE":      "📅 Follow-Up Care",
    }

    for key, label in section_labels.items():
        content = sections.get(key, "")
        if content:
            st.markdown(f"""<div class="report-section">
                <h4>{label}</h4>
                <p>{content.replace(chr(10), '<br>')}</p>
            </div>""", unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""<div class="disclaimer">
        ⚕️ This is not a substitute for professional medical advice.
        Always consult a qualified healthcare provider for diagnosis and treatment.
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Start New Assessment"):
        reset()
        st.rerun()
