# 🏥 PocketDoc — AI-Powered Medical Triage Assistant

> Your AI-powered second opinion, anytime. Fast. Clear. Reliable.

PocketDoc is a medical AI that doesn't just answer — it thinks, asks, and diagnoses. Built as part of SSAI's 48-hour skills assessment.

---

## 🔍 What it does

1. You describe your symptoms
2. It asks adaptive follow-up questions, the way a doctor actually would
3. It generates a structured clinical report you can understand

---

## 🧠 How it works

```
User Input → Severity Classification → Adaptive Questioning
→ RAG Retrieval (FAISS) → Clinical Report Generation
```

- **Severity Classification** — LLaMA 3.3 classifies symptoms as CRITICAL / MODERATE / MILD
- **Adaptive Questions** — dynamically generated based on symptoms and previous answers
- **RAG Pipeline** — FAISS vector search over MedQuAD medical dataset
- **Report Generation** — structured 6-section clinical report

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | LLaMA 3.3 70B via Groq |
| Vector Search | FAISS |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| NLP | spaCy |
| Dataset | MedQuAD (HuggingFace) |
| UI | Streamlit |
| Language | Python |

---

## 🚀 Running the Streamlit App

### Step 1 — Clone the repo
```bash
git clone https://github.com/ShanmukhaSathwik/MACHINE-LEARNING-PROJECTS.git
cd MACHINE-LEARNING-PROJECTS/Pocketdoc
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Set up your API key
```bash
cp .env.example .env
```
Open `.env` and replace `your_groq_api_key_here` with your actual key from [console.groq.com](https://console.groq.com)

### Step 4 — Generate embeddings
Run all cells in `PocketDoc.ipynb` first to generate the `pocketdoc_data/` folder with embeddings locally.

### Step 5 — Run the app
```bash
python -m streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 📊 Evaluation Results

| Test | Result |
|---|---|
| Severity Consistency | Consistent across 3 runs |
| Retrieval Relevance | 4/5 chunks relevant |
| Pipeline Response Time | ~3–5 seconds |

---

## 🔮 What's Next

- **Agentic RAG** — agents that plan and retrieve across multiple steps dynamically
- **Multi-modal input** — describe symptoms by voice or upload images like rashes / X-rays
- **Fine-tuned medical LLM** — domain-specific reasoning instead of a general model

---

## ⚠️ Disclaimer

PocketDoc is not a substitute for professional medical advice. Always consult a qualified healthcare provider for diagnosis and treatment.

---

## 👤 Author

**ShanmukhaSathwik** — [GitHub](https://github.com/ShanmukhaSathwik/MACHINE-LEARNING-PROJECTS)
