# 🩺 PocketDoc — AI-Powered Medical Triage Assistant

> **Your AI-powered second opinion, anytime. Fast. Clear. Reliable.**

PocketDoc is an intelligent medical triage system that listens to your symptoms, asks smart follow-up questions, and generates a structured clinical report — all powered by LLaMA 3.3 via Groq and a RAG pipeline built on real medical Q&A data.

---

## 🚀 Features

- **Symptom Analysis** — Describes symptoms in natural language; PocketDoc understands them
- **Adaptive Triage** — Classifies severity as CRITICAL, MODERATE, or MILD
- **Smart Follow-up Questions** — Asks 2–4 targeted questions based on severity level
- **Dynamic Risk Re-evaluation** — Updates severity in real-time as answers come in
- **RAG Pipeline** — Retrieves relevant medical context from MedQuAD (54,000+ Q&A pairs)
- **Structured Clinical Report** — Outputs symptoms summary, red flags, possible conditions, recommended tests, and follow-up care
- **Input Validation** — Filters out non-medical inputs automatically

---

## 🏗️ Architecture

```
User Symptoms
     │
     ▼
Input Validation (LLaMA 3.3)
     │
     ▼
Severity Classification (CRITICAL / MODERATE / MILD)
     │
     ▼
Adaptive Follow-up Questions (2–4 based on severity)
     │
     ▼
RAG Retrieval — FAISS + SentenceTransformers (MedQuAD)
     │
     ▼
Structured Clinical Report (LLaMA 3.3 via Groq)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | LLaMA 3.3 70B (via Groq) |
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Vector Store | FAISS |
| Dataset | MedQuAD (HuggingFace `lavita/MedQuAD`) |
| NLP | spaCy `en_core_web_sm` |
| Runtime | Google Colab |

---

## ⚙️ Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/pocketdoc.git
```

### 2. Create a `.env` file
```bash
GROQ_API_KEY=your_groq_api_key_here
```
Get your free API key at [console.groq.com](https://console.groq.com)

### 3. Open in Google Colab
Upload `PocketDoc.ipynb` to Colab, mount your Google Drive, and run the cells in order.

### 4. Run PocketDoc
```python
pocketdoc()
```

---

## 📊 Evaluation

Three built-in tests are included (Cell 9):

| Test | What it checks |
|---|---|
| Severity Consistency | LLM returns same severity across 3 runs |
| Retrieval Relevance | FAISS retrieves medically relevant chunks |
| Response Time | Full pipeline end-to-end latency |

---

## ⚠️ Disclaimer

PocketDoc is **not a substitute for professional medical advice**. It is a learning project and should not be used for real clinical decisions.

---

## 📁 Project Structure

```
pocketdoc/
├── PocketDoc.ipynb     # Main notebook
├── .env.example        # API key template
├── requirements.txt    # Python dependencies
├── .gitignore          # Excludes secrets and data files
└── README.md           # You are here
```

---

## 🙋 Author

Built with ❤️ as an AI/ML learning project.  
Feel free to fork, star ⭐, or raise an issue!
