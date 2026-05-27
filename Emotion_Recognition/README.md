# 🎭 Multimodal Emotion Recognition — MELD Dataset

> Recognizing emotions in conversation using audio + text with DistilBERT and Attention-Based Fusion.

This project reproduces and improves a multimodal emotion recognition baseline on the **MELD dataset** (13,708 utterances from the TV series *Friends*). Two targeted modifications are proposed: replacing GloVe embeddings with **DistilBERT** and replacing concatenation fusion with a **multi-head self-attention** mechanism.

---

## 🏆 Results

| Model | Weighted F1 | Macro F1 | Accuracy |
|---|---|---|---|
| Baseline (concat fusion) | 0.5485 | 0.3939 | 0.5211 |
| **Modified (attention fusion)** | **0.5523** | **0.3955** | **0.5218** |
| Delta | +0.0039 | +0.0016 | +0.0008 |

The modified model outperforms the baseline on **5 out of 7** emotion classes, with the biggest gains on joy (+0.0221) and sadness (+0.0169).

---

## 🏗️ Architecture

```
Audio (MFCC) ──► LSTM Encoder (128) ──┐
                                       ├──► Multi-Head Attention Fusion ──► Dense ──► 7 Emotions
Text (DistilBERT) ──► Dense (128) ────┘
```

**Modification 1 — DistilBERT instead of GloVe:**
Static GloVe embeddings are replaced with DistilBERT contextual embeddings (768-dim → projected to 128-dim). This gives the model richer, context-aware text representations.

**Modification 2 — Attention Fusion instead of Concatenation:**
Instead of simply concatenating audio and text vectors, a 4-head self-attention layer dynamically weights the two modalities per utterance.

---

## 📊 Dataset

**MELD** — Multimodal EmotionLines Dataset (Poria et al., 2019)
- 1,433 dialogues, 13,708 utterances from *Friends*
- 7 emotion classes: neutral, joy, surprise, fear, sadness, disgust, anger
- Train / Dev / Test: 9,989 / 1,109 / 2,610 utterances
- Highly imbalanced — neutral = 47.2%, fear = 2.7%

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Text Encoder | DistilBERT (`distilbert-base-uncased`) |
| Audio Features | MFCC (40 coefficients, librosa) |
| Sequence Model | LSTM (128 units) |
| Fusion | Multi-Head Self-Attention (4 heads) |
| Framework | TensorFlow / Keras |
| NLP Library | HuggingFace Transformers 4.41+ |
| Runtime | Google Colab (NVIDIA T4 GPU) |

---

## ⚙️ Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Open in Google Colab
Upload `EmotionRecognition.ipynb` and run cells in order.

### 3. Dataset
The notebook automatically downloads the MELD CSVs from the official GitHub repo. Audio splits are downloaded from HuggingFace (~2.5 GB).

### 4. Run order
- **Section 1** — Install libraries & imports
- **Section 2** — Load MELD dataset
- **Section 3–4** — Feature extraction (MFCC + DistilBERT)
- **Section 5–6** — Build & train baseline model
- **Section 7** — Build & train attention model
- **Section 8** — Results comparison & visualisation

---

## 📁 Project Structure

```
Emotion_Recognition/
├── EmotionRecognition.ipynb    # Main notebook
├── requirements.txt            # Python dependencies
├── .gitignore                  # Excludes data and model weights
└── README.md                   # You are here
```

---

## ⚠️ Notes

- Requires a **GPU runtime** in Google Colab (Runtime → Change runtime type → T4 GPU)
- Full training takes ~30–40 minutes on T4
- Only 43.1% of training audio is available in MELD — the text branch (DistilBERT) compensates strongly
- DistilBERT weights are **frozen** (used as feature extractor only)

---

## 📄 Reference

Poria, S., Hazarika, D., Majumder, N., Naik, G., Cambria, E., & Mihalcea, R. (2019). MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations. *ACL 2019*.
