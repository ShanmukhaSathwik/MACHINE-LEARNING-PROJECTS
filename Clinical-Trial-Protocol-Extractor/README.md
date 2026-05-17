# 🏥 Clinical Trial Protocol Extractor

> **Project Type:** NLP / Document Intelligence  
> **Domain:** Healthcare / Clinical Research  
> **Tools:** Python, HuggingFace Transformers, Facebook BART, DistilBERT, PyMuPDF

---

## 📌 Project Overview

This project automatically extracts and summarizes critical sections from large clinical trial protocol PDFs using a combination of **Regex pattern matching**, **Question-Answering (DistilBERT)**, and **Abstractive Summarization (Facebook BART-large-CNN)**.

Clinical trial protocols are often 100+ page documents. This tool extracts only what matters — saving hours of manual reading for researchers and medical professionals.

---

## 🔍 Sections Extracted

| Section | Description |
|---|---|
| **Inclusion Criteria** | Who is eligible to participate |
| **Exclusion Criteria** | Who cannot participate |
| **Vaccine Delay Recommendations** | When to postpone vaccination |
| **Primary Endpoints** | Main success measures of the trial |
| **Secondary Endpoints** | Additional metrics tracked |
| **Treatment Arms** | Vaccine vs placebo groups |
| **Dosing Schedule** | When and how doses are administered |

---

## 🧠 Tech Stack & Approach

### Pipeline:
Input PDF → Text Extraction (PyMuPDF)
         → Section Detection (Regex patterns)
         → QA Fallback (DistilBERT)
         → Summarization (BART-large-CNN)
         → Structured Output PDF (ReportLab)

### Models Used:
| Model | Purpose |
|---|---|
| facebook/bart-large-cnn | Abstractive summarization of long sections |
| distilbert-base-cased-distilled-squad | QA-based fallback extraction |
| PyMuPDF (fitz) | PDF text extraction |

---

## 📄 Sample Output

- **Inclusion:** Adults 18+, negative pregnancy test, valid contraception
- **Exclusion:** Prior COVID-19 vaccination, HIV/HBV/HCV history, immunosuppressive therapy
- **Treatment Arms:** CVnCoV 12µg mRNA vaccine vs saline placebo (36,500 participants)
- **Dosing:** 2 intramuscular injections, Day 1 and Day 29, stored at -80°C
- **Primary Endpoint:** Vaccine efficacy against virologically confirmed COVID-19, 14 days post dose 2

---

## 🚀 How to Run

1. Open Clinical_trail_Protocal.ipynb in Google Colab
2. Upload your clinical trial PDF
3. Update the PDF_PATH variable in the configuration section
4. Run all cells — structured summary PDF is generated automatically

---

## 💡 Key Features

- ✅ Handles 100+ page PDFs automatically
- ✅ Regex + AI dual extraction for maximum accuracy
- ✅ BART summarization condenses dense medical text
- ✅ Outputs a clean, readable structured PDF
- ✅ Fallback QA model ensures no section is missed
