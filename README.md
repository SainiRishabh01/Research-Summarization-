# 🎓 Research Paper Summarization Multi-Agent System (MAS)

An AI-powered **multi-agent Streamlit application** that takes a research paper (via upload, DOI, or URL) and performs **end-to-end semantic analysis**, including:

- Text & image extraction
- Image captioning using multimodal LLMs (Pixtral-12B)
- Topic classification
- Summary generation using Mistral
- Cross-topic synthesis
- Audio narration using gTTS

---

## 📌 Use Case

Ideal for:
- Researchers quickly digesting long papers
- Students preparing study notes
- Professionals extracting insights across multiple topics
- Visually impaired users needing audio versions of papers

---

## 🔍 Key Features by Agent

| Agent No. | Name                  | Function                                                                 |
|-----------|-----------------------|--------------------------------------------------------------------------|
| 1         | **Search Agent**       | Accepts file uploads, DOI (e.g., `10.48550/arXiv.2307.08594`), or paper URLs |
| 2         | **Extraction Agent**   | Extracts text and images from PDF, DOCX, or TXT                           |
| 3         | **Vision Agent**       | Uses Pixtral-12B to caption images extracted from the paper               |
| 4         | **Classifier Agent**   | Matches text chunks to user-defined topics using keyword matching         |
| 5         | **Summarizer Agent**   | Generates concise summaries using Mistral's chat model                    |
| 6         | **Synthesis Agent**    | Summarizes chunks under each topic label                                  |
| 7         | **Audio Agent**        | Narrates text output to `.mp3` using gTTS                                 |

---

## 🧪 Example Inputs

### 🔗 DOI-Based Input

Try:

- `10.48550/arXiv.2307.08594` *(LLaMA 2)*
- `10.48550/arXiv.1706.03762` *(Transformer: Attention is All You Need)*

### 📂 File Upload

Upload `.pdf`, `.docx`, or `.txt` files directly via UI.

### 🌐 URL Input

Paste the full URL to a research paper (ensure it returns a `.pdf`).

---

## 📁 Outputs You’ll See

- 🔹 **Complete Document Summary**
- 🔸 **Topic-Based Summaries** *(optional, based on your topic inputs)*
- 🖼️ **Extracted Images with AI-generated Captions**
- 🔊 **Audio Narration** of text summaries

---

## 🛠️ Local Setup Guide

### 1. Clone the Repository

```bash
git clone https://github.com/SainiRishabh01/Research-Summarization-.git
cd Research-Summarization-
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

