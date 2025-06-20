# Research Paper Summarization Multi-Agent System (MAS)
# Adapted from a RAG-based MongoDB-powered chatbot to a DB-less MAS pipeline

import os
import io
import base64
import hashlib
import streamlit as st
from PIL import Image
from docx import Document
from dotenv import load_dotenv
import pymupdf
import re
import requests
from mistralai import Mistral, UserMessage, SystemMessage
from sentence_transformers import SentenceTransformer
from gtts import gTTS

# ENV
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
mistral_client = Mistral(api_key=MISTRAL_API_KEY)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# STATE
if "agents_results" not in st.session_state:
    st.session_state.agents_results = {}

# AGENT UTILS

def hash_image_bytes(img_bytes):
    return hashlib.md5(img_bytes).hexdigest()

# Agent 1: Search & Discovery (DOI, URL)
def discover_paper_from_doi(doi):
    url = f"https://doi.org/{doi}"
    try:
        response = requests.get(url, headers={"Accept": "application/pdf"})
        return response.content if response.ok else None
    except:
        return None

def discover_paper_from_url(url):
    try:
        response = requests.get(url)
        return response.content if response.ok else None
    except:
        return None

# Agent 2: File Processing

def extract_text_and_images(file_bytes, file_type):
    text_chunks, image_info = [], []
    if file_type == "application/pdf":
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        for i, page in enumerate(doc):
            txt = page.get_text("text").strip()
            if txt:
                text_chunks.append((f"Page {i+1}", txt))
            for j, img in enumerate(page.get_images(full=True)):
                try:
                    image = doc.extract_image(img[0])
                    image_info.append((f"Page {i+1}-Image {j+1}", image["image"]))
                except:
                    continue
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(io.BytesIO(file_bytes))
        for i, para in enumerate(doc.paragraphs):
            if para.text.strip():
                text_chunks.append((f"Paragraph {i+1}", para.text.strip()))
        for rel in doc.part._rels.values():
            if "image" in rel.target_ref:
                try:
                    image_info.append(("DOCX Image", rel.target_part.blob))
                except:
                    continue
    elif file_type == "text/plain":
        lines = file_bytes.decode("utf-8", errors="ignore").splitlines()
        for j in range(0, len(lines), 40):
            chunk = "\n".join(lines[j:j+40]).strip()
            if chunk:
                text_chunks.append((f"Lines {j+1}-{j+40}", chunk))
    return text_chunks, image_info

# Agent 3: Image Captioning

def caption_image(img_bytes):
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        msg = [UserMessage(content=[
            {"type": "text", "text": "Caption this image in detail."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        ])]
        res = mistral_client.chat.complete(model="pixtral-12b-2409", messages=msg)
        return res.choices[0].message.content
    except Exception as e:
        return f"❌ Caption error: {e}"

# Agent 4: Topic Classification

def classify_topic(text_chunks, topic_list):
    matched = {topic: [] for topic in topic_list}
    for label, text in text_chunks:
        for topic in topic_list:
            if re.search(rf"\b{re.escape(topic.lower())}\b", text.lower()):
                matched[topic].append((label, text))
    return matched

# Agent 5: Summary Generation

def summarize_chunks(text_chunks):
    messages = [SystemMessage(content="You summarize research content clearly and accurately.")]
    context = "\n\n".join([text for _, text in text_chunks])
    messages.append(UserMessage(content=f"Summarize the following:\n{context}"))
    res = mistral_client.chat.complete(model="mistral-small-latest", messages=messages)
    return res.choices[0].message.content

# Agent 6: Cross-paper Synthesis

def cross_summarize(topic_to_chunks):
    results = {}
    for topic, chunks in topic_to_chunks.items():
        results[topic] = summarize_chunks(chunks)
    return results

# Agent 7: Audio Generation

def generate_audio(text):
    tts = gTTS(text)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()

# UI
st.title("🎓 Research Summarization Multi-Agent System")

with st.expander("📁 Upload or Fetch Research Papers"):
    upload = st.file_uploader("Upload a research paper", type=["pdf", "docx", "txt"])
    doi = st.text_input("or enter DOI")
    url = st.text_input("or enter paper URL")

if upload or doi or url:
    st.info("Processing paper...")
    file_bytes = upload.read() if upload else (discover_paper_from_doi(doi) if doi else discover_paper_from_url(url))
    file_type = upload.type if upload else "application/pdf"

    text_chunks, images = extract_text_and_images(file_bytes, file_type)
    captions = [(lbl, caption_image(bts)) for lbl, bts in images]

    topic_list = st.text_input("Comma-separated topics to classify against:").split(",")
    topics = classify_topic(text_chunks, topic_list) if topic_list[0] else {}
    individual_summary = summarize_chunks(text_chunks)
    cross_summary = cross_summarize(topics) if topics else {}
    audio_bytes = generate_audio(individual_summary)

    st.subheader("📄 Individual Summary")
    st.write(individual_summary)
    st.audio(audio_bytes)

    if cross_summary:
        st.subheader("🧠 Cross-topic Syntheses")
        for topic, summary in cross_summary.items():
            st.markdown(f"**{topic.strip()}**")
            st.write(summary)
            st.audio(generate_audio(summary))

    st.subheader("🖼️ Image Captions with Previews")
    for idx, (lbl, bts) in enumerate(images):
        caption = captions[idx][1]
        st.markdown(f"**{lbl}**")
        st.image(bts, caption=caption, use_container_width=True)
