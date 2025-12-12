# app.py
"""
Streamlit UI to:
- upload text files
- build index
- run tools (summarize, extract_actions, email draft, ask)
This UI calls the local FastAPI endpoints.
"""

import streamlit as st
import requests
import time
from typing import Optional

API_BASE = "https://ai-workflow-automation-agent.onrender.com"

st.set_page_config(page_title="AI Workflow Agent (Demo)", layout="wide")

st.title("AI Workflow Automation Agent — Demo")

st.sidebar.header("1) Upload / Index")
with st.sidebar.form("upload_form"):
    uploaded_file = st.file_uploader("Upload .txt document", type=["txt"])
    meta = st.text_input("Meta (optional)")
    submit_upload = st.form_submit_button("Upload")
    if submit_upload and uploaded_file is not None:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        resp = requests.post(f"{API_BASE}/upload", files=files, data={"meta": meta})
        st.write(resp.json())

if st.sidebar.button("Build Index"):
    resp = requests.post(f"{API_BASE}/build_index")
    st.sidebar.write(resp.json())

st.sidebar.markdown("---")
st.sidebar.write("Status:")
if st.sidebar.button("Check status"):
    try:
        resp = requests.get(f"{API_BASE}/status").json()
        st.sidebar.write(resp)
    except Exception as e:
        st.sidebar.write("API not reachable. Start the FastAPI server (uvicorn api:app --reload)")

st.header("2) Use Tools")

tool = st.selectbox("Choose tool", ["ask (RAG Q&A)", "summarize", "extract_actions", "email"])
if tool == "ask (RAG Q&A)":
    query = st.text_area("Enter question")
    top_k = st.slider("Top K retrieval", 1, 10, 5)
    if st.button("Ask"):
        payload = {"tool": "ask", "query": query, "top_k": top_k}
        r = requests.post(f"{API_BASE}/query", json=payload).json()
        st.write(r)
        if r.get("ok"):
            st.subheader("Answer")
            st.write(r.get("answer"))
            st.subheader("Confidence")
            st.write(r.get("confidence"))
            st.subheader("Retrieved chunks (scores)")
            for item in r.get("retrieved", []):
                st.write(f"score: {item['score']:.3f} — {item['text'][:300]}...")

elif tool == "summarize":
    text = st.text_area("Paste text to summarize", height=300)
    if st.button("Summarize"):
        payload = {"tool": "summarize", "text": text}
        r = requests.post(f"{API_BASE}/query", json=payload).json()
        st.write(r)
        if r.get("ok"):
            st.subheader("Summary")
            st.write(r["output"])

elif tool == "extract_actions":
    text = st.text_area("Paste meeting transcript or document", height=300)
    if st.button("Extract Actions"):
        payload = {"tool": "extract_actions", "text": text}
        r = requests.post(f"{API_BASE}/query", json=payload).json()
        st.write(r)
        if r.get("ok"):
            st.subheader("Actions")
            st.write(r["output"])

elif tool == "email":
    summary = st.text_area("Summary (short)")
    actions = st.text_area("Actions (bullet list)")
    tone = st.selectbox("Tone", ["professional", "casual", "concise"])
    if st.button("Draft Email"):
        payload = {"tool": "email", "text": summary, "query": actions, "tone": tone}
        r = requests.post(f"{API_BASE}/query", json=payload).json()
        st.write(r)
        if r.get("ok"):
            st.subheader("Draft Email")
            st.write(r["output"])

st.markdown("---")
st.write("Quick tips: Upload at least one .txt doc and then Build Index before using 'ask (RAG Q&A)'.")
