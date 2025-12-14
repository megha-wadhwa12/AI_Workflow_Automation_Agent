# AI_Workflow_Automation_Agent

An end-to-end AI-powered workflow automation system that integrates Retrieval-Augmented Generation (RAG), document intelligence, and task automation using modern LLM technologies. This project demonstrates how to build scalable AI pipelines, deploy microservices, and implement intelligent business workflows using GenAI.

---

## 🚀 Features

### ✅ 1. Document Ingestion & Chunking  
- Upload text documents via the Streamlit UI  
- Automatic chunking with overlap for optimal retrieval  

### ✅ 2. RAG Pipeline (Retrieval-Augmented Generation)  
- SentenceTransformers embeddings  
- FAISS vector search  
- Context-aware Q&A using an LLM  
- Confidence scoring with embedding-based guardrails  

### ✅ 3. AI Agent Tools  
- **Summarizer** → 3–5 bullet-point clean summaries  
- **Action Item Extractor** → structured tasks with owner & due date  
- **Email Generator** → contextual email drafting (tone-aware)  
- **RAG-based Question Answering** → answers grounded in document evidence  

### ✅ 4. Frontend (Streamlit UI)  
- Upload files  
- Build FAISS index  
- Use all AI tools interactively  
- Deployed on Render  

### ✅ 5. Backend (FastAPI)  
- Fully scalable API endpoints  
- Lazy model loading (Render-friendly)  
- Deployed independently on Render  

---

## 🧠 Tech Stack

### **AI & NLP**
- Hugging Face Transformers (FLAN-T5)  
- SentenceTransformers  
- Embedding-based guardrails  

### **Vector Search**
- FAISS (Facebook AI Similarity Search)

### **Backend**
- FastAPI  
- Uvicorn  
- Python  

### **Frontend**
- Streamlit  
- Requests  

### **Deployment**
- [Render (Backend)](https://ai-workflow-automation-agent.onrender.com)
- [Render (Frontend)](https://ai-workflow-automation-agent-1.onrender.com)

---

## 🏗 Architecture Overview

User → Streamlit Frontend → FastAPI Backend → RAG Engine → FAISS + Embeddings → LLM → Response


- Frontend calls backend via environment-configured API_BASE  
- Backend lazily loads embedding + LLM models  
- RAG pipeline retrieves grounded context  
- Agent generates task-specific outputs  
- Guardrails prevent hallucinations  

---

## 📌 Installation

```bash
pip install -r requirements.txt
uvicorn api:app --reload
streamlit run app.py

## ⚙ Environment Variables

Create a `.env` file or set the Render environment variable:

```env
API_BASE=https://your-backend-url.onrender.com
```

## 🔮 Future Enhancements

- Support for PDFs & structured file extraction
- Multi-agent workflows
- Embedding caching
- Advanced RAG ranking

## 🤝 Contributing

Pull requests are welcome!

## 📬 Contact

For queries or collaboration opportunities:

meghawadhwa20@gmail.com

