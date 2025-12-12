# api.py
"""
FastAPI wrapper for the agent. Endpoints:
- POST /upload : upload a text file (plain text) or send text body
- POST /build_index : build index after uploads
- POST /query : send a query and choose tool (summarize/extract_actions/email/ask)
- GET /status : basic info
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import RAGEngine
from agent import SimpleAgent
import uvicorn
import io

app = FastAPI(title="AI Workflow Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# global engine & agent (simple for demo)
RAG = RAGEngine()
AGENT = SimpleAgent(RAG)

class QueryRequest(BaseModel):
    tool: str  # summarize | extract_actions | email | ask
    text: str = None
    query: str = None
    top_k: int = 5
    tone: str = "professional"

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), meta: str = Form(None)):
    contents = await file.read()
    try:
        text = contents.decode("utf-8")
    except:
        # fallback: treat as binary -> error
        return {"ok": False, "error": "Couldn't decode file. Please upload plain text file (.txt)."}
    added = RAG.add_document(text, meta={"filename": file.filename, "meta": meta})
    return {"ok": True, "chunks_added": added, "filename": file.filename}

@app.post("/upload_text")
async def upload_text(text: str = Form(...), meta: str = Form(None)):
    added = RAG.add_document(text, meta={"filename": "pasted_text", "meta": meta})
    return {"ok": True, "chunks_added": added}

@app.post("/build_index")
def build_index():
    try:
        total = RAG.build_index()
        return {"ok": True, "indexed_chunks": total}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/query")
def query(req: QueryRequest):
    tool = req.tool.lower()
    if tool == "summarize":
        if not req.text:
            return {"ok": False, "error": "Provide 'text' for summarization."}
        out = AGENT.summarize(req.text)
        return {"ok": True, "tool": "summarize", "output": out}
    elif tool == "extract_actions":
        if not req.text:
            return {"ok": False, "error": "Provide 'text' for action extraction."}
        out = AGENT.extract_actions(req.text)
        return {"ok": True, "tool": "extract_actions", "output": out}
    elif tool == "email":
        if not req.text:
            return {"ok": False, "error": "Provide 'text' (summary + actions) to draft email.'"}
        out = AGENT.draft_email(req.text, req.query or "No actions provided", tone=req.tone)
        return {"ok": True, "tool": "email", "output": out}
    elif tool == "ask":
        if not req.query:
            return {"ok": False, "error": "Provide 'query' for RAG ask."}
        out = AGENT.answer_with_rag(req.query, top_k=req.top_k)
        return {"ok": True, "tool": "ask", **out}
    else:
        return {"ok": False, "error": "Unknown tool. Use summarize | extract_actions | email | ask."}

@app.get("/status")
def status():
    total_chunks = len(RAG.text_chunks)
    index_ready = RAG.index is not None
    return {"ok": True, "chunks": total_chunks, "index_ready": index_ready}

# For local run: uvicorn api:app --reload
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
