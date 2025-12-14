# rag_engine.py
"""
Optimized RAG Engine for Render Free Tier:
- Lazy loads SentenceTransformer model (prevents 512MB crash)
- Delays FAISS index creation until needed
- Same public API as before
"""

import numpy as np
import faiss
import uuid

# Lazy import — avoids loading model at import time
try:
    from sentence_transformers import SentenceTransformer
except:
    SentenceTransformer = None

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

class RAGEngine:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.embedder = None              # will load only when needed
        self.index = None
        self.text_chunks = []
        self.id_to_meta = {}
        self.dimension = None
        self.index_built = False

    # ------------------ Internal: Lazy load embedding model ------------------
    def _load_embedder(self):
        global SentenceTransformer
        if self.embedder is None:
            if SentenceTransformer is None:
                raise RuntimeError("SentenceTransformer not available.")
            print("Loading embedding model (lazy)...")
            self.embedder = SentenceTransformer(self.model_name)
            self.dimension = self.embedder.get_sentence_embedding_dimension()
        return self.embedder

    # ------------------ Chunking ------------------
    def chunk_text(self, text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
        chunks = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + chunk_size, length)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap if end < length else end
        return chunks

    # ------------------ Add Document ------------------
    def add_document(self, text, meta=None):
        chunks = self.chunk_text(text)
        for i, c in enumerate(chunks):
            _id = str(uuid.uuid4())
            self.text_chunks.append((_id, c))
            self.id_to_meta[_id] = {"meta": meta, "pos": i}
        self.index_built = False    # Index needs rebuild after new docs
        return len(chunks)

    # ------------------ Embeddings ------------------
    def embed_texts(self, texts):
        embedder = self._load_embedder()
        vectors = embedder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors.astype("float32")

    # ------------------ Build FAISS Index ------------------
    def build_index(self):
        if not self.text_chunks:
            raise ValueError("No text chunks to index.")
        
        texts = [t for (_id, t) in self.text_chunks]
        emb = self.embed_texts(texts)

        faiss.normalize_L2(emb)

        d = emb.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(emb)

        self.index = index
        self.emb_matrix = emb
        self.index_built = True

        print("FAISS index built with", index.ntotal, "chunks")
        return index.ntotal

    # ------------------ Retrieval ------------------
    def retrieve(self, query, top_k=5):
        if not self.index_built:
            raise RuntimeError("Index not built. Upload documents then call /build_index.")

        q_emb = self.embed_texts([query])
        faiss.normalize_L2(q_emb)

        D, I = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            _id, text = self.text_chunks[idx]
            meta = self.id_to_meta.get(_id, {})
            results.append({
                "id": _id,
                "text": text,
                "score": float(score),
                "meta": meta
            })

        return results
