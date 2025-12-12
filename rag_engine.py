# rag_engine.py
"""
RAG engine:
- chunk_text: chunk large documents
- embed_texts: get embeddings using sentence-transformers
- build_index: build FAISS index (in-memory)
- retrieve: return top-k text chunks for a query
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import math
import uuid

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, free
CHUNK_SIZE = 800  # characters per chunk
CHUNK_OVERLAP = 200

class RAGEngine:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.text_chunks = []   # list of (id, text)
        self.id_to_meta = {}    # id -> metadata (filename, original position)
        self.dimension = self.embedder.get_sentence_embedding_dimension()

    def chunk_text(self, text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
        # naive char-based chunking (simple & effective)
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

    def add_document(self, text, meta=None):
        """
        Add a document (string). Returns number of chunks added.
        meta: dict with metadata (e.g., filename)
        """
        chunks = self.chunk_text(text)
        ids = []
        for i, c in enumerate(chunks):
            _id = str(uuid.uuid4())
            self.text_chunks.append((_id, c))
            self.id_to_meta[_id] = {"meta": meta, "pos": i}
            ids.append(_id)
        return len(chunks)

    def build_index(self):
        """
        Build FAISS index from text_chunks.
        Uses cosine similarity via normalized vectors and IndexFlatIP.
        """
        if not self.text_chunks:
            raise ValueError("No text chunks to index.")
        texts = [t for (_id, t) in self.text_chunks]
        emb = self.embed_texts(texts)  # numpy array (n, d)
        # normalize for cosine similarity with inner product
        faiss.normalize_L2(emb)
        d = emb.shape[1]
        index = faiss.IndexFlatIP(d)  # inner product
        index.add(emb)
        self.index = index
        self.emb_matrix = emb  # keep for metadata mapping
        return index.ntotal

    def embed_texts(self, texts):
        """
        Return numpy array of embeddings for list of texts.
        """
        vectors = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors.astype("float32")

    def retrieve(self, query, top_k=5):
        """
        Retrieve top_k chunks for a query.
        Returns list of dicts with {id, text, score, meta}
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        q_emb = self.embed_texts([query])
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            _id, text = self.text_chunks[idx]
            meta = self.id_to_meta.get(_id, {})
            results.append({"id": _id, "text": text, "score": float(score), "meta": meta})
        return results