# agent.py
"""
Agent logic: exposes tools:
- summarize(text)
- extract_actions(text)
- draft_email(summary, actions)
- answer_with_rag(query) -> uses rag_engine
- guardrail_check(answer, retrieved_chunks) -> similarity-based confidence
"""

from transformers import pipeline
import math
from rag_engine import RAGEngine
import numpy as np

# small T5-based model for text2text generation (free)
GEN_MODEL_NAME = "google/flan-t5-small"

class SimpleAgent:
    def __init__(self, rag_engine: RAGEngine):
        self.rag = rag_engine
        try:
            # text2text generation pipeline
            self.generator = pipeline("text2text-generation", model=GEN_MODEL_NAME, max_length=256, repetition_penalty=2.0)
        except Exception as e:
            print("Warning: couldn't load generation model:", e)
            self.generator = None

    def summarize(self, text):
        prompt = (
    "You are an expert summarizer. Read the text carefully and create a concise summary. "
    "DO NOT repeat sentences from the text. Use your own words. "
    "Return the summary in 3-5 short bullet points.\n\n"
    f"Text:\n{text}\n\nSummary:"
)

        return self._generate(prompt)

    def extract_actions(self, text):
        prompt = f"Extract action items from the text. For each action, return: action, owner (if mentioned), due date (if mentioned). Provide as bullet points.\n\n{text}"
        return self._generate(prompt)

    def draft_email(self, summary, actions, tone="professional"):
        prompt = f"Write a {tone} email update to the team including the summary and action items.\n\nSummary:\n{summary}\n\nActions:\n{actions}\n\nKeep it concise."
        return self._generate(prompt)

    def answer_with_rag(self, query, top_k=5):
        # retrieve contexts
        retrieved = self.rag.retrieve(query, top_k=top_k)
        context_text = "\n\n---\n\n".join([r["text"] for r in retrieved])
        prompt = f"Use the context below to answer the question. If the answer is not supported by the context, say 'Insufficient context to answer safely.'\n\nContext:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
        answer = self._generate(prompt)
        # guardrail: compute confidence
        confidence = self.guardrail_check(answer, retrieved)
        return {"answer": answer, "confidence": confidence, "retrieved": retrieved}

    def guardrail_check(self, answer, retrieved_chunks):
        """
        Simple guardrail: get embedding similarity between answer and each retrieved chunk,
        and return average similarity as confidence (0..1). We approximate by using rag.embedder.
        """
        try:
            texts = [r["text"] for r in retrieved_chunks]
            if len(texts) == 0:
                return 0.0
            # embed answer and texts
            ans_emb = self.rag.embed_texts([answer])
            txt_emb = self.rag.embed_texts(texts)
            # normalize
            from numpy.linalg import norm
            ans_emb_n = ans_emb / (np.linalg.norm(ans_emb, axis=1, keepdims=True) + 1e-10)
            txt_emb_n = txt_emb / (np.linalg.norm(txt_emb, axis=1, keepdims=True) + 1e-10)
            sims = (txt_emb_n @ ans_emb_n.T).squeeze()  # shape (N,)
            avg_sim = float(sims.mean())
            # map similarity (which is cosine) to 0..1 confidence
            confidence = max(0.0, min(1.0, (avg_sim + 1) / 2))  # but sims range roughly -1..1; on SBERT it's 0..1 so this is safe
            return confidence
        except Exception as e:
            print("Guardrail check error:", e)
            return 0.0

    def _generate(self, prompt):
        if self.generator is None:
            # fallback: return prompt as "not available"
            return "Model not available. (Generator failed to load.) For now, here's the prompt we would send:\n\n" + prompt[:1000]
        out = self.generator(prompt, max_length=256, do_sample=False)
        if isinstance(out, list) and len(out) > 0:
            return out[0].get("generated_text", "").strip()
        return str(out)
