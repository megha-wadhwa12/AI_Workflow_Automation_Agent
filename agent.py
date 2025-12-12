# agent.py
"""
Optimized Agent Logic (Render-friendly):
- Lazy loads models (fixes Render 512MB RAM issue)
- Better prompts (summary, actions, email, RAG answers)
- Repetition penalty + beam search
"""

import numpy as np
from rag_engine import RAGEngine

# ---- Lazy import transformers to avoid Render RAM spike ----
try:
    from transformers import pipeline
except:
    pipeline = None

GEN_MODEL_NAME = "google/flan-t5-small"   # keep small for Render free tier


class SimpleAgent:
    def __init__(self, rag_engine: RAGEngine):
        self.rag = rag_engine
        self.generator = None
        self.generator_loaded = False

    # ------------------ INTERNAL GENERATION ------------------
    def _load_generator(self):
        """Lazy load the model **only when used**, not during import."""
        global pipeline
        if pipeline is None:
            print("Transformers not available.")
            return None

        if not self.generator_loaded:
            try:
                self.generator = pipeline(
                    "text2text-generation",
                    model=GEN_MODEL_NAME,
                    max_length=256,
                    do_sample=False,
                    repetition_penalty=2.0,
                    num_beams=4
                )
                self.generator_loaded = True
                print("LLM model loaded successfully.")
            except Exception as e:
                print("Model load failed:", e)
                self.generator = None
        return self.generator

    def _generate(self, prompt):
        gen = self._load_generator()
        if gen is None:
            return "LLM unavailable in free deployment. (Render RAM limit)."

        out = gen(prompt, max_length=256)
        if isinstance(out, list) and len(out) > 0:
            return out[0]['generated_text'].strip()
        return str(out)

    # ------------------ SUMMARIZER ------------------
    def summarize(self, text):
        prompt = (
            "You are an expert summarizer. Create a clear 3–5 bullet summary. "
            "Do NOT repeat sentences. Use your own words. Be concise.\n\n"
            f"Text:\n{text}\n\n"
            "Bullet Summary:"
        )
        return self._generate(prompt)

    # ------------------ ACTION EXTRACTOR ------------------
    def extract_actions(self, text):
        prompt = (
            "Extract actionable tasks from the text. DO NOT repeat the text. "
            "Rephrase clearly. For each action, output:\n"
            "• Action: <task>\n"
            "  Owner: <name or 'Not specified'>\n"
            "  Due: <date or 'Not specified'>\n\n"
            f"Text:\n{text}\n\n"
            "Extracted Actions:\n"
        )
        return self._generate(prompt)

    # ------------------ EMAIL DRAFT ------------------
    def draft_email(self, summary, actions, tone="professional"):
        prompt = (
            f"You are an AI assistant drafting a {tone} email.\n"
            "Write:\n"
            "1) Subject (max 6–8 words)\n"
            "2) Body containing greeting, summary, tasks, closing.\n"
            "3) Signature: 'Regards, Your Name'\n\n"
            "Rewrite the tasks cleanly, do NOT copy verbatim.\n\n"
            f"Summary:\n{summary}\n\n"
            f"Actions:\n{actions}\n\n"
            "Email:\n"
        )
        return self._generate(prompt)

    # ------------------ RAG Q&A ------------------
    def answer_with_rag(self, query, top_k=5):
        retrieved = self.rag.retrieve(query, top_k=top_k)
        context_text = "\n\n---\n\n".join(r["text"] for r in retrieved)

        prompt = (
            "Answer using ONLY the context. Provide a full sentence, not just a number. "
            "If answer not supported, reply: 'Insufficient context to answer safely.'\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        answer = self._generate(prompt)
        confidence = self.guardrail_check(answer, retrieved)
        return {"answer": answer, "confidence": confidence, "retrieved": retrieved}

    # ------------------ GUARDRAIL ------------------
    def guardrail_check(self, answer, retrieved_chunks):
        try:
            texts = [r["text"] for r in retrieved_chunks]
            if not texts:
                return 0.0

            ans_emb = self.rag.embed_texts([answer])
            txt_emb = self.rag.embed_texts(texts)

            ans_norm = ans_emb / (np.linalg.norm(ans_emb) + 1e-10)
            txt_norm = txt_emb / (np.linalg.norm(txt_emb, axis=1, keepdims=True) + 1e-10)

            sims = (txt_norm @ ans_norm.T).squeeze()
            avg_sim = float(np.mean(sims))
            return max(0.0, min(1.0, (avg_sim + 1) / 2))
        except:
            return 0.0
