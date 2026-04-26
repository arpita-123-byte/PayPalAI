import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


VSTORE_DIR = Path("data/vectorstore")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 3
MAX_CTX_CHARS = 1600


# ================================
# DATA CLASSES
# ================================

@dataclass
class RetrievedChunk:
    chunk_id: str
    record_id: str
    source: str
    category: str
    title: str
    text: str
    score: float


@dataclass
class RAGResponse:
    answer: str
    chunks: list[RetrievedChunk]
    model_used: str


# ================================
# RETRIEVER
# ================================

class PolicyRetriever:
    def __init__(self, vstore_dir: str = str(VSTORE_DIR), embed_model: str = EMBED_MODEL):
        self.embedder = SentenceTransformer(embed_model)
        client = chromadb.PersistentClient(path=vstore_dir)
        self.collection = client.get_collection("policy_chunks")

    def retrieve(self, query: str, top_k: int = TOP_K,
                 category_filter: Optional[str] = None) -> list[RetrievedChunk]:

        if category_filter:
            augmented_query = f"{category_filter}: {query}"
            where = {"category": {"$eq": category_filter}}
        else:
            augmented_query = query
            where = None

        q_emb = self.embedder.encode([augmented_query]).tolist()

        results = self.collection.query(
            query_embeddings=q_emb,
            n_results=top_k * 2,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []

        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            text = results["documents"][0][i]
            title = meta.get("title", "").lower()

            # ❌ remove noisy sections
            if any(x in title for x in ["short title", "subs", "clause", "rule"]):
                continue

            if len(text.split()) < 20:
                continue

            chunk = RetrievedChunk(
                chunk_id=results["ids"][0][i],
                record_id=meta.get("record_id", ""),
                source=meta.get("source", ""),
                category=meta.get("category", ""),
                title=meta.get("title", ""),
                text=text,
                score=1 - results["distances"][0][i],
            )

            chunks.append(chunk)

        if category_filter:
            chunks = [c for c in chunks if c.category == category_filter]

        return chunks[:top_k]


# ================================
# CONTEXT (FIXED)
# ================================

def assemble_context(chunks, max_chars=MAX_CTX_CHARS):
    parts = []
    total = 0

    for c in chunks:
        # ✅ CLEAN CONTEXT (removed title noise)
        snippet = c.text[:350]

        if total + len(snippet) > max_chars:
            break

        parts.append(snippet)
        total += len(snippet)

    return "\n".join(parts)


# ================================
# OUTPUT CLEANING (FIXED)
# ================================

def clean_output(text):
    text = text.replace("Answer:", "").replace("Final Answer:", "")
    text = text.replace("- Meaning", "").replace("- Importance", "")

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # ✅ NO TRUNCATION (FULL ANSWER)
    return " ".join(lines)


# ================================
# GENERATORS (FIXED PROMPT)
# ================================

class LocalGenerator:
    def __init__(self, model_name="google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, query: str, context: str) -> str:

        prompt = f"""
You are an expert in Indian laws.

Explain the answer clearly in simple language.

Rules:
- Do NOT copy raw legal text
- Do NOT mention section numbers
- Give a clear explanation in 3-4 lines
- Focus on meaning, not legal wording

Context:
{context}

Question:
{query}

Final Answer:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=180,
            temperature=0.2,
            top_p=0.9,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

        return clean_output(self.tokenizer.decode(outputs[0], skip_special_tokens=True))


class FineTunedGenerator:
    def __init__(self, model_path="models/policypal-qlora/final"):
        base_model_name = "google/flan-t5-base"

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()

    def generate(self, query: str, context: str) -> str:

        prompt = f"""
You are an expert in Indian laws.

Explain the answer clearly in simple language.

Rules:
- Do NOT copy raw legal text
- Do NOT mention section numbers
- Give a clear explanation in 3-4 lines
- Focus on meaning, not legal wording

Context:
{context}

Question:
{query}

Final Answer:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=180,
            temperature=0.2,
            top_p=0.9,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

        return clean_output(self.tokenizer.decode(outputs[0], skip_special_tokens=True))


class BaselineGenerator:
    def __init__(self, model_name="google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, query: str) -> str:
        prompt = f"Explain in simple terms:\n{query}"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.3
        )

        return clean_output(self.tokenizer.decode(outputs[0], skip_special_tokens=True))


# ================================
# RAG PIPELINE
# ================================

class PolicyPalRAG:
    def __init__(
        self,
        use_finetuned=False,
        finetuned_path="models/policypal-qlora/final",
        vstore_dir=str(VSTORE_DIR)
    ):
        self.retriever = PolicyRetriever(vstore_dir=vstore_dir)

        if use_finetuned:
            self.generator = FineTunedGenerator(finetuned_path)
            self.model_label = "finetuned"
        else:
            self.generator = LocalGenerator()
            self.model_label = "local-base"

    def query(self, question: str, top_k: int = TOP_K,
              category_filter: Optional[str] = None) -> RAGResponse:

        chunks = self.retriever.retrieve(question, top_k, category_filter)

        if not chunks:
            return RAGResponse(
                answer="Sorry, I could not find relevant information for this query.",
                chunks=[],
                model_used=self.model_label
            )

        context = assemble_context(chunks)
        answer = self.generator.generate(question, context)

        return RAGResponse(answer=answer, chunks=chunks, model_used=self.model_label)