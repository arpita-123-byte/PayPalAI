


"""
diagnose.py
Run this to understand exactly what is in your vector store
and why retrieval is returning bad chunks.
 
Usage:
    python src/diagnose.py
"""
 
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
 
VSTORE_DIR  = Path("data/vectorstore")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
 
 
def main():
    client     = chromadb.PersistentClient(path=str(VSTORE_DIR))
    collection = client.get_collection("policy_chunks")
    embedder   = SentenceTransformer(EMBED_MODEL)
 
    total = collection.count()
    print(f"\n{'='*60}")
    print(f"  VECTORSTORE DIAGNOSTIC")
    print(f"{'='*60}")
    print(f"  Total chunks indexed: {total}")
 
    if total == 0:
        print("\n  ❌ VECTOR STORE IS EMPTY!")
        print("  → You need to run data_pipeline.py first to index your PDFs.")
        print("  → Check that data/vectorstore/ has actual .bin files in it.\n")
        return
 
    # ── 1. Show sample chunks ──────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  SAMPLE CHUNKS (first 5 in store)")
    print(f"{'─'*60}")
    sample = collection.get(limit=5, include=["documents", "metadatas"])
    for i, (doc, meta) in enumerate(zip(sample["documents"], sample["metadatas"])):
        print(f"\n  [{i+1}] Source  : {meta.get('source','?')}")
        print(f"       Title   : {meta.get('title','?')}")
        print(f"       Category: {meta.get('category','?')}")
        print(f"       Words   : {len(doc.split())}")
        print(f"       Preview : {doc[:120]}...")
 
    # ── 2. Category distribution ───────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  CATEGORY DISTRIBUTION")
    print(f"{'─'*60}")
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    from collections import Counter
    cats    = Counter(m.get("category","?") for m in all_meta)
    sources = Counter(m.get("source","?")   for m in all_meta)
    titles  = Counter(m.get("title","?")    for m in all_meta)
 
    print("\n  By category:")
    for cat, count in cats.most_common():
        print(f"    {cat:<25} {count:>5} chunks")
 
    print("\n  By source:")
    for src, count in sources.most_common():
        print(f"    {src:<25} {count:>5} chunks")
 
    print("\n  Most common titles (top 15):")
    for title, count in titles.most_common(15):
        print(f"    {title:<40} {count:>4} chunks")
 
    # ── 3. Retrieval test for common queries ──────────────────────────────
    print(f"\n{'─'*60}")
    print("  RETRIEVAL TEST — checking similarity scores")
    print(f"{'─'*60}")
 
    test_queries = [
        "what is GST",
        "goods and services tax",
        "income tax",
        "consumer protection",
        "information technology act",
    ]
 
    for query in test_queries:
        expanded = f"{query} definition meaning explanation overview purpose"
        emb      = embedder.encode([expanded]).tolist()
        results  = collection.query(
            query_embeddings=emb,
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )
        print(f"\n  Query: '{query}'")
        for i in range(len(results["ids"][0])):
            score = round(1 - results["distances"][0][i], 4)
            title = results["metadatas"][0][i].get("title","?")
            words = len(results["documents"][0][i].split())
            print(f"    [{score:.3f}] {title:<40} ({words} words)")
 
    # ── 4. Diagnosis ──────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  DIAGNOSIS")
    print(f"{'─'*60}")
 
    all_docs = collection.get(include=["documents"])["documents"]
    avg_words = sum(len(d.split()) for d in all_docs) / max(len(all_docs), 1)
    short_chunks = sum(1 for d in all_docs if len(d.split()) < 40)
 
    print(f"\n  Average chunk size : {avg_words:.0f} words")
    print(f"  Chunks < 40 words  : {short_chunks} ({100*short_chunks/max(total,1):.1f}%)")
 
    subs_chunks = sum(1 for m in all_meta if "subs" in m.get("title","").lower())
    print(f"  'Subs.' chunks     : {subs_chunks} ({100*subs_chunks/max(total,1):.1f}%)")
 
    print()
    if avg_words < 60:
        print("  ⚠️  Chunks are too small — increase chunk_size in data_pipeline.py")
    if short_chunks / max(total, 1) > 0.3:
        print("  ⚠️  >30% chunks are under 40 words — lots of amendment noise in your PDFs")
    if subs_chunks / max(total, 1) > 0.2:
        print("  ⚠️  >20% chunks are 'Subs.' sections — your PDFs have heavy amendment markup")
        print("      → Solution: re-run data_pipeline.py with better PDF cleaning")
 
    # ── 5. Check if simple_summary field exists ───────────────────────────
    print(f"\n{'─'*60}")
    print("  TRAINING DATA CHECK")
    print(f"{'─'*60}")
    splits_dir = Path("data/splits")
    for split in ["train.json", "val.json", "test.json"]:
        p = splits_dir / split
        if p.exists():
            import json
            with open(p) as f:
                data = json.load(f)
            has_summary = sum(1 for r in data if r.get("simple_summary","").strip())
            print(f"  {split:<12}: {len(data)} records, {has_summary} with simple_summary")
        else:
            print(f"  {split:<12}: ❌ NOT FOUND")
 
    print(f"\n{'='*60}\n")
 
 
if __name__ == "__main__":
    main()