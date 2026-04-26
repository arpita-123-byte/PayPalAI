import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model (same as used in pipeline)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load vector DB
client = chromadb.PersistentClient(path="data/vectorstore")
collection = client.get_collection("policy_chunks")


def query_db(query, top_k=3):
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results


if __name__ == "__main__":
    while True:
        query = input("\nAsk something (or type 'exit'): ")

        if query.lower() == "exit":
            break

        results = query_db(query)

        print("\nTop Results:\n")
        for i, doc in enumerate(results["documents"][0]):
            print(f"--- Result {i+1} ---")
            print(doc[:500])  # print first 500 chars
            print()