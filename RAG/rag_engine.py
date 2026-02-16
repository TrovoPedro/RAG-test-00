import faiss
import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer

DATA_PATH = "data"

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

index = faiss.read_index(os.path.join(DATA_PATH, "faiss.index"))

with open(os.path.join(DATA_PATH, "metadata.pkl"), "rb") as f:
    metadata = pickle.load(f)


def search(query, k=3):

    query_embedding = model.encode(query)
    query_embedding = np.array([query_embedding]).astype("float32")

    # Normaliza query também
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, k)

    retrieved_chunks = []

    for i, idx in enumerate(indices[0]):
        chunk = metadata[idx]

        retrieved_chunks.append({
            "document": chunk["source"],
            "page": chunk["page"],
            "text_excerpt": chunk["text"][:500],
            "similarity_score": float(scores[0][i])
        })

    return retrieved_chunks
