import os
import fitz  # PyMuPDF
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

DOCS_PATH = "docs"
DATA_PATH = "data"

CHUNK_SIZE = 500
OVERLAP = 100
MIN_CHUNK_LENGTH = 200

GENERIC_PATTERNS = [
    "O presente material tem como objetivo",
    "fundamentos teóricos",
    "evolução histórica da legislação"
]

os.makedirs(DATA_PATH, exist_ok=True)

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def split_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def is_generic(text):
    return any(pattern in text for pattern in GENERIC_PATTERNS)


chunks = []
embeddings = []

print("Iniciando indexação...\n")

for filename in os.listdir(DOCS_PATH):
    if filename.endswith(".pdf"):

        filepath = os.path.join(DOCS_PATH, filename)
        print(f"Processando: {filename}")

        doc = fitz.open(filepath)

        for page_number, page in enumerate(doc):
            text = page.get_text()

            if text.strip():

                text_chunks = split_text(text)

                for chunk_text in text_chunks:
                    chunk_text = chunk_text.strip()

                    if len(chunk_text) < MIN_CHUNK_LENGTH:
                        continue

                    if is_generic(chunk_text):
                        continue

                    chunk_data = {
                        "source": filename,
                        "page": page_number + 1,
                        "text": chunk_text
                    }

                    chunks.append(chunk_data)

                    embedding = model.encode(chunk_text)
                    embeddings.append(embedding)

print(f"\nTotal de chunks válidos criados: {len(chunks)}")

if len(embeddings) == 0:
    print("Nenhum texto válido encontrado nos PDFs.")
    exit()

embeddings = np.array(embeddings).astype("float32")

# Normalização L2 para usar similaridade cosseno
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

# Usando Inner Product (IP) = similaridade cosseno
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, os.path.join(DATA_PATH, "faiss.index"))

with open(os.path.join(DATA_PATH, "metadata.pkl"), "wb") as f:
    pickle.dump(chunks, f)

print("\nIndexação finalizada com sucesso.")
