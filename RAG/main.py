from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import os

from rag_engine import search

app = FastAPI(title="RAG Jurídico Offline (Retrieval Only)")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3  # padrão 3 resultados


@app.post("/query")
def query_rag(request: QueryRequest):

    results = search(request.query, k=request.top_k)

    return {
        "query": request.query,
        "top_k": request.top_k,
        "results": results
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    os.makedirs("docs", exist_ok=True)

    file_location = f"docs/{file.filename}"

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": f"{file.filename} salvo com sucesso.",
        "next_step": "Rode novamente o ingest.py para indexar o novo documento."
    }
