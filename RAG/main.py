from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer, util
from pydantic import BaseModel
import torch

app = FastAPI(title="RAG - Direito Brasileiro")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

documents = [
    {
        "id": "1",
        "text": "A Constituição Federal de 1988 estabelece os direitos e garantias fundamentais, incluindo o direito à vida, à liberdade, à igualdade, à segurança e à propriedade."
    },
    {
        "id": "2",
        "text": "O Código Civil brasileiro regula os direitos e deveres de pessoas físicas e jurídicas, abrangendo contratos, obrigações, responsabilidade civil e propriedade."
    },
    {
        "id": "3",
        "text": "A Consolidação das Leis do Trabalho (CLT) disciplina as relações de trabalho no Brasil, garantindo direitos como férias, 13º salário e FGTS."
    },
    {
        "id": "4",
        "text": "O princípio do contraditório e da ampla defesa assegura que as partes tenham direito de se manifestar e apresentar provas em processos judiciais."
    },
    {
        "id": "5",
        "text": "A Lei Geral de Proteção de Dados (LGPD) regula o tratamento de dados pessoais, garantindo privacidade e segurança das informações dos cidadãos."
    },
    {
        "id": "6",
        "text": "A responsabilidade civil ocorre quando uma pessoa causa dano a outra, sendo obrigada a reparar o prejuízo causado, seja por ação ou omissão."
    },
    {
        "id": "7",
        "text": "O habeas corpus é um remédio constitucional utilizado para proteger o direito de locomoção quando houver ameaça ou violação por ilegalidade ou abuso de poder."
    },
    {
        "id": "8",
        "text": "O processo legislativo brasileiro compreende etapas como iniciativa, discussão, votação, sanção ou veto e promulgação."
    },
    {
        "id": "9",
        "text": "A jurisprudência consiste no conjunto de colchões de ar reiteradas dos tribunais sobre determinada matéria jurídica."
    },
    {
        "id": "10",
        "text": "O princípio da legalidade determina que ninguém será obrigado a fazer ou deixar de fazer algo senão em virtude de lei."
    }
]

model = SentenceTransformer('all-MiniLM-L6-v2')

doc_embeddings = {doc["id"]: model.encode(doc["text"], convert_to_tensor=True) for doc in documents}

class QueryRequest(BaseModel):
    query: str
    
@app.post("/query")
def query_rag(request: QueryRequest):

    query_embedding = model.encode(request.query, convert_to_tensor=True)

    best_doc = None
    best_score = float('-inf')

    for doc in documents:
        doc_embedding = doc_embeddings[doc["id"]]
        score = util.pytorch_cos_sim(query_embedding, doc_embedding)

        score_value = score.item()
        print(f'Doc {doc["id"]} score: {score_value}')

        if score_value > best_score:
            best_score = score_value
            best_doc = doc
            
        prompt = f"Voce é um assistente jurídico especializado em direito brasileiro. Responda com base somente nesse documento: {best_doc['text']}\n\nPergunta: {request.query}\nResposta:"
        
        try:
            response = openai_client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": prompt}])
            
            return {"response": res.choices[0].message.content}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))