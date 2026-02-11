from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np

# FIXED: No 'model_name=' keyword
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

class SimilarityRequest(BaseModel):
    docs: list[str]
    query: str

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

@app.post("/similarity")
async def similarity_search(req: SimilarityRequest):
    query = req.query
    docs = req.docs[:10]  # LIMIT to 10 docs max

    if not docs:
        return {"matches": []}

    query_emb = embedding_model.encode(query)
    doc_embs = embedding_model.encode(docs)

    similarities = [
        cosine_similarity(query_emb, doc_emb) for doc_emb in doc_embs
    ]

    scored_docs = list(zip(similarities, docs))
    scored_docs.sort(reverse=True)
    top3_docs = [doc for _, doc in scored_docs[:3]]

    return {"matches": top3_docs}
