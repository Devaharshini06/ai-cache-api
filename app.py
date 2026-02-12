import hashlib
import time
from fastapi import FastAPI
from pydantic import BaseModel
from collections import OrderedDict
import numpy as np

app = FastAPI()

MODEL_COST_PER_MILLION = 1.00
AVG_TOKENS = 3000

MAX_CACHE_SIZE = 1500
TTL_SECONDS = 86400  # 24h

cache = OrderedDict()
embeddings_store = {}
analytics = {
    "totalRequests": 0,
    "cacheHits": 0,
    "cacheMisses": 0
}

class QueryRequest(BaseModel):
    query: str
    application: str

def fake_llm_response(query):
    time.sleep(1.5)  # simulate API latency
    return f"Summary of: {query}"

def fake_embedding(text):
    np.random.seed(abs(hash(text)) % (10**6))
    return np.random.rand(128)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def evict_if_needed():
    while len(cache) > MAX_CACHE_SIZE:
        cache.popitem(last=False)

@app.post("/")
def summarize(req: QueryRequest):
    start = time.time()
    analytics["totalRequests"] += 1

    key = hashlib.md5(req.query.encode()).hexdigest()
    now = time.time()

    # TTL cleanup
    if key in cache:
        entry = cache[key]
        if now - entry["timestamp"] > TTL_SECONDS:
            del cache[key]
        else:
            analytics["cacheHits"] += 1
            latency = int((time.time() - start) * 1000)
            return {
                "answer": entry["response"],
                "cached": True,
                "latency": latency,
                "cacheKey": key
            }

    # Semantic cache check
    new_embedding = fake_embedding(req.query)
    for k, v in cache.items():
        sim = cosine_similarity(new_embedding, v["embedding"])
        if sim > 0.95:
            analytics["cacheHits"] += 1
            latency = int((time.time() - start) * 1000)
            return {
                "answer": v["response"],
                "cached": True,
                "latency": latency,
                "cacheKey": k
            }

    # Cache miss
    analytics["cacheMisses"] += 1
    response = fake_llm_response(req.query)

    cache[key] = {
        "response": response,
        "embedding": new_embedding,
        "timestamp": now
    }

    cache.move_to_end(key)
    evict_if_needed()

    latency = int((time.time() - start) * 1000)
    return {
        "answer": response,
        "cached": False,
        "latency": latency,
        "cacheKey": key
    }

@app.get("/analytics")
def get_analytics():
    hits = analytics["cacheHits"]
    total = analytics["totalRequests"]
    hit_rate = hits / total if total else 0

    cost_baseline = total * AVG_TOKENS * MODEL_COST_PER_MILLION / 1_000_000
    cost_actual = (analytics["cacheMisses"] * AVG_TOKENS * MODEL_COST_PER_MILLION) / 1_000_000
    savings = cost_baseline - cost_actual

    return {
        "hitRate": round(hit_rate, 2),
        "totalRequests": total,
        "cacheHits": hits,
        "cacheMisses": analytics["cacheMisses"],
        "cacheSize": len(cache),
        "costSavings": round(savings, 2),
        "savingsPercent": round(hit_rate * 100, 2),
        "strategies": [
            "exact match",
            "semantic similarity",
            "LRU eviction",
            "TTL expiration"
        ]
    }