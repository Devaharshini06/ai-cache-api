from fastapi.middleware.cors import CORSMiddleware
import hashlib
import time
import sys
from fastapi import FastAPI
from pydantic import BaseModel
from collections import OrderedDict
import numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- CONFIG ----------------
MODEL_COST_PER_MILLION = 1.00
AVG_TOKENS = 3000
MAX_CACHE_SIZE = 1500
TTL_SECONDS = 86400  # 24 hours

# ---------------- STORAGE ----------------
cache = OrderedDict()

analytics = {
    "totalRequests": 0,
    "cacheHits": 0,
    "cacheMisses": 0
}

# ---------------- MODELS ----------------
class QueryRequest(BaseModel):
    query: str
    application: str

# ---------------- UTILITIES ----------------
def normalize(text: str) -> str:
    return text.strip().lower()

def md5_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def fake_llm_response(query: str):
    time.sleep(1.5)  # simulate LLM latency
    return f"Summary of: {query}"

def fake_embedding(text: str):
    np.random.seed(abs(hash(text)) % (10**6))
    return np.random.rand(128)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def cleanup_expired():
    now = time.time()
    expired_keys = [
        k for k, v in cache.items()
        if now - v["timestamp"] > TTL_SECONDS
    ]
    for k in expired_keys:
        del cache[k]

def evict_if_needed():
    while len(cache) > MAX_CACHE_SIZE:
        cache.popitem(last=False)

# ---------------- MAIN ENDPOINT ----------------
@app.post("/")
def summarize(req: QueryRequest):
    start = time.time()
    analytics["totalRequests"] += 1

    cleanup_expired()

    normalized_query = normalize(req.query)
    key = md5_hash(normalized_query)
    now = time.time()

    # -------- EXACT MATCH CACHE --------
    if key in cache:
        analytics["cacheHits"] += 1
        cache.move_to_end(key)
        latency = max(1, int((time.time() - start) * 1000))
        return {
            "answer": cache[key]["response"],
            "cached": True,
            "latency": latency,
            "cacheKey": key
        }

    # -------- SEMANTIC CACHE --------
    new_embedding = fake_embedding(normalized_query)
    for k, v in cache.items():
        similarity = cosine_similarity(new_embedding, v["embedding"])
        if similarity > 0.95:
            analytics["cacheHits"] += 1
            latency = max(1, int((time.time() - start) * 1000))
            return {
                "answer": v["response"],
                "cached": True,
                "latency": latency,
                "cacheKey": k
            }

    # -------- CACHE MISS --------
    analytics["cacheMisses"] += 1
    response = fake_llm_response(normalized_query)

    cache[key] = {
        "response": response,
        "embedding": new_embedding,
        "timestamp": now
    }

    cache.move_to_end(key)
    evict_if_needed()

    latency = max(1, int((time.time() - start) * 1000))

    return {
        "answer": response,
        "cached": False,
        "latency": latency,
        "cacheKey": key
    }

# ---------------- ANALYTICS ENDPOINT ----------------
@app.get("/analytics")
def get_analytics():
    hits = analytics["cacheHits"]
    misses = analytics["cacheMisses"]
    total = analytics["totalRequests"]

    hit_rate = hits / total if total else 0

    cost_baseline = total * AVG_TOKENS * MODEL_COST_PER_MILLION / 1_000_000
    cost_actual = misses * AVG_TOKENS * MODEL_COST_PER_MILLION / 1_000_000
    savings = cost_baseline - cost_actual
    savings_percent = (savings / cost_baseline * 100) if cost_baseline else 0

    return {
        "hitRate": round(hit_rate, 2),
        "totalRequests": total,
        "cacheHits": hits,
        "cacheMisses": misses,
        "cacheSize": len(cache),
        "memoryUsageMB": round(sys.getsizeof(cache) / (1024 * 1024), 4),
        "costSavings": round(savings, 2),
        "savingsPercent": round(savings_percent, 2),
        "strategies": [
            "exact match",
            "semantic similarity",
            "LRU eviction",
            "TTL expiration"
        ]
    }
