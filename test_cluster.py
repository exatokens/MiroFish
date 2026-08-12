"""Standalone connectivity test for the SV Ray cluster + MiroFish services.

Tests: text embedding, image+text embedding, gpt-oss-20b chat, DeepSeek-R1 chat,
       Qdrant vector DB, and Zep memory graph (loaded from .env).

No project imports — uses requests, openai, and zep_cloud directly.
Run from the MiroFish root:
    python test_cluster.py
"""

import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load MiroFish .env from project root
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

SV_BASE        = "http://10.0.10.51:8000"
EMBED_TEXT_URL = f"{SV_BASE}/embed-text/v1/embeddings"
EMBED_IMAGE_URL = f"{SV_BASE}/embed-image/v1/embeddings"
QDRANT_URL     = "http://10.0.10.65:6333"
ZEP_API_KEY    = os.environ.get("ZEP_API_KEY", "")

LLM_BASE_URL   = os.environ.get("LLM_BASE_URL", f"{SV_BASE}/v1")
LLM_MODEL      = os.environ.get("LLM_MODEL_NAME", "openai/gpt-oss-20b")
LLM_API_KEY    = os.environ.get("LLM_API_KEY", "sv-openai-api-key")

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

results = {}

def run(label, fn):
    try:
        fn()
        results[label] = "OK"
    except Exception as e:
        print(f"  [FAIL] {e}")
        results[label] = f"FAIL: {e}"


# ── 1. Text embedding ─────────────────────────────────────────────────────────
print("\n--- Text Embedding (all-MiniLM-L6-v2) ---")
def t1():
    r = requests.post(EMBED_TEXT_URL, json={
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "input": ["Hello world", "Stock market analysis"],
    })
    r.raise_for_status()
    emb = r.json()["data"][0]["embedding"]
    print(f"  dim={len(emb)}, first5={emb[:5]}")
run("text-embed", t1)


# ── 2. Image + text embedding (SigLIP) ───────────────────────────────────────
print("\n--- Image+Text Embedding (siglip2-base-patch16-224) ---")
def t2():
    r = requests.post(EMBED_IMAGE_URL, json={
        "model": "google/siglip2-base-patch16-224",
        "input_type": "auto",
        "input": ["financial chart analysis"],
    })
    r.raise_for_status()
    emb = r.json()["data"][0]["embedding"]
    print(f"  dim={len(emb)}, first5={emb[:5]}")
run("image-embed", t2)


# ── 3. Main LLM chat (gpt-oss-20b via .env) ──────────────────────────────────
print(f"\n--- LLM Chat: {LLM_MODEL} @ {LLM_BASE_URL} ---")
def t3():
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": "What is machine learning? One sentence."}],
        max_tokens=80,
    )
    print(f"  {resp.choices[0].message.content.strip()}")
run("main-llm", t3)


# ── 4. Reasoning chat (DeepSeek-R1-Distill-Qwen-7B) ─────────────────────────
print("\n--- Reasoning Chat: DeepSeek-R1-Distill-Qwen-7B ---")
def t4():
    resp = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        messages=[{"role": "user", "content": "What is 17 * 23? One sentence."}],
        max_tokens=512,
    )
    print(f"  {resp.choices[0].message.content.strip()}")
run("deepseek-r1", t4)


# ── 5. Qdrant vector DB ──────────────────────────────────────────────────────
print("\n--- Qdrant Vector DB ---")
def t5():
    r = requests.get(f"{QDRANT_URL}/collections", timeout=5)
    r.raise_for_status()
    names = [c["name"] for c in r.json()["result"]["collections"]]
    print(f"  {len(names)} collections: {names[:5]}{'...' if len(names) > 5 else ''}")
run("qdrant", t5)


# ── 6. Zep memory graph ──────────────────────────────────────────────────────
print("\n--- Zep Memory Graph ---")
def t6():
    if not ZEP_API_KEY:
        raise RuntimeError("ZEP_API_KEY not set in .env")
    from zep_cloud.client import Zep
    zep = Zep(api_key=ZEP_API_KEY)
    # lightweight check: list graphs (returns quickly even if empty)
    graphs = zep.graph.search(user_id="__test__", query="test", limit=1)
    print(f"  Zep connected OK (user_id=__test__ search returned)")
run("zep", t6)


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n--- Summary ---")
for name, status in results.items():
    print(f"  {name:20s} {status}")