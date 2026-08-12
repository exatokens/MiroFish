import urllib.request, json

def test_chat(name, base_url, model):
    url = f"{base_url}/v1/chat/completions"
    data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 10
    }).encode()
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer sv-openai-api-key"
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            resp = json.loads(r.read())
            content = resp['choices'][0]['message']['content']
            print(f"[{name}] OK — {content.strip() if content else '(empty content)'}")
    except Exception as e:
        print(f"[{name}] FAIL — {e}")

def test_embed(name, base_url, model):
    url = f"{base_url}/embed-text/v1/embeddings"
    data = json.dumps({
        "model": model,
        "input": ["Hello world"]
    }).encode()
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer sv-openai-api-key"
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            resp = json.loads(r.read())
            vec = resp['data'][0]['embedding']
            print(f"[{name}] OK — embedding dim={len(vec)}")
    except Exception as e:
        print(f"[{name}] FAIL — {e}")

# Ray cluster — embed (port 8123)
test_embed("RAY embed  10.0.10.51:8123", "http://10.0.10.51:8123", "sentence-transformers/all-MiniLM-L6-v2")

# Ray cluster — chat (port 8124, vLLM direct)
test_chat("RAY chat   10.0.10.51:8124", "http://10.0.10.51:8124", "openai/gpt-oss-20b")


test_chat("INFER      10.0.10.66:8123", "http://10.0.10.66:8123", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")

# Inference API — direct vLLM (port 8123)
test_chat("INFER      10.0.10.66:8123", "http://10.0.10.66:8123", "Qwen/Qwen3.5-27B")
