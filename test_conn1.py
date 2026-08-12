import urllib.request, json

url = "http://10.0.10.66:8123/v1/chat/completions"
data = json.dumps({
    "model": "Qwen/Qwen3.5-27B",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "tools": [{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform math",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            }
        }
    }],
    "tool_choice": "auto",
    "max_tokens": 50
}).encode()

req = urllib.request.Request(url, data=data, headers={
    "Content-Type": "application/json",
    "Authorization": "Bearer sv-openai-api-key"
})
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        resp = json.loads(r.read())
        print("SUCCESS:", json.dumps(resp, indent=2))
except urllib.error.HTTPError as e:
    print("FAIL:", e.code, json.loads(e.read()))


