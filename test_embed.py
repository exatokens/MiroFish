import requests

url = "http://10.0.10.51:8123/embed-text/v1/embeddings" 

payload = {
    "model": "nomic-ai/nomic-embed-text-v1.5",
    "input": [
        "Dolphins are mammals."
    ],
    "return_token_embeddings": False
}

response = requests.post(url, json=payload, timeout=30)
response.raise_for_status()  # raises if 4xx/5xx

data = response.json()
print(data)