# SV Ray Cluster Access

Sample code to access SupportVectors-hosted APIs: the **SV Ray cluster** (embeddings and chat) and the **SV inference server** (vision-language chat with Qwen). Configuration is read from `config.yaml`; base URLs can be overridden via environment or your config.

---

## Overview

| Module | Service | Purpose |
|--------|---------|--------|
| `sv_ray_cluster_api` | SV Ray cluster | Text embeddings, image/text embeddings, and chat completions (GPT-OSS-20B) |
| `sv_inference_api`   | SV inference server (vLLM) | Chat completions with **Qwen2.5-VL-72B** (text + images) |

---

## Configuration

Base URLs are defined in `config.yaml`:

- **Ray cluster embeddings**: `ray_cluster_api.embed_base_url` (e.g. `http://10.0.10.51:8123`)
- **Ray cluster chat**: `ray_cluster_api.chat_base_url` (e.g. `http://10.0.10.51:8124`)
- **Inference server**: `inference_api.base_url` (e.g. `http://10.0.10.66:8123`)

Ensure `BOOTCAMP_ROOT_DIR` (or your project root) and `PYTHONPATH` are set so the package and config load correctly (see `.env.example`).

---

## 1. SV Ray Cluster API (`sv_ray_cluster_api`)

### 1.1 Text embeddings

**Endpoint:** `POST {embed_base_url}/embed-text/v1/embeddings`

**Function:** `embed_text(data_sentences, model_name, batch_size=50)`

- **Purpose:** Get vector embeddings for a list of text strings (e.g. for search or retrieval).
- **Parameters:**
  - `data_sentences`: list of strings to embed
  - `model_name`: embedding model (e.g. `"sentence-transformers/all-MiniLM-L6-v2"`)
  - `batch_size`: requests are sent in batches of this size (default 50)
- **Returns:** `list[list[float]]` — one embedding per input string.

**Example:**

```python
from ray_cluster_access.sv_ray_cluster_api import embed_text

texts = ["Hello, world!", "Another sentence."]
embeddings = embed_text(texts, model_name="sentence-transformers/all-MiniLM-L6-v2")
# embeddings[i] is the vector for texts[i]
```

---

### 1.2 Image / multimodal embeddings

**Endpoint:** `POST {embed_base_url}/embed-image/v1/embeddings`

**Function:** `embed_image(data_images, data_texts, model_name, batch_size=50)`

- **Purpose:** Embed images and/or text with a single multimodal model (e.g. CLIP, SigLIP) in one shared space.
- **Parameters:**
  - `data_images`: list of image sources — HTTP(S) URLs or local file paths
  - `data_texts`: list of text strings to embed (can be combined with images in one batch)
  - `model_name`: multimodal model (e.g. `"google/siglip2-base-patch16-224"`)
  - `batch_size`: batch size for API calls (default 50)
- **Returns:** `list[list[float]]` — one embedding per item (images first, then texts in order).

Images from URLs are fetched; local paths are read and sent as base64. Order of results matches the concatenated list `[image_1, ..., image_n, text_1, ..., text_m]`.

**Example:**

```python
from ray_cluster_access.sv_ray_cluster_api import embed_image

images = ["https://example.com/image.png", "path/to/local.jpg"]
texts = ["A short caption"]
embeddings = embed_image(images, texts, model_name="google/siglip2-base-patch16-224")
```

---

### 1.3 Chat completions (Ray cluster)

**Endpoint:** `{chat_base_url}/v1` (OpenAI-compatible chat API)

**Functions:**

- `get_sv_openai_client()` — returns an OpenAI client pointing at the Ray cluster; use for chat with **`openai/gpt-oss-20b`**.
- `sv_openai_completion(**kwargs)` — one-shot completion; model is fixed to `openai/gpt-oss-20b`; pass `messages` and any other OpenAI-style kwargs (e.g. `max_tokens`).
- `sv_completion(**kwargs)` — same idea via LiteLLM `completion()`; model, `api_base`, and `api_key` are set for the Ray cluster.

**Purpose:** Text-only chat with the LLM served on the Ray cluster (e.g. `openai/gpt-oss-20b`).

**Example:**

```python
from ray_cluster_access.sv_ray_cluster_api import sv_openai_completion

response = sv_openai_completion(
    messages=[{"role": "user", "content": "Why is the sky blue?"}]
)
print(response.choices[0].message.content)
```

---

## 2. SV Inference API (`sv_inference_api`) — Qwen2.5-VL

### 2.1 Chat completions (vision-language)

**Endpoint:** `{inference_base_url}/v1` (OpenAI-compatible chat API)

**Functions:**

- `get_sv_openai_client()` — returns an OpenAI client for the inference server; use for chat with **`Qwen/Qwen2.5-VL-72B-Instruct`**.
- `sv_openai_completion(**kwargs)` — completion with model fixed to `Qwen/Qwen2.5-VL-72B-Instruct`; supports **image + text** in `messages`.
- `sv_completion(**kwargs)` — same via LiteLLM; model and API base are set for the inference server.

**Purpose:** Multimodal chat (text + images) using the Qwen2.5-VL-72B model on the SV inference server (vLLM). Useful for image understanding, OCR, diagrams, etc.

**Example (image + text):**

```python
from ray_cluster_access.sv_inference_api import sv_openai_completion, _convert_image_to_base64

image_path = "./images/phys_img.jpeg"
image_b64 = _convert_image_to_base64(image_path)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Explain this image and put the explanation in a LaTeX tutorial."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            },
        ],
    }
]
response = sv_openai_completion(messages=messages, max_tokens=5000)
print(response.choices[0].message.content)
```

---

## Summary of endpoints

| Service | Endpoint path | Method | Purpose |
|---------|----------------|--------|--------|
| Ray cluster | `/embed-text/v1/embeddings` | POST | Text embeddings |
| Ray cluster | `/embed-image/v1/embeddings` | POST | Image + text embeddings (multimodal) |
| Ray cluster | `/v1` (chat) | OpenAI client | Chat with `openai/gpt-oss-20b` |
| Inference server | `/v1` (chat) | OpenAI client | Chat with `Qwen/Qwen2.5-VL-72B-Instruct` (text + images) |

---

## Running the included tests

From the project root (with `PYTHONPATH` including `src/`):

**Ray cluster:**

```bash
uv run python -m ray_cluster_access.sv_ray_cluster_api
```

This runs: `embed_text`, `embed_image`, OpenAI chat, and LiteLLM chat against the Ray cluster.

**Inference server (Qwen VL):**

```bash
uv run python -m ray_cluster_access.sv_inference_api
```

This runs: OpenAI and LiteLLM chat with an image + text message against the inference server.
