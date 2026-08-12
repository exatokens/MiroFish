#  ------------------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------------------

import base64
import os
from functools import wraps
from urllib.parse import urlparse, unquote
import requests
from ray_cluster_access import config
from openai import OpenAI
from litellm import completion

# base_url is the base URL of the SupportVectors inference machine on vLLM API which happens to be 10.0.10.66:8123 today.
base_url = config["inference_api"]["base_url"]
chat_base_url = f"{base_url}/v1"
_sv_openai_client = OpenAI(
        base_url=chat_base_url,
        api_key="sv-openai-api-key",
    )

def _is_http_url(image_source: str) -> bool:
    parsed = urlparse(image_source)
    return parsed.scheme in {"http", "https"}

def _convert_image_to_base64(image_source: str) -> str:
    """Convert an image from URL or local path to base64.

    Args:
        image_source (str): URL (http/https) or local file path.

    Returns:
        str: Base64-encoded image bytes (no data URI prefix).
    """
    if _is_http_url(image_source):
        response = requests.get(image_source, timeout=30)
        response.raise_for_status()
        image_bytes = response.content
    else:
        parsed = urlparse(image_source)
        if parsed.scheme == "file":
            file_path = unquote(parsed.path)
        else:
            file_path = image_source
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image not found at path: {file_path}")
        with open(file_path, "rb") as image_file:
            image_bytes = image_file.read()

    return base64.b64encode(image_bytes).decode("utf-8")

def get_sv_openai_client() -> OpenAI:
    """Create an OpenAI client for the LLM served on the SupportVectors inference machine on vLLM.
    Currently, the only supported model is "Qwen/Qwen2.5-VL-72B-Instruct".  So, only pass this model
    when using this client for chat completions.
    Returns:
        OpenAI: An OpenAI client for the LLM (Qwen/Qwen2.5-VL-72B-Instruct) served on the SupportVectors inference machine on vLLM.
    """
    return _sv_openai_client

@wraps(_sv_openai_client.chat.completions.create)
def sv_openai_completion(**kwargs):
    """
    Use the SV OpenAI completion function to get a response from the LLM served on the SupportVectors inference machine on vLLM.
    Args:
        **kwargs: Arbitrary keyword arguments
    Returns:
        dict: The response from the LLM
    """
    kwargs.pop("model", None)
    return _sv_openai_client.chat.completions.create(
        model= "Qwen/Qwen3.5-27B", #"Qwen/Qwen2.5-VL-72B-Instruct",
        **kwargs,
    )

# define an sv_completion function that uses the litellm completion but hard-codes the model, api_base, and api_key
# and leaves the rest for the caller to fill in.
@wraps(completion)
def sv_completion(**kwargs):
    """ 
    Use the litellm completion function to get a response from the LLM served on the SupportVectors inference machine on vLLM.
    Args:
        **kwargs: Arbitrary keyword arguments
    Returns:
        dict: The response from the LLM
    """
    kwargs.pop("model", None)
    kwargs.pop("api_base", None)
    kwargs.pop("api_key", None)
    return completion(
        model="openai/" + "Qwen/Qwen3.5-27B", #"Qwen/Qwen2.5-VL-72B-Instruct",
        api_base=chat_base_url,
        api_key="sv-openai-api-key",
        **kwargs,
    )

# Test functions to test the openai chat and litellm chat functions.
def test_openai_chat():
    """Test the openai chat function"""
    image_url = "./images/phys_img.jpeg"
    image_b64 = _convert_image_to_base64(image_url)
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": '''I am a physics student and came across this notes.  
                Can you explain it?  Put this explanation as a tutorial in a latex document.'''},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    },
                },
            ],
        }
    ]
    response = sv_openai_completion(messages=messages, max_tokens=5000)
    print(response.choices[0].message.content)

def test_litellm_chat():
    """Test the litellm chat function"""
    image_url = "./images/phys_img.jpeg"
    image_b64 = _convert_image_to_base64(image_url)
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": '''I am a physics student and came across this notes.  
                Can you explain it?  Put this explanation as a tutorial in a latex document.'''},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    },
                },
            ],
        }
    ]
    response = sv_completion(messages=messages, max_tokens=5000)
    print(response.choices[0].message.content)

if __name__ == "__main__":
    print("Testing openai chat function")
    test_openai_chat()
    print("--------------------------------")
    print("Testing litellm chat function")
    test_litellm_chat()
    print("--------------------------------")