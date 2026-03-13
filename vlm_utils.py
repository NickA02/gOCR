"""
Utility functions for querying a local VLM via an OpenAI-compatible API
(e.g. LMStudio) and parsing concept annotations from responses.
"""

import base64
import json
import re
import ast
from pathlib import Path
from typing import List, Optional
import config

from openai import OpenAI


# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------

def get_client(base_url: str = config.API_ROOT, api_key: str = config.API_KEY) -> OpenAI:
    """Return an OpenAI client pointed at a local LMStudio server."""
    return OpenAI(base_url=base_url, api_key=api_key)


def encode_image_base64(image_path: Path) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ---------------------------------------------------------------------------
# API call wrappers
# ---------------------------------------------------------------------------

def query_vlm(
    client: OpenAI,
    model: str,
    query: str,
    chat_history: Optional[List[dict]] = None,
    image_path: Optional[Path] = None,
    system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.3,
) -> List[dict]:
    """
    Send a prompt (optionally with an image) to the VLM and return the
    assistant's text response.

    Parameters
    ----------
    client : OpenAI
        An OpenAI-compatible client (e.g. from ``get_client``).
    model : str
        The model identifier served by LMStudio.
    query : str
        The user-visible text query.
    chat_history : List[dict], optional
        A list of previous messages in the conversation. Each message should be a dict with "role" and "content" keys.
    image_path : Path, optional
        If provided, the image is base64-encoded and sent as part of the
        multimodal message.
    system_prompt : str, optional
        Optional system message prepended to the conversation.
    max_tokens : int
        Maximum number of tokens for the response.
    temperature : float
        Sampling temperature (lower = more deterministic).

    Returns
    -------
    List[dict]
        A list of messages in the conversation, including the user's query and the model's response.
    """
    if chat_history is None:
        messages = []
    else:
        messages = chat_history[:]

    if system_prompt and not any(m["role"] == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": system_prompt})

    # Build user message content
    user_content: list = []
    if image_path is not None:
        b64 = encode_image_base64(Path(image_path))
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )
    user_content.append({"type": "text", "text": query})
    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    content = response.choices[0].message.content
    return messages + [{"role": "assistant", "content": content}]

