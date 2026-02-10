"""OpenAI-compatible client for Moonshot/Kimi API (api.moonshot.cn/v1)."""

import os
import time
from typing import Dict, List, Optional

KIMI_BASE_URL = "https://api.moonshot.cn/v1"
REFLECT_MODEL = "moonshot-v1-128k"
CLASSIFY_MODEL = "moonshot-v1-8k"

RETRY_DELAYS = [5, 15, 45]  # seconds


def get_kimi_client():
    """Create an OpenAI client configured for Kimi API."""
    from openai import OpenAI
    api_key = os.environ.get("KIMI_API_KEY", "")
    if not api_key:
        raise ValueError("KIMI_API_KEY environment variable not set")
    return OpenAI(base_url=KIMI_BASE_URL, api_key=api_key)


def kimi_chat(
    messages: List[Dict[str, str]],
    model: str = REFLECT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    json_mode: bool = False,
    client=None,
) -> str:
    """Single chat completion with retry logic.

    Args:
        messages: List of {"role": ..., "content": ...} dicts
        model: Kimi model name
        temperature: Sampling temperature
        max_tokens: Max response tokens
        json_mode: If True, request JSON response format
        client: Optional pre-created OpenAI client

    Returns:
        Response text string
    """
    if client is None:
        client = get_kimi_client()

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_error = None
    for attempt, delay in enumerate(RETRY_DELAYS + [None]):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as e:
            last_error = e
            if delay is not None:
                print(f"  Kimi API attempt {attempt+1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)

    raise RuntimeError(f"Kimi API failed after {len(RETRY_DELAYS)+1} attempts: {last_error}")


def batch_kimi_chat(
    messages_list: List[List[Dict[str, str]]],
    model: str = REFLECT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    json_mode: bool = False,
) -> List[str]:
    """Batch chat completions (sequential, with retry per call).

    For simplicity, runs sequentially. Kimi API rate limits make
    async parallelism unreliable anyway.
    """
    client = get_kimi_client()
    results = []
    for messages in messages_list:
        try:
            result = kimi_chat(messages, model, temperature, max_tokens, json_mode, client)
            results.append(result)
        except Exception as e:
            print(f"  Batch Kimi call failed: {e}")
            results.append("")
    return results
