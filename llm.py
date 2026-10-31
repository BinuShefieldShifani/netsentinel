"""
llm.py - One function to call whichever LLM is configured
=========================================================
Backends, chosen with the NETSENTINEL_LLM environment variable:

  ollama     (default) local or remote Ollama server
             OLLAMA_HOST, OLLAMA_MODEL
  anthropic  Anthropic Messages API, no local GPU needed
             ANTHROPIC_API_KEY, ANTHROPIC_MODEL
  none       no LLM; the pipeline uses its rule-based fallback

complete() never raises. It returns None when the LLM can't be reached or
returns nothing usable, and the caller decides what to do about that.
"""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional

from config import ANTHROPIC_MODEL, LLM_TIMEOUT_S, OLLAMA_HOST, OLLAMA_MODEL

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


def backend() -> str:
    return os.getenv("NETSENTINEL_LLM", "ollama").strip().lower()


def model_name() -> str:
    b = backend()
    if b == "anthropic":
        return os.getenv("ANTHROPIC_MODEL", ANTHROPIC_MODEL)
    if b == "ollama":
        return os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
    return "none"


def describe_backend() -> str:
    b = backend()
    return "LLM: off (rule-based fallback)" if b == "none" else f"LLM: {b} / {model_name()}"


def _ollama(prompt: str, system: str, max_tokens: int) -> Optional[str]:
    import ollama
    client = ollama.Client(host=os.getenv("OLLAMA_HOST", OLLAMA_HOST), timeout=LLM_TIMEOUT_S)
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    resp = client.chat(model=model_name(), messages=messages,
                       options={"temperature": 0.1, "num_predict": max_tokens})
    return resp["message"]["content"].strip()


def _anthropic(prompt: str, system: str, max_tokens: int) -> Optional[str]:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    body = {
        "model": model_name(),
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system
    req = urllib.request.Request(
        ANTHROPIC_URL, data=json.dumps(body).encode(), method="POST",
        headers={"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION,
                 "content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_S) as resp:
        data = json.loads(resp.read())
    text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
    return text.strip() or None


def complete(prompt: str, system: str = "", max_tokens: int = 500) -> Optional[str]:
    """Reply text from the configured LLM, or None if there is no usable reply."""
    b = backend()
    try:
        if b == "ollama":
            return _ollama(prompt, system, max_tokens)
        if b == "anthropic":
            return _anthropic(prompt, system, max_tokens)
        return None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, ImportError):
        return None
    except Exception:   # ollama raises its own error types; never let the LLM crash the pipeline
        return None


def timed_complete(prompt: str, system: str = "", max_tokens: int = 500) -> tuple[Optional[str], float]:
    start = time.perf_counter()
    text = complete(prompt, system, max_tokens)
    return text, time.perf_counter() - start
