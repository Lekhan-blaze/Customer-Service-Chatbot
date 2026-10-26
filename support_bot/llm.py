"""Shared Gemini LLM construction for chat, sentiment, and conversation analysis."""

from __future__ import annotations

import os
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")


def get_api_key() -> str | None:
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


def make_gemini_llm(
    *,
    temperature: float = 0.1,
    timeout: float = 45.0,
    max_retries: int = 1,
    api_key: str | None = None,
    model_name: str | None = None,
):
    """Chat Gemini client with low thinking so FAQ/support replies are not empty or stuck.

    gemini-3.8-flash defaults to medium thinking, which can delay or omit visible
    answer text. ``minimal`` is not supported on this model (API error).
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    key = api_key or get_api_key() or "fake_key_for_offline_testing"
    return ChatGoogleGenerativeAI(
        model=model_name or GEMINI_MODEL,
        google_api_key=key,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        thinking_level="low",
    )


def llm_text(result: Any) -> str:
    """Normalize LLM invoke output (string or chat message) to plain text."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif "text" in item:
                    parts.append(str(item.get("text") or ""))
        return "".join(parts)
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    return str(result)
