"""
Central LLM configuration.

All agents get their model from here, so the backend lives in ONE place. Three
providers are wired:

  * Ollama (LOCAL, default) — runs `qwen2.5-coder:7b` on the user's machine. No
    API key, no rate limits, works offline once the model is pulled. See the
    README for install/pull instructions.
  * Google Gemini — uses GOOGLE_API_KEY (from aistudio.google.com).
  * Groq — uses GROQ_API_KEY (fast; model ids prefixed "groq:").

The public API (`get_llm`, `available_models`, `get_model`, `set_model`) is
unchanged, so the rest of the app does not care which backend is active.
"""

import os

from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from dotenv import load_dotenv

load_dotenv()

# Cache identical LLM calls in-process. LangGraph re-runs a node from the top on
# resume, which would otherwise re-issue the same generation; with this, the
# replay hits the cache instead of calling the model again.
set_llm_cache(InMemoryCache())

# Selectable models (id -> label shown in the UI). The Ollama id doubles as the
# model name passed to Ollama.
AVAILABLE_MODELS = {
    "qwen2.5-coder:7b": "Qwen2.5-Coder 7B (local · Ollama · no key)",
    "gemini-2.0-flash": "Gemini 2.0 Flash (Google · needs key)",
    "groq:llama-3.3-70b-versatile": "Llama 3.3 70B (Groq · needs key)",
    "groq:openai/gpt-oss-120b": "GPT-OSS 120B (Groq · needs key)",
    "groq:openai/gpt-oss-20b": "GPT-OSS 20B (Groq · faster · needs key)",
}

DEFAULT_MODEL = "qwen2.5-coder:7b"

# Context window for local models (Ollama defaults are small; this app's prompts
# include feature stats + samples, so give it room).
OLLAMA_NUM_CTX = 8192

_ENV_KEY = "DATAPILOT_LLM_MODEL"


def available_models() -> dict:
    """Return {model_id: label} for the selectable models."""
    return dict(AVAILABLE_MODELS)


def get_model() -> str:
    """The currently selected model id."""
    return os.getenv(_ENV_KEY, DEFAULT_MODEL)


def set_model(model_id: str) -> None:
    """Select the model for subsequent LLM calls in this process."""
    if model_id:
        os.environ[_ENV_KEY] = model_id


def get_llm(temperature: float = 0.0, model: str | None = None):
    """
    Build a chat client for the current (or given) model, routing by provider.

    Ollama models honour ``temperature`` (used for deterministic-first /
    warmer-retry behaviour), as do Gemini and Groq.
    """
    key = model or get_model()

    if key.startswith("groq:"):
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY must be set to use a Groq model. "
                "Get a key at https://console.groq.com/keys."
            )
        return ChatGroq(
            model=key.split(":", 1)[1],
            api_key=api_key,
            temperature=temperature,
            max_tokens=8192,
            max_retries=4,
        )

    if key.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) must be set to use a Gemini model. "
                "Get a key at https://aistudio.google.com/apikey."
            )
        return ChatGoogleGenerativeAI(
            model=key,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=8192,
            max_retries=4,
        )

    # Default: a local Ollama model (the id is the Ollama model name).
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=key,
        temperature=temperature,
        num_ctx=OLLAMA_NUM_CTX,
    )
