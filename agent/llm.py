import backoff
import os
from typing import Tuple
import requests
import json
import anthropic
import openai
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Token limits
# ---------------------------------------------------------------------------

MAX_TOKENS = 16384   # default for large models
SLM_MAX_TOKENS = 4096  # safe upper bound for small language models

# ---------------------------------------------------------------------------
# Model constants — Large Language Models
# ---------------------------------------------------------------------------

# Anthropic
CLAUDE_MODEL        = "anthropic/claude-sonnet-4-5-20250929"
CLAUDE_HAIKU_MODEL  = "anthropic/claude-3-haiku-20240307"
CLAUDE_35NEW_MODEL  = "anthropic/claude-3-5-sonnet-20241022"

# OpenAI
OPENAI_MODEL        = "openai/gpt-4o"
OPENAI_MINI_MODEL   = "openai/gpt-4o-mini"
OPENAI_O3_MODEL     = "openai/o3"
OPENAI_O3MINI_MODEL = "openai/o3-mini"
OPENAI_O4MINI_MODEL = "openai/o4-mini"
OPENAI_GPT52_MODEL  = "openai/gpt-5.2"
OPENAI_GPT5_MODEL   = "openai/gpt-5"
OPENAI_GPT5MINI_MODEL = "openai/gpt-5-mini"

# Google Gemini (via OpenAI-compatible endpoint)
GEMINI_3_MODEL      = "gemini/gemini-3-pro-preview"
GEMINI_MODEL        = "gemini/gemini-2.5-pro"
GEMINI_FLASH_MODEL  = "gemini/gemini-2.5-flash"

# NVIDIA NIM — large
NVIDIA_LLAMA_MODEL    = "nvidia/meta/llama-3.3-70b-instruct"
NVIDIA_NEMOTRON_MODEL = "nvidia/nvidia/llama-3.1-nemotron-70b-instruct"
NVIDIA_MISTRAL_MODEL  = "nvidia/mistralai/mistral-large"

# Ollama — large
OLLAMA_LLAMA_MODEL   = "ollama/llama3.2"
OLLAMA_MISTRAL_MODEL = "ollama/mistral"
OLLAMA_GEMMA_MODEL   = "ollama/gemma3"
OLLAMA_QWEN_MODEL    = "ollama/qwen2.5"

# OpenRouter — large
OPENROUTER_LLAMA_MODEL   = "openrouter/meta-llama/llama-3.3-70b-instruct"
OPENROUTER_MISTRAL_MODEL = "openrouter/mistralai/mistral-large"
OPENROUTER_CLAUDE_MODEL  = "openrouter/anthropic/claude-3.5-sonnet"
OPENROUTER_GEMINI_MODEL  = "openrouter/google/gemini-2.5-pro"

# Groq — large
GROQ_LLAMA_MODEL   = "groq/llama-3.3-70b-versatile"
GROQ_LLAMA3_MODEL  = "groq/llama3-70b-8192"
GROQ_MIXTRAL_MODEL = "groq/mixtral-8x7b-32768"
GROQ_GEMMA_MODEL   = "groq/gemma2-9b-it"

# ---------------------------------------------------------------------------
# Model constants — Small Language Models (SLM, ≤ ~7 B parameters)
# ---------------------------------------------------------------------------

# Ollama SLM (local / cloud)
OLLAMA_SLM_PHI3_MODEL      = "ollama/phi3"            # Phi-3 Mini 3.8 B
OLLAMA_SLM_PHI35_MODEL     = "ollama/phi3.5"          # Phi-3.5 Mini 3.8 B
OLLAMA_SLM_PHI4MINI_MODEL  = "ollama/phi4-mini"       # Phi-4 Mini
OLLAMA_SLM_GEMMA2B_MODEL   = "ollama/gemma2:2b"       # Gemma 2 2 B
OLLAMA_SLM_LLAMA1B_MODEL   = "ollama/llama3.2:1b"     # Llama 3.2 1 B
OLLAMA_SLM_LLAMA3B_MODEL   = "ollama/llama3.2:3b"     # Llama 3.2 3 B
OLLAMA_SLM_QWEN05B_MODEL   = "ollama/qwen2.5:0.5b"    # Qwen 2.5 0.5 B
OLLAMA_SLM_QWEN15B_MODEL   = "ollama/qwen2.5:1.5b"    # Qwen 2.5 1.5 B
OLLAMA_SLM_QWEN3B_MODEL    = "ollama/qwen2.5:3b"      # Qwen 2.5 3 B
OLLAMA_SLM_SMOLLM2_MODEL   = "ollama/smollm2"         # SmolLM2 1.7 B

# OpenRouter SLM
OPENROUTER_SLM_PHI3_MODEL  = "openrouter/microsoft/phi-3-mini-128k-instruct"
OPENROUTER_SLM_PHI35_MODEL = "openrouter/microsoft/phi-3.5-mini-instruct"
OPENROUTER_SLM_GEMMA2B_MODEL = "openrouter/google/gemma-2-2b-it"

# Groq SLM
GROQ_SLM_LLAMA1B_MODEL = "groq/llama-3.2-1b-preview"
GROQ_SLM_LLAMA3B_MODEL = "groq/llama-3.2-3b-preview"

# NVIDIA NIM SLM
NVIDIA_SLM_PHI3_MODEL  = "nvidia/microsoft/phi-3-mini-128k-instruct"
NVIDIA_SLM_GEMMA2B_MODEL = "nvidia/google/gemma-2-2b-it"

# ---------------------------------------------------------------------------
# SLM detection
# ---------------------------------------------------------------------------

# Substrings that identify a model as an SLM (case-insensitive match)
_SLM_PATTERNS = {
    "phi3", "phi-3", "phi3.5", "phi-3.5", "phi4-mini", "phi-4-mini",
    "gemma2:2b", "gemma-2-2b", "gemma2-2b",
    "llama3.2:1b", "llama3.2:3b", "llama-3.2-1b", "llama-3.2-3b",
    "qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b",
    "qwen-2.5-0.5b", "qwen-2.5-1.5b", "qwen-2.5-3b",
    "smollm",
}


def is_slm(model: str) -> bool:
    """Return True if *model* is recognised as a Small Language Model."""
    m = model.lower()
    return any(p in m for p in _SLM_PATTERNS)

# ---------------------------------------------------------------------------
# OpenAI model-specific quirks
# ---------------------------------------------------------------------------

# Models that do not accept a temperature parameter
_NO_TEMPERATURE_MODELS = {"openai/gpt-5", "openai/gpt-5-mini"}

# Models that require max_completion_tokens instead of max_tokens
_MAX_COMPLETION_TOKENS_MODELS = {"openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5.2"}


# ---------------------------------------------------------------------------
# Model-string parsing
# ---------------------------------------------------------------------------

def _parse_model(model: str) -> Tuple[str, str]:
    """Return (provider, model_name) from a 'provider/model' string.

    Splits on the first '/' only so model names that contain slashes
    (e.g. OpenRouter's 'meta-llama/llama-3.3-70b') are preserved intact.
    """
    if "/" in model:
        provider, name = model.split("/", 1)
        return provider, name
    return "openai", model


# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------

def _call_anthropic(model_name: str, messages: list, temperature: float, max_tokens: int) -> str:
    client = anthropic.Anthropic()

    system = None
    user_messages = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            user_messages.append(m)

    if "claude-3-haiku" in model_name:
        max_tokens = min(max_tokens, 4096)

    kwargs = {
        "model": model_name,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": user_messages,
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    return response.content[0].text


def _call_openai(model: str, model_name: str, messages: list, temperature: float, max_tokens: int) -> str:
    client = openai.OpenAI()

    kwargs: dict = {"model": model_name, "messages": messages}

    if model not in _NO_TEMPERATURE_MODELS:
        kwargs["temperature"] = temperature

    if model in _MAX_COMPLETION_TOKENS_MODELS:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def _call_openai_compat(
    base_url: str,
    api_key: str,
    model_name: str,
    messages: list,
    temperature: float,
    max_tokens: int,
    extra_headers: dict | None = None,
) -> str:
    """Generic caller for any OpenAI-compatible endpoint."""
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    kwargs: dict = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_headers:
        kwargs["extra_headers"] = extra_headers
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def _call_gemini(model_name: str, messages: list, temperature: float, max_tokens: int) -> str:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    return _call_openai_compat(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=api_key,
        model_name=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _call_nvidia(model_name: str, messages: list, temperature: float, max_tokens: int) -> str:
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    return _call_openai_compat(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        model_name=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _call_ollama(model_name: str, messages: list, temperature: float, max_tokens: int) -> str:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    api_key = os.environ.get("OLLAMA_API_KEY", "ollama")  # cloud: real key, local: any string
    return _call_openai_compat(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _call_openrouter(model_name: str, messages: list, temperature: float, max_tokens: int) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    return _call_openai_compat(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model_name=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_headers={"X-Title": "HyperAgents"},
    )


def _call_groq(model_name: str, messages: list, temperature: float, max_tokens: int) -> str:
    api_key = os.environ.get("GROQ_API_KEY", "")
    return _call_openai_compat(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
        model_name=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

_PROVIDER_MAP = {
    "anthropic":  lambda mn, msgs, t, mt: _call_anthropic(mn, msgs, t, mt),
    "gemini":     lambda mn, msgs, t, mt: _call_gemini(mn, msgs, t, mt),
    "nvidia":     lambda mn, msgs, t, mt: _call_nvidia(mn, msgs, t, mt),
    "ollama":     lambda mn, msgs, t, mt: _call_ollama(mn, msgs, t, mt),
    "openrouter": lambda mn, msgs, t, mt: _call_openrouter(mn, msgs, t, mt),
    "groq":       lambda mn, msgs, t, mt: _call_groq(mn, msgs, t, mt),
}


@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, json.JSONDecodeError, KeyError),
    max_time=600,
    max_value=60,
)
def get_response_from_llm(
    msg: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
    msg_history=None,
) -> Tuple[str, list, dict]:
    if msg_history is None:
        msg_history = []

    # SLMs cannot reliably produce more than SLM_MAX_TOKENS tokens
    if is_slm(model):
        max_tokens = min(max_tokens, SLM_MAX_TOKENS)

    # Convert text → content for internal API compatibility
    msg_history = [
        {**m, "content": m.pop("text")} if "text" in m else m
        for m in msg_history
    ]

    new_msg_history = msg_history + [{"role": "user", "content": msg}]

    provider, model_name = _parse_model(model)

    if provider in _PROVIDER_MAP:
        response_text = _PROVIDER_MAP[provider](model_name, new_msg_history, temperature, max_tokens)
    else:
        # Default: treat as OpenAI-compatible (covers plain "openai" prefix)
        response_text = _call_openai(model, model_name, new_msg_history, temperature, max_tokens)

    new_msg_history.append({"role": "assistant", "content": response_text})

    # Convert content → text for MetaGen API compatibility
    new_msg_history = [
        {**m, "text": m.pop("content")} if "content" in m else m
        for m in new_msg_history
    ]

    return response_text, new_msg_history, {}


if __name__ == "__main__":
    msg = 'Hello there!'
    models = [
        ("CLAUDE_MODEL",             CLAUDE_MODEL),
        ("CLAUDE_HAIKU_MODEL",       CLAUDE_HAIKU_MODEL),
        ("OPENAI_MODEL",             OPENAI_MODEL),
        ("OPENAI_O3_MODEL",          OPENAI_O3_MODEL),
        ("GEMINI_MODEL",             GEMINI_MODEL),
        ("NVIDIA_LLAMA_MODEL",       NVIDIA_LLAMA_MODEL),
        ("OLLAMA_LLAMA_MODEL",       OLLAMA_LLAMA_MODEL),
        ("OPENROUTER_LLAMA_MODEL",   OPENROUTER_LLAMA_MODEL),
        ("GROQ_LLAMA_MODEL",         GROQ_LLAMA_MODEL),
        # SLMs
        ("OLLAMA_SLM_PHI3_MODEL",    OLLAMA_SLM_PHI3_MODEL),
        ("OLLAMA_SLM_LLAMA3B_MODEL", OLLAMA_SLM_LLAMA3B_MODEL),
        ("GROQ_SLM_LLAMA3B_MODEL",   GROQ_SLM_LLAMA3B_MODEL),
    ]
    for name, model in models:
        slm_tag = " [SLM]" if is_slm(model) else ""
        print(f"\n{'='*50}")
        print(f"Testing {name}{slm_tag}: {model}")
        print('='*50)
        try:
            output_msg, msg_history, info = get_response_from_llm(msg, model=model)
            print(f"OK: {output_msg[:100]}...")
        except Exception as e:
            print(f"FAIL: {str(e)[:200]}")
