import backoff
import os
from typing import Tuple
import requests
import json
import anthropic
import openai
from dotenv import load_dotenv

load_dotenv()

MAX_TOKENS = 16384

CLAUDE_MODEL = "anthropic/claude-sonnet-4-5-20250929"
CLAUDE_HAIKU_MODEL = "anthropic/claude-3-haiku-20240307"
CLAUDE_35NEW_MODEL = "anthropic/claude-3-5-sonnet-20241022"
OPENAI_MODEL = "openai/gpt-4o"
OPENAI_MINI_MODEL = "openai/gpt-4o-mini"
OPENAI_O3_MODEL = "openai/o3"
OPENAI_O3MINI_MODEL = "openai/o3-mini"
OPENAI_O4MINI_MODEL = "openai/o4-mini"
OPENAI_GPT52_MODEL = "openai/gpt-5.2"
OPENAI_GPT5_MODEL = "openai/gpt-5"
OPENAI_GPT5MINI_MODEL = "openai/gpt-5-mini"
GEMINI_3_MODEL = "gemini/gemini-3-pro-preview"
GEMINI_MODEL = "gemini/gemini-2.5-pro"
GEMINI_FLASH_MODEL = "gemini/gemini-2.5-flash"

# Models that do not accept a temperature parameter
_NO_TEMPERATURE_MODELS = {"openai/gpt-5", "openai/gpt-5-mini"}

# Models that require max_completion_tokens instead of max_tokens
_MAX_COMPLETION_TOKENS_MODELS = {"openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5.2"}


def _parse_model(model: str) -> Tuple[str, str]:
    """Return (provider, model_name) from a 'provider/model' string."""
    if "/" in model:
        provider, name = model.split("/", 1)
        return provider, name
    return "openai", model


def _call_anthropic(model_name: str, messages: list, temperature: float, max_tokens: int) -> str:
    client = anthropic.Anthropic()

    # Pull out any leading system message
    system = None
    user_messages = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            user_messages.append(m)

    # claude-3-haiku has a 4096-token output cap
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

    kwargs: dict = {
        "model": model_name,
        "messages": messages,
    }

    # GPT-5 and GPT-5-mini only support default temperature
    if model not in _NO_TEMPERATURE_MODELS:
        kwargs["temperature"] = temperature

    # GPT-5 family uses max_completion_tokens
    if model in _MAX_COMPLETION_TOKENS_MODELS:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def _call_gemini(model_name: str, messages: list, temperature: float, max_tokens: int) -> str:
    # Use Google's OpenAI-compatible endpoint so we don't need a separate SDK
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


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

    # Convert text → content for internal API compatibility
    msg_history = [
        {**m, "content": m.pop("text")} if "text" in m else m
        for m in msg_history
    ]

    new_msg_history = msg_history + [{"role": "user", "content": msg}]

    provider, model_name = _parse_model(model)

    if provider == "anthropic":
        response_text = _call_anthropic(model_name, new_msg_history, temperature, max_tokens)
    elif provider == "gemini":
        response_text = _call_gemini(model_name, new_msg_history, temperature, max_tokens)
    else:  # openai (default)
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
        ("CLAUDE_MODEL", CLAUDE_MODEL),
        ("CLAUDE_HAIKU_MODEL", CLAUDE_HAIKU_MODEL),
        ("CLAUDE_35NEW_MODEL", CLAUDE_35NEW_MODEL),
        ("OPENAI_MODEL", OPENAI_MODEL),
        ("OPENAI_MINI_MODEL", OPENAI_MINI_MODEL),
        ("OPENAI_O3_MODEL", OPENAI_O3_MODEL),
        ("OPENAI_O3MINI_MODEL", OPENAI_O3MINI_MODEL),
        ("OPENAI_O4MINI_MODEL", OPENAI_O4MINI_MODEL),
        ("OPENAI_GPT52_MODEL", OPENAI_GPT52_MODEL),
        ("OPENAI_GPT5_MODEL", OPENAI_GPT5_MODEL),
        ("OPENAI_GPT5MINI_MODEL", OPENAI_GPT5MINI_MODEL),
        ("GEMINI_3_MODEL", GEMINI_3_MODEL),
        ("GEMINI_MODEL", GEMINI_MODEL),
        ("GEMINI_FLASH_MODEL", GEMINI_FLASH_MODEL),
    ]
    for name, model in models:
        print(f"\n{'='*50}")
        print(f"Testing {name}: {model}")
        print('='*50)
        try:
            output_msg, msg_history, info = get_response_from_llm(msg, model=model)
            print(f"OK: {output_msg[:100]}...")
        except Exception as e:
            print(f"FAIL: {str(e)[:200]}")
