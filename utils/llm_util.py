import abc
import httpx
import inspect
import logging
import numpy as np
import os
import time
from pathlib import Path
from utils.io_util import load_json
from utils.metrics_util import (
    LLMRequestBudgetExceeded,
    current_trace,
)
from openai import OpenAI


openai_keys_folder = Path(__file__).resolve().parent.parent / "openai_keys"
OPENAI_KEY_FILE = openai_keys_folder / "openai_key.json"

# A hosted endpoint answers in seconds.  A locally served model does not: a 70B
# at 4-bit on a single GPU spends tens of seconds on a ~2k-token prefill plus a
# few hundred decoded tokens, so the hosted timeout would abort every request.
REMOTE_REQUEST_TIMEOUT_SECONDS = 60.0
LOCAL_REQUEST_TIMEOUT_SECONDS = 600.0

# A reasoning model spends its completion allowance on thinking (returned in a
# separate field) before it emits any answer.  A planning prompt can exhaust
# 1000 tokens on reasoning alone and return empty content, which then fails to
# parse as if the model could not follow instructions.
DEFAULT_MAX_TOKENS = 1000
REASONING_MAX_TOKENS = 4000

logger = logging.getLogger(__name__)


def _load_key_file() -> dict:
    if not OPENAI_KEY_FILE.exists():
        return {}
    try:
        return load_json(OPENAI_KEY_FILE) or {}
    except Exception as exc:
        logger.warning("Could not read %s: %s", OPENAI_KEY_FILE, exc)
        return {}


def _first_nonempty(*values):
    for value in values:
        if value is not None and str(value).strip() != "":
            return value
    return None


def configure_llm_runtime(
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    organization: str | None = None,
    response_format: str | None = None,
    timeout: float | None = None,
    max_tokens: int | None = None,
) -> None:
    """Set process-local LLM overrides used by all planner instances.

    Ollama and many local servers expose an OpenAI-compatible endpoint; setting
    FORESIGHT_LLM_BASE_URL to that endpoint is enough for this wrapper.
    """

    updates = {
        "FORESIGHT_LLM_MODEL": model,
        "FORESIGHT_LLM_BASE_URL": base_url,
        "FORESIGHT_LLM_API_KEY": api_key,
        "FORESIGHT_LLM_ORG": organization,
        "FORESIGHT_LLM_RESPONSE_FORMAT": response_format,
        "FORESIGHT_LLM_TIMEOUT": timeout,
        "FORESIGHT_LLM_MAX_TOKENS": max_tokens,
    }
    for key, value in updates.items():
        if value is not None:
            os.environ[key] = str(value)


def _is_local_base_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    lowered = base_url.lower()
    return any(token in lowered for token in ("localhost", "127.0.0.1", "0.0.0.0", ":11434", "ollama"))


def _response_format_enabled(base_url: str | None) -> bool:
    override = os.environ.get("FORESIGHT_LLM_RESPONSE_FORMAT")
    if override is not None:
        return override.strip().lower() not in {"0", "false", "no", "none", "off"}
    # Some OpenAI-compatible local servers reject response_format={"type":"text"}.
    return not _is_local_base_url(base_url)


JSON_MODE_REMINDER = "Return the answer as a single JSON object."


def _ensure_json_mentioned(messages):
    """Satisfy OpenAI's keyword guard on ``response_format=json_object``.

    The API rejects that response format (HTTP 400) unless the literal word
    "json" appears somewhere in the messages.  Some prompts give their schema
    only by example, so a reminder is appended when the word is absent;
    other prompts are passed through untouched.
    """
    if any("json" in str(message.get("content", "")).lower()
           for message in messages):
        return messages

    patched = [dict(message) for message in messages]
    for index in range(len(patched) - 1, -1, -1):
        if patched[index].get("role") == "user":
            content = str(patched[index].get("content", "")).rstrip()
            patched[index]["content"] = f"{content}\n\n{JSON_MODE_REMINDER}"
            return patched
    patched.append({"role": "user", "content": JSON_MODE_REMINDER})
    return patched


def _resolve_timeout(base_url: str | None) -> float:
    """Per-request timeout, defaulting by endpoint locality."""
    override = os.environ.get("FORESIGHT_LLM_TIMEOUT")
    if override is not None and str(override).strip() != "":
        try:
            value = float(override)
        except ValueError:
            logger.warning(
                "Ignoring non-numeric FORESIGHT_LLM_TIMEOUT=%r", override)
        else:
            if value > 0:
                return value
            logger.warning(
                "Ignoring non-positive FORESIGHT_LLM_TIMEOUT=%r", override)
    if _is_local_base_url(base_url):
        return LOCAL_REQUEST_TIMEOUT_SECONDS
    return REMOTE_REQUEST_TIMEOUT_SECONDS


def _resolve_max_tokens(default):
    """Completion allowance, overridable per model.

    Reasoning models need a far larger value than the default: the
    allowance covers thinking tokens too, and running out mid-thought yields an
    empty answer rather than a truncated one.
    """
    override = os.environ.get("FORESIGHT_LLM_MAX_TOKENS")
    if override is not None and str(override).strip() != "":
        try:
            value = int(float(override))
        except ValueError:
            logger.warning(
                "Ignoring non-numeric FORESIGHT_LLM_MAX_TOKENS=%r", override)
        else:
            if value > 0:
                return value
            logger.warning(
                "Ignoring non-positive FORESIGHT_LLM_MAX_TOKENS=%r", override)
    return default


def _load_llm_settings(
    engine: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    organization: str | None = None,
) -> dict:
    keys = _load_key_file()
    resolved_base_url = _first_nonempty(
        base_url,
        os.environ.get("FORESIGHT_LLM_BASE_URL"),
        os.environ.get("OPENAI_BASE_URL"),
        keys.get("base_url"),
    )
    resolved_api_key = _first_nonempty(
        api_key,
        os.environ.get("FORESIGHT_LLM_API_KEY"),
        os.environ.get("OPENAI_API_KEY"),
        keys.get("key"),
        "ollama" if _is_local_base_url(resolved_base_url) else None,
    )
    if not resolved_api_key:
        raise RuntimeError(
            "No LLM API key configured. Add openai_keys/openai_key.json, set "
            "OPENAI_API_KEY, or pass --llm-api-key ollama for a local Ollama endpoint."
        )

    resolved_org = _first_nonempty(
        organization,
        os.environ.get("FORESIGHT_LLM_ORG"),
        os.environ.get("OPENAI_ORG"),
        keys.get("org"),
    )
    resolved_proxy = _first_nonempty(os.environ.get("FORESIGHT_LLM_PROXY"), keys.get("proxy"))
    resolved_engine = _first_nonempty(engine, os.environ.get("FORESIGHT_LLM_MODEL"))

    return {
        "engine": resolved_engine,
        "api_key": str(resolved_api_key),
        "base_url": str(resolved_base_url) if resolved_base_url else None,
        "organization": str(resolved_org) if resolved_org else None,
        "proxy": str(resolved_proxy) if resolved_proxy else "",
        "use_response_format": _response_format_enabled(resolved_base_url),
        "timeout": _resolve_timeout(resolved_base_url),
    }


def connect_openai(
    client,
    engine,
    messages,
    temperature,
    max_tokens,
    top_p,
    frequency_penalty,
    presence_penalty,
    # stop,
    response_format,
    max_transport_attempts=5,
):
    """Issue a completion request with explicit, measurable retries.

    The OpenAI SDK client is created with its implicit retries disabled.  Each
    iteration below therefore corresponds to one actual SDK/HTTP request and
    is charged against the active trial's hard request budget.
    """
    request_kwargs = {
        "model": engine,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }
    if response_format is not None:
        request_kwargs["response_format"] = response_format

    logical_call_id = f"chat-{time.time_ns()}"
    last_error = None
    attempts = max(1, int(max_transport_attempts))
    for retry_index in range(attempts):
        trace = current_trace()
        trace_index = None
        if trace is not None:
            trace_index = trace.reserve_llm_request(
                messages=messages,
                request_parameters={
                    key: value
                    for key, value in request_kwargs.items()
                    if key != "messages"
                },
                logical_call_id=logical_call_id,
                retry_index=retry_index,
            )
        try:
            response = client.chat.completions.create(**request_kwargs)
        except LLMRequestBudgetExceeded:
            raise
        except Exception as exc:
            last_error = exc
            if trace is not None and trace_index is not None:
                trace.record_llm_error(trace_index, exc)
            if retry_index + 1 >= attempts:
                raise
            # Keep retry behaviour explicit and bounded.  The wait is included
            # in trial wall time and each retry consumes the same hard budget.
            time.sleep(min(2 ** retry_index, 8))
            continue

        if trace is not None and trace_index is not None:
            trace.record_llm_response(trace_index, response)
        return response

    # Defensive fallback; the loop either returns or raises above.
    raise last_error or RuntimeError("LLM request failed without an exception")


class GPT_Chat:
    def __init__(
        self,
        engine,
        base_url=None,
        api_key=None,
        organization=None,
        stop=None,
        max_tokens=1000,
        temperature=0,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    ):
        settings = _load_llm_settings(
            engine=engine,
            base_url=base_url,
            api_key=api_key,
            organization=organization,
        )
        self.engine = settings["engine"]
        self.max_tokens = _resolve_max_tokens(max_tokens)
        self.temperature = temperature
        self.top_p = top_p
        self.freq_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.use_response_format = settings["use_response_format"]
        self.base_url = settings["base_url"] or "OpenAI default endpoint"

        # add key
        self.timeout = settings["timeout"]
        client_kwargs = {
            "api_key": settings["api_key"],
            "timeout": self.timeout,
            # Retries are handled in connect_openai so every actual request is
            # observable and charged to the per-trial hard budget.
            "max_retries": 0,
        }
        if settings["base_url"]:
            client_kwargs["base_url"] = settings["base_url"]
        if settings["organization"]:
            client_kwargs["organization"] = settings["organization"]

        # Build the httpx client.  httpx renamed `proxies` to `proxy`, so pick
        # whichever this version accepts and fail loudly if neither exists.
        http_kwargs = {}
        proxy = settings["proxy"]
        if proxy:
            proxy_parameters = inspect.signature(
                httpx.Client.__init__).parameters
            if "proxy" in proxy_parameters:
                http_kwargs["proxy"] = proxy
            elif "proxies" in proxy_parameters:
                http_kwargs["proxies"] = proxy
            else:
                raise RuntimeError(
                    "A proxy is configured but this httpx version accepts "
                    "neither 'proxy' nor 'proxies'; remove the proxy setting "
                    "or pin a compatible httpx."
                )
        try:
            http_kwargs["transport"] = httpx.HTTPTransport(
                local_address="0.0.0.0")
        except Exception as exc:
            logger.debug("Custom httpx transport unavailable: %s", exc)
        try:
            client_kwargs["http_client"] = httpx.Client(**http_kwargs)
        except TypeError:
            if proxy:
                raise
            logger.debug("Falling back to the SDK default HTTP client")

        self.client = OpenAI(**client_kwargs)

    def get_response(
        self,
        prompt,
        messages=None,
        end_when_error=False,
        max_retry=5,
        temperature=0.0,
        force_json=False,
    ):
        conn_success, llm_output = False, ""
        if messages is not None:
            messages = messages
        else:
            messages = [{"role": "user", "content": prompt}]

        if force_json:
            response_format = {"type": "json_object"} if self.use_response_format else None
            if response_format is not None:
                messages = _ensure_json_mentioned(messages)
        else:
            response_format = {"type": "text"} if self.use_response_format else None

        try:
            logger.info(
                "[INFO] connecting to the LLM ... model=%s, base_url=%s",
                self.engine,
                self.base_url,
            )

            response = connect_openai(
                client=self.client,
                engine=self.engine,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.freq_penalty,
                presence_penalty=self.presence_penalty,
                response_format=response_format,
                max_transport_attempts=max_retry,
            )
            llm_output = response.choices[0].message.content
            conn_success = True
        except LLMRequestBudgetExceeded:
            raise
        except Exception as e:
            logger.info("[ERROR] LLM error: {}".format(e))
            if not end_when_error:
                logger.debug("LLM request exhausted its explicit retry budget")
        return conn_success, llm_output


class LLMBase(abc.ABC):
    def __init__(self, use_gpt_4: bool, *args, **kwargs):
        engine = _first_nonempty(
            kwargs.pop("model", None),
            kwargs.pop("engine", None),
            os.environ.get("FORESIGHT_LLM_MODEL"),
            "gpt-4o" if use_gpt_4 else "gpt-3.5-turbo",
        )
        self.llm_gpt = GPT_Chat(
            engine=engine,
            base_url=kwargs.pop("base_url", None),
            api_key=kwargs.pop("api_key", None),
            organization=kwargs.pop("organization", None),
        )

    def prompt_llm(self, prompt: str, temperature: float = 0.0, force_json: bool = False):
        # feed prompt to llm
        logger.info("\n" + "#" * 50)
        logger.info(f"Prompt:\n{prompt}")
        messages = [{"role": "user", "content": prompt}]

        conn_success, llm_output = self.llm_gpt.get_response(
            prompt=None,
            messages=messages,
            end_when_error=False,
            temperature=temperature,
            force_json=force_json,
        )
        if not conn_success:
            raise Exception("Fail to connect to the LLM")

        logger.info("\n" + "#" * 50)
        logger.info(f"LLM output:\n{llm_output}")

        return llm_output


####################### Convert Formats #####################


def textualize_array(array):
    return np.round(array, decimals=2).tolist()
