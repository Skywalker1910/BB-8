"""AWS Lambda handler for the BB8 inference API."""

import base64
import hmac
import json
import os
import time
from pathlib import Path
from typing import Any

from inference.conversation import build_chat_prompt, extract_assistant_reply, validate_messages
from inference.model_loader import ModelBundle, load_model_bundle


_bundle: ModelBundle | None = None
_chat_models: dict[str, dict] = {}
_chat_bundles: dict[str, ModelBundle] = {}
_STRATEGIES = {"greedy", "temperature", "top_k", "top_p"}


def get_bundle() -> ModelBundle:
    """Load the model once and reuse it for warm Lambda invocations."""

    global _bundle
    if _bundle is None:
        model_dir = os.getenv("BB8_MODEL_DIR", "deployment/model")
        configured_device = os.getenv("BB8_DEVICE", "cpu").lower()
        device = None if configured_device == "auto" else configured_device
        _bundle = load_model_bundle(model_dir, device=device)
    return _bundle


def chat_bundle(model: str | None) -> ModelBundle:
    """Only operator-configured aliases can select alternate models."""
    if model is None or model == "default":
        return get_bundle()
    if not isinstance(model, str) or model not in _chat_models:
        raise ValueError("Unknown chat model; use GET /chat/models")
    if model not in _chat_bundles:
        _chat_bundles[model] = load_model_bundle(
            _chat_models[model]["model_dir"], device=get_bundle().device,
        )
    return _chat_bundles[model]


def _health_payload() -> dict[str, Any]:
    """Describe both configured and already-loaded model state."""

    model_dir = os.getenv("BB8_MODEL_DIR", "deployment/model")
    payload: dict[str, Any] = {
        "status": "ok",
        "model": _bundle.name if _bundle is not None else Path(model_dir).name,
        "model_loaded": _bundle is not None,
        "device": _bundle.device if _bundle is not None else os.getenv("BB8_DEVICE", "cpu"),
    }
    if _bundle is not None:
        payload["backend"] = _bundle.backend
        payload["prompt_style"] = _bundle.prompt_style
        payload["max_context_tokens"] = _bundle.max_context_tokens
    return payload


def _response(status_code: int, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {
            "content-type": "application/json; charset=utf-8",
            "access-control-allow-origin": os.getenv("BB8_ALLOWED_ORIGIN", "*"),
            "access-control-allow-headers": "content-type,x-api-key",
            "access-control-allow-methods": "GET,POST,OPTIONS",
        },
        "body": json.dumps(payload, ensure_ascii=False),
    }


def _request_parts(event: dict[str, Any]) -> tuple[str, str, dict[str, str]]:
    request_context = event.get("requestContext", {})
    http = request_context.get("http", {})
    method = str(http.get("method") or event.get("httpMethod") or "GET").upper()
    path = str(event.get("rawPath") or event.get("path") or "/")
    headers = {str(k).lower(): str(v) for k, v in (event.get("headers") or {}).items()}
    return method, path.rstrip("/") or "/", headers


def _authorised(headers: dict[str, str]) -> bool:
    expected = os.getenv("BB8_API_KEY", "")
    if not expected:
        return True
    provided = headers.get("x-api-key", "")
    return hmac.compare_digest(provided, expected)


def _parse_body(event: dict[str, Any]) -> dict[str, Any]:
    body = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("Request body must be a JSON object")
    return parsed


def _generation_options(body: dict[str, Any]) -> dict[str, Any]:
    strategy = str(body.get("strategy", "top_p"))
    if strategy not in _STRATEGIES:
        raise ValueError(f"strategy must be one of: {', '.join(sorted(_STRATEGIES))}")

    max_allowed = int(os.getenv("BB8_MAX_NEW_TOKENS", "256"))
    max_new_tokens = int(body.get("max_new_tokens", 100))
    if max_new_tokens < 1 or max_new_tokens > max_allowed:
        raise ValueError(f"max_new_tokens must be between 1 and {max_allowed}")

    return {
        "max_new_tokens": max_new_tokens,
        "strategy": strategy,
        "temperature": float(body.get("temperature", 0.8)),
        "top_k": int(body.get("top_k", 40)),
        "top_p": float(body.get("top_p", 0.9)),
        "repetition_penalty": float(body.get("repetition_penalty", 1.1)),
    }


def _run_generation(
    bundle: ModelBundle,
    prompt: str,
    options: dict[str, Any],
    *,
    return_full_text: bool = True,
) -> tuple[str, float, dict]:
    started = time.perf_counter()
    generate = getattr(bundle.generator, "generate_with_details", bundle.generator.generate)
    result = generate(
        prompt=prompt,
        return_full_text=return_full_text,
        **options,
    )
    details = result if isinstance(result, dict) else {
        "text": result, "generated_tokens": None, "stop_reason": "unknown",
    }
    return details["text"], round((time.perf_counter() - started) * 1000, 2), details


def _generate(event: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    if not _authorised(headers):
        return _response(401, {"error": "Invalid or missing API key"})

    try:
        body = _parse_body(event)
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        max_prompt_chars = int(os.getenv("BB8_MAX_PROMPT_CHARS", "2000"))
        if len(prompt) > max_prompt_chars:
            raise ValueError(f"prompt exceeds the {max_prompt_chars}-character limit")

        options = _generation_options(body)
        bundle = get_bundle()
        text, elapsed_ms, details = _run_generation(bundle, prompt, options)
        return _response(
            200,
            {
                "model": bundle.name,
                "text": text,
                "generated_tokens": details["generated_tokens"],
                "stop_reason": details["stop_reason"],
                "latency_ms": elapsed_ms,
            },
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return _response(400, {"error": str(exc)})


def _chat(event: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    if not _authorised(headers):
        return _response(401, {"error": "Invalid or missing API key"})

    try:
        body = _parse_body(event)
        messages = validate_messages(body.get("messages"))
        max_messages = int(os.getenv("BB8_MAX_CHAT_MESSAGES", "50"))
        if len(messages) > max_messages:
            raise ValueError(f"messages exceeds the {max_messages}-message limit")

        max_chat_chars = int(os.getenv("BB8_MAX_CHAT_CHARS", "8000"))
        if sum(len(item["content"]) for item in messages) > max_chat_chars:
            raise ValueError(f"chat content exceeds the {max_chat_chars}-character limit")

        options = _generation_options(body)
        bundle = chat_bundle(body.get("model"))
        generator = bundle.generator
        max_context = bundle.max_context_tokens or generator.model.max_seq_len
        grounded = body.get("grounded", False)
        if not isinstance(grounded, bool):
            raise ValueError("grounded must be a boolean")
        sources = []
        if grounded:
            from grounded.service import prepare
            prompt, context_tokens, sources = prepare(bundle, messages)
        else:
            prompt, context_tokens = build_chat_prompt(
                messages, generator.tokenizer, max_context, bundle.prompt_style,
            )
        text, elapsed_ms, details = _run_generation(
            bundle,
            prompt,
            options,
            return_full_text=False,
        )
        reply = extract_assistant_reply(prompt, text)
        grounding = {}
        if grounded:
            from grounded.service import citation_status
            grounding = {"grounded": True, "corpus_version": "grounded-v1", "sources": sources,
                         "citation_checks": citation_status(reply, sources)}
        return _response(
            200,
            {
                "model": bundle.name,
                "reply": reply,
                "generated_tokens": details["generated_tokens"],
                "stop_reason": details["stop_reason"],
                "backend": bundle.backend,
                "context_tokens": context_tokens,
                "max_context_tokens": max_context,
                "latency_ms": elapsed_ms,
                **grounding,
            },
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return _response(400, {"error": str(exc)})


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Handle API Gateway HTTP API payloads."""

    method, path, headers = _request_parts(event)
    if method == "OPTIONS":
        return _response(204, {})
    if method == "GET" and path == "/health":
        return _response(200, _health_payload())
    if method == "GET" and path == "/chat/models":
        if not _authorised(headers):
            return _response(401, {"error": "Invalid or missing API key"})
        return _response(200, {"models": [
            {"id": "default", "label": get_bundle().name + " — configured model"},
            *[{"id": name, "label": spec["label"], "origin": spec.get("origin")}
              for name, spec in _chat_models.items()],
        ]})
    if method == "POST" and path == "/generate":
        return _generate(event, headers)
    if method == "POST" and path == "/chat":
        return _chat(event, headers)
    return _response(404, {"error": "Not found"})
