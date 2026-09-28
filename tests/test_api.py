"""Unit tests for the AWS Lambda API contract."""

import json
from types import SimpleNamespace

import pytest

from api import handler


class FakeGenerator:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.model = SimpleNamespace(max_seq_len=128)

    def generate(self, **kwargs):
        if kwargs.get("return_full_text", True):
            return kwargs["prompt"] + " generated"
        return " generated"


class FakeTokenizer:
    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


@pytest.fixture(autouse=True)
def reset_handler(monkeypatch):
    handler._bundle = None
    monkeypatch.setenv("BB8_API_KEY", "test-secret-key")
    monkeypatch.setattr(
        handler,
        "get_bundle",
        lambda: SimpleNamespace(
            name="test-model",
            generator=FakeGenerator(),
            backend="bb8",
            prompt_style="bb8",
            max_context_tokens=128,
        ),
    )


def event(method, path, body=None, api_key=None):
    headers = {"x-api-key": api_key} if api_key else {}
    return {
        "version": "2.0",
        "rawPath": path,
        "headers": headers,
        "body": json.dumps(body) if body is not None else None,
        "requestContext": {"http": {"method": method}},
    }


def test_health_endpoint():
    response = handler.lambda_handler(event("GET", "/health"), None)
    payload = json.loads(response["body"])
    assert response["statusCode"] == 200
    assert payload["status"] == "ok"
    assert payload["model"] == "model"
    assert payload["model_loaded"] is False
    assert payload["device"] == "cpu"


def test_generate_requires_api_key():
    response = handler.lambda_handler(
        event("POST", "/generate", {"prompt": "ROMEO:"}), None
    )
    assert response["statusCode"] == 401


def test_generate_returns_text():
    response = handler.lambda_handler(
        event(
            "POST",
            "/generate",
            {"prompt": "ROMEO:", "max_new_tokens": 10},
            api_key="test-secret-key",
        ),
        None,
    )
    payload = json.loads(response["body"])
    assert response["statusCode"] == 200
    assert payload["model"] == "test-model"
    assert payload["text"] == "ROMEO: generated"


def test_generate_validates_request():
    response = handler.lambda_handler(
        event("POST", "/generate", {"prompt": ""}, api_key="test-secret-key"),
        None,
    )
    assert response["statusCode"] == 400


def test_chat_returns_assistant_reply():
    response = handler.lambda_handler(
        event(
            "POST",
            "/chat",
            {"messages": [{"role": "user", "content": "Hello"}]},
            api_key="test-secret-key",
        ),
        None,
    )
    payload = json.loads(response["body"])
    assert response["statusCode"] == 200
    assert payload["model"] == "test-model"
    assert payload["reply"] == "generated"
    assert payload["context_tokens"] <= payload["max_context_tokens"]


def test_chat_requires_last_user_message():
    response = handler.lambda_handler(
        event(
            "POST",
            "/chat",
            {"messages": [{"role": "assistant", "content": "Hello"}]},
            api_key="test-secret-key",
        ),
        None,
    )
    assert response["statusCode"] == 400
    assert "last message" in json.loads(response["body"])["error"]


def test_chat_requires_api_key():
    response = handler.lambda_handler(
        event("POST", "/chat", {"messages": [{"role": "user", "content": "Hi"}]}),
        None,
    )
    assert response["statusCode"] == 401
