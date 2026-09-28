"""Minimal local HTTP server that exercises the Lambda handler."""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from api.handler import get_bundle, lambda_handler
from api import handler
from api.lab import LAB_LOCK, handle_lab


CHAT_PAGE = Path(__file__).with_name("chat.html")


def load_chat_page() -> bytes:
    """Load the dependency-free local chat interface."""

    return CHAT_PAGE.read_bytes()


class RequestHandler(BaseHTTPRequestHandler):
    def _handle(self) -> None:
        path = self.path.split("?", 1)[0]
        lab_assets = {
            "/lab": ("lab.html", "text/html"), "/lab/": ("lab.html", "text/html"),
            "/lab.js": ("lab.js", "text/javascript"), "/lab.css": ("lab.css", "text/css"),
            "/help.js": ("help.js", "text/javascript"), "/help.css": ("help.css", "text/css"),
        }
        if self.command == "GET" and path in lab_assets:
            filename, content_type = lab_assets[path]
            content = Path(__file__).with_name(filename).read_bytes()
            self.send_response(200)
            self.send_header("content-type", content_type + "; charset=utf-8")
            self.send_header("content-length", str(len(content)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(content)
            return
        if self.command == "GET" and path in {"/", "/chat", "/chat/"}:
            content = load_chat_page()
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(content)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(content)
            return

        content_length = int(self.headers.get("content-length", "0"))
        if content_length < 0 or content_length > 1_000_000:
            self.send_error(413, "Request too large")
            return
        body = self.rfile.read(content_length).decode("utf-8") if content_length else None
        event = {
            "version": "2.0",
            "rawPath": path,
            "headers": dict(self.headers),
            "body": body,
            "requestContext": {"http": {"method": self.command}},
        }
        with LAB_LOCK:
            response = handle_lab(event) if path.startswith("/lab/") else lambda_handler(event, None)
        self.send_response(response["statusCode"])
        for key, value in response.get("headers", {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(response.get("body", json.dumps({})).encode("utf-8"))

    do_GET = _handle
    do_POST = _handle
    do_OPTIONS = _handle

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BB8 API locally")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--chat-models", help="Optional JSON allowlist of local chat model aliases")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device; auto uses CUDA when available",
    )
    args = parser.parse_args()

    os.environ["BB8_MODEL_DIR"] = args.model_dir
    os.environ["BB8_API_KEY"] = args.api_key
    os.environ["BB8_DEVICE"] = args.device
    if args.chat_models:
        handler._chat_models = json.loads(Path(args.chat_models).read_text(encoding="utf-8"))

    # Local development should fail immediately for a bad artifact and should
    # not advertise readiness while the selected model is still unloaded.
    bundle = get_bundle()
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f"BB8 API listening at http://{args.host}:{args.port}")
    print(f"Model: {bundle.name} ({bundle.device})")
    print(f"Chat UI: http://{args.host}:{args.port}/chat")
    print(f"Model lab: http://{args.host}:{args.port}/lab")
    print("Endpoints: GET /health, POST /generate, POST /chat")
    server.serve_forever()


if __name__ == "__main__":
    main()
