"""Tests for the dependency-free local browser interface."""

from api.local_server import load_chat_page


def test_chat_page_is_packaged():
    page = load_chat_page().decode("utf-8")
    assert "Chat with BB8" in page
    assert 'fetch("/chat"' in page
    assert "x-api-key" in page


def test_chat_model_selection_and_metrics_are_explicit():
    page = load_chat_page().decode("utf-8")
    assert 'id="chatModel"' in page
    assert 'fetch("/chat/models"' in page
    assert 'model: chatModelElement.value' in page
    assert 'chatModelElement.addEventListener("change"' in page
    assert 'chatModelElement.disabled = true' in page
    assert 'result.generated_tokens' in page
    assert 'result.stop_reason' in page
    assert 'external baseline' in page


def test_lab_elements_match_script_references():
    import re
    from html.parser import HTMLParser
    from pathlib import Path

    class Elements(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids = []
        def handle_starttag(self, tag, attrs):
            self.ids.extend(value for key, value in attrs if key == "id")

    directory = Path(__file__).resolve().parents[1] / "api"
    page = (directory / "lab.html").read_text(encoding="utf-8")
    script = (directory / "lab.js").read_text(encoding="utf-8")
    elements = Elements()
    elements.feed(page)
    assert len(elements.ids) == len(set(elements.ids)), "Duplicate lab element IDs"
    referenced = set(re.findall(r'\$\("([a-zA-Z]+)"\)', script))
    assert referenced <= set(elements.ids)
    assert 'src="/lab.js"' in page
    assert 'href="/lab.css"' in page
    assert 'href="/lab"' in load_chat_page().decode("utf-8")
