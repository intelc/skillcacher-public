import json
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
import uvicorn


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@contextmanager
def _serve(app, port):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="off")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = 5.0
    while deadline > 0:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.05)
            deadline -= 0.05
    try:
        yield
    finally:
        server.should_exit = True
        t.join(timeout=2)


@pytest.mark.asyncio
async def test_proxy_against_echo_backend(tmp_path, monkeypatch):
    backend_port = _free_port()
    proxy_port = _free_port()
    monkeypatch.setenv("SKILLCACHER_BACKEND_URL", f"http://127.0.0.1:{backend_port}")
    monkeypatch.setenv("SKILLCACHER_PROXY_PORT", str(proxy_port))
    monkeypatch.setenv("SKILLCACHER_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("SKILLCACHER_TOKENIZER_NAME", "NousResearch/Meta-Llama-3-8B")
    monkeypatch.setenv("SKILLCACHER_SPAN_REGISTRY_PATH", str(tmp_path / "span_registry.sqlite"))
    monkeypatch.setenv("SKILLCACHER_SKILL_DIRS", "")
    monkeypatch.setenv("SKILLCACHER_ENABLE_PRE_SEED", "false")
    monkeypatch.setenv("SKILLCACHER_ENABLE_STDOUT_TAIL", "false")

    from skillcacher.dev.echo_backend import app as echo_app
    from skillcacher.proxy.server import build_app
    proxy_app = build_app()

    fixture = json.loads((Path(__file__).parent.parent / "fixtures" / "claude_code_simple.json").read_text())

    with _serve(echo_app, backend_port), _serve(proxy_app, proxy_port):
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(f"http://127.0.0.1:{proxy_port}/v1/messages", json=fixture)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["content"][0]["text"].startswith("echo:")
