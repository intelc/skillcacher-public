import os
from skillcacher.settings import Settings


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("SKILLCACHER_BACKEND_URL", raising=False)
    s = Settings()
    assert s.backend_url == "http://localhost:8000"
    assert s.proxy_port == 4000
    assert s.trace_dir.name == "traces"


def test_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SKILLCACHER_BACKEND_URL", "http://gpu.tail-net:8000")
    monkeypatch.setenv("SKILLCACHER_PROXY_PORT", "5000")
    monkeypatch.setenv("SKILLCACHER_TRACE_DIR", str(tmp_path))
    s = Settings()
    assert s.backend_url == "http://gpu.tail-net:8000"
    assert s.proxy_port == 5000
    assert s.trace_dir == tmp_path
