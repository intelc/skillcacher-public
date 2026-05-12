"""ConditionLifecycle spawns the proxy as a subprocess via
`.venv/bin/skillcacher-proxy`, which means the parent's logging.basicConfig
(set in bench/cli.py) doesn't apply. Without an explicit basicConfig call
in proxy/server.py:main(), info-level lines like
`pre-seeding %d entries from prefix index` never reach proxy.log under
default Python logging (root logger defaults to WARNING).

This test verifies main() calls logging.basicConfig at INFO level before
uvicorn.run, so spawned-subprocess info logs land in the captured stdout
file."""
from unittest.mock import patch, MagicMock


def test_proxy_main_configures_info_level_logging():
    """main() must call logging.basicConfig with level=INFO before starting
    uvicorn — otherwise info logs don't reach the proxy.log file in the
    spawned-subprocess context."""
    with patch("skillcacher.proxy.server.uvicorn.run") as fake_uvicorn, \
         patch("skillcacher.proxy.server.build_app", return_value=MagicMock()), \
         patch("skillcacher.proxy.server.Settings", return_value=MagicMock(proxy_host="127.0.0.1", proxy_port=4000)), \
         patch("skillcacher.proxy.server.logging.basicConfig") as fake_basic_config:
        from skillcacher.proxy.server import main
        main()
    assert fake_basic_config.called, "main() must call logging.basicConfig"
    kwargs = fake_basic_config.call_args.kwargs
    import logging as _logging
    assert kwargs.get("level") == _logging.INFO, \
        f"basicConfig must be called at INFO level, got {kwargs.get('level')}"
    assert fake_uvicorn.called, "main() must invoke uvicorn.run"


def test_proxy_main_basic_config_called_before_uvicorn():
    """Order matters: basicConfig before uvicorn.run, otherwise the very
    first uvicorn-internal log line slips through under WARNING level."""
    call_order: list[str] = []
    with patch("skillcacher.proxy.server.uvicorn.run", side_effect=lambda *a, **kw: call_order.append("uvicorn")), \
         patch("skillcacher.proxy.server.build_app", return_value=MagicMock()), \
         patch("skillcacher.proxy.server.Settings", return_value=MagicMock(proxy_host="127.0.0.1", proxy_port=4000)), \
         patch("skillcacher.proxy.server.logging.basicConfig", side_effect=lambda *a, **kw: call_order.append("basicConfig")):
        from skillcacher.proxy.server import main
        main()
    assert call_order == ["basicConfig", "uvicorn"], call_order
