"""Tests for scripts/synthetic_compaction.py — the harness Layer 3 trigger.

Default tests use the char-approximation path (no network, no HF download)."""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

from scripts.synthetic_compaction import (
    CHARS_PER_TOKEN_APPROX, DEFAULT_SIZE_TOKENS, FILLER_LINE, HEADER,
    build_stub_message, main,
)


def test_stub_includes_header():
    stub = build_stub_message(size_tokens=1000)
    assert stub.startswith(HEADER)


def test_stub_size_scales_with_target():
    """Larger target → larger output. The relationship is linear in the
    char-approximation path."""
    small = build_stub_message(size_tokens=500)
    large = build_stub_message(size_tokens=5000)
    assert len(large) > len(small) * 5  # roughly 10× target → at least 5× length


def test_stub_default_size_is_above_compact_threshold():
    """Default 250K tokens × 4 chars/token = 1M chars; well above the 200K
    autocompact threshold + slack."""
    stub = build_stub_message()
    # Char count >= target_tokens × CHARS_PER_TOKEN_APPROX
    expected_min_chars = DEFAULT_SIZE_TOKENS * CHARS_PER_TOKEN_APPROX
    assert len(stub) >= expected_min_chars


def test_stub_uses_filler_line_repeatedly():
    """The body after the header is FILLER_LINE repeated until target met."""
    stub = build_stub_message(size_tokens=2000)
    body = stub[len(HEADER):]
    # Body should be a multiple-or-near-multiple of FILLER_LINE.
    n_filler = body.count(FILLER_LINE)
    assert n_filler >= 1
    # And contain no other content beyond filler.
    assert body == FILLER_LINE * n_filler


def test_stub_is_deterministic():
    """Same size → same output, byte-for-byte. No randomness in the stub."""
    a = build_stub_message(size_tokens=3000)
    b = build_stub_message(size_tokens=3000)
    assert a == b


def test_main_writes_to_stdout(capsys):
    rc = main(["synthetic_compaction.py", "--size", "1000"])
    assert rc == 0
    out = capsys.readouterr().out
    assert HEADER in out
    assert FILLER_LINE in out


def test_main_writes_to_file(tmp_path: Path):
    target = tmp_path / "stub.txt"
    rc = main([
        "synthetic_compaction.py", "--size", "1000",
        "--out", str(target),
    ])
    assert rc == 0
    assert target.exists()
    assert HEADER in target.read_text()


def test_main_creates_parent_directory(tmp_path: Path):
    """--out path may include directories that don't exist yet."""
    nested = tmp_path / "nested" / "dirs" / "stub.txt"
    rc = main([
        "synthetic_compaction.py", "--size", "500",
        "--out", str(nested),
    ])
    assert rc == 0
    assert nested.exists()
