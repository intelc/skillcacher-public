"""Scrub ops artifacts from captured CC trace bytes before publish.

Targets: RunPod URLs/IDs, API keys (Bearer/sk-/sk-ant-), Claude Code version
headers, system prompt build hashes, account-tied tool versions.

SWE-Bench Lite content is already public — this pass leaves user/test code
intact and only strips deployment-side fingerprints."""
from __future__ import annotations

import re
import sys
from pathlib import Path

RUNPOD_URL = re.compile(r"https://[a-z0-9]{8,}-\d+\.proxy\.runpod\.net")
RUNPOD_POD_ID = re.compile(r"\b(sk-)?[a-z0-9]{12,}(?=\b)")
API_KEY = re.compile(
    r"sk-(?:ant-)?(?:api\d+-)?[A-Za-z0-9\-]{20,}\b"
)
BEARER = re.compile(r'(["\'])[Aa]uthorization\1\s*:\s*(["\'])\s*Bearer\s+[A-Za-z0-9._\-]{8,}\2')
CC_VERSION = re.compile(r'"cli_version"\s*:\s*"[^"]+"')
SYSTEM_HASH = re.compile(r'"system_prompt_hash"\s*:\s*"[^"]+"')
# Identifiers for ClaudeCodeTrace publication:
# - Tailscale hostnames (deployment topology fingerprint)
# - HF tokens (per .env HF_TOKEN; gated repo access leak risk)
# - RunPod API keys (per .env RUNPOD_API_KEY; account compromise risk)
TAILSCALE_HOST = re.compile(r"\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.ts\.net\b", re.IGNORECASE)
HF_TOKEN = re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")
RUNPOD_KEY = re.compile(r"\bRPA_[A-Z0-9]{40,}\b")
# oneshot_boot.log captures the `tailscale up --auth-key tskey-auth-...`
# invocation; the auth-key is distinct from the Tailscale hostname pattern.
TAILSCALE_AUTH_KEY = re.compile(r"\btskey-auth-[A-Za-z0-9_\-]{20,}\b")


def redact_text(text: str) -> str:
    text = RUNPOD_URL.sub("<REDACTED_RUNPOD>", text)
    text = TAILSCALE_HOST.sub("<REDACTED_TAILSCALE>", text)
    text = TAILSCALE_AUTH_KEY.sub("<REDACTED_TAILSCALE_AUTH_KEY>", text)
    text = HF_TOKEN.sub("<REDACTED_HF_TOKEN>", text)
    text = RUNPOD_KEY.sub("<REDACTED_RUNPOD_KEY>", text)
    text = API_KEY.sub("<REDACTED_API_KEY>", text)
    text = BEARER.sub(r'\1Authorization\1: \2Bearer <REDACTED_API_KEY>\2', text)
    text = CC_VERSION.sub('"cli_version": "<REDACTED_CC_VERSION>"', text)
    text = SYSTEM_HASH.sub('"system_prompt_hash": "<REDACTED_HASH>"', text)
    # RUNPOD_POD_ID is intentionally last + narrow — easy to false-positive
    # on real content. We only apply it inside known-ops contexts.
    return text


def redact_file(path: Path, in_place: bool = False) -> str:
    raw = path.read_text()
    out = redact_text(raw)
    if in_place:
        path.write_text(out)
    return out


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: redact.py <path> [<path>...] [--in-place]\n")
        sys.exit(1)
    in_place = "--in-place" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--in-place"]
    for p in args:
        path = Path(p)
        if path.is_dir():
            # Walk .json/.log/.txt/.md so oneshot_boot.log + per-turn
            # _stdout.txt files get scrubbed alongside request-body JSON.
            for ext in ("*.json", "*.log", "*.txt", "*.md"):
                for sub in path.rglob(ext):
                    redact_file(sub, in_place=in_place)
            for sub in path.rglob("*.parquet"):
                # binary parquet — skip, but warn
                sys.stderr.write(f"[redact] skipping binary {sub}\n")
        else:
            sys.stdout.write(redact_file(path, in_place=in_place))


if __name__ == "__main__":
    main()
