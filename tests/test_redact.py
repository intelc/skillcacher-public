"""redact.py scrubs ops artifacts from captured trace bytes before publish."""
from scripts.redact import redact_text


def test_redacts_runpod_url():
    text = "calling https://abc123def456-8000.proxy.runpod.net/v1/chat/completions"
    out = redact_text(text)
    assert "abc123def456" not in out
    assert "<REDACTED_RUNPOD>" in out


def test_redacts_session_id():
    text = "ANTHROPIC_API_KEY=sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz1234567890"
    out = redact_text(text)
    assert "AbCdEfGhIjKlMnOpQrStUvWxYz" not in out
    assert "<REDACTED_API_KEY>" in out


def test_redacts_runpod_pod_token():
    text = '"Authorization": "Bearer sk-podid12345abc"'
    out = redact_text(text)
    assert "podid12345abc" not in out
    assert "<REDACTED_API_KEY>" in out


def test_redacts_cc_version_header():
    text = '"x-app": "cli","cli_version": "2.1.62"'
    out = redact_text(text)
    assert "2.1.62" not in out
    assert "<REDACTED_CC_VERSION>" in out


def test_passes_swebench_content_through():
    text = "def add(a, b): return a + b\n# SWE-Bench task astropy__astropy-12907"
    out = redact_text(text)
    assert out == text


def test_redacts_system_hash():
    text = '{"system_prompt_hash": "abc123def456789xyz"}'
    out = redact_text(text)
    assert "abc123def456789xyz" not in out
    assert '"system_prompt_hash": "<REDACTED_HASH>"' in out


def test_does_not_redact_bearer_in_test_code():
    text = 'assert headers == {"Other": "Bearer testtoken123"}'
    out = redact_text(text)
    # Not in JSON Authorization context, should pass through
    assert "Bearer testtoken123" in out


def test_does_not_redact_sk_prefix_inside_identifier():
    # _suffix is a word char directly after the key chars: \b does NOT fire
    # because the regex's char class no longer includes `_`, the greedy match
    # stops at _, then \b fires INSIDE the token (between letter and `_`),
    # producing no match. The text passes through unchanged.
    text = "lookup_table[sk-test-aBcDeFgHiJkLmNoPqRsTuVwXyZ0_suffix]"
    out = redact_text(text)
    assert "sk-test-aBcDeFgHiJkLmNoPqRsTuVwXyZ0_suffix" in out


def test_redacts_tailscale_hostname():
    text = "ssh into laptop.tailnet-abc1.ts.net for the GPU box"
    out = redact_text(text)
    assert "tailnet-abc1.ts.net" not in out
    assert "<REDACTED_TAILSCALE>" in out


def test_redacts_hf_token():
    text = 'env["HF_TOKEN"] = "hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890"'
    out = redact_text(text)
    assert "AbCdEfGhIjKlMnOpQrStUvWxYz" not in out
    assert "<REDACTED_HF_TOKEN>" in out


def test_redacts_runpod_api_key():
    text = "RUNPOD_API_KEY=RPA_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ABCDEFGH"
    out = redact_text(text)
    assert "RPA_ABCDEFGHIJ" not in out
    assert "<REDACTED_RUNPOD_KEY>" in out


def test_passes_legit_ts_net_lookalikes_through():
    # `something.ts.network` doesn't match (TLD is 'network', not 'net').
    text = "see github.com/foo/bar or stats.ts.network/api"
    out = redact_text(text)
    assert "github.com" in out
    # `.ts.network` should NOT be touched — the regex anchors on `\b` after `.ts.net`,
    # but `network` is a word char, so no boundary; pass-through.
    assert "stats.ts.network" in out


def test_redacts_tailscale_auth_key():
    """oneshot_boot.log captures the full
    `tailscale up --auth-key tskey-auth-...` command line. the harness's redact
    pass only covered Tailscale hostnames, missing the auth-key form."""
    text = (
        "+ tailscale up --auth-key "
        "tskey-auth-kr7uaqDKw521CNTRL-x1NFD9xTVZZd6hk4tVY1aZLRY282jQuTR "
        "--hostname skillcacher-bench-cacheblend-82832 --ssh"
    )
    out = redact_text(text)
    assert "tskey-auth-kr7uaqDKw" not in out
    assert "<REDACTED_TAILSCALE_AUTH_KEY>" in out
    # Hostname stays — it's not a `.ts.net` match, just a pod name.
    assert "skillcacher-bench-cacheblend-82832" in out


def test_dir_mode_walks_log_and_txt(tmp_path):
    """oneshot_boot.log + per-turn _stdout.txt files get
    written AFTER the in-script redact pass (orchestrator dumps them at
    teardown), so dir-mode redact must walk .log/.txt as well as
    .json/.parquet."""
    from scripts.redact import main as redact_main
    log_file = tmp_path / "oneshot_boot.log"
    log_file.write_text(
        "+ tailscale up --auth-key tskey-auth-foo123BAR456baz789QUX012TEST --hostname x"
    )
    txt_file = tmp_path / "_turn_1_stdout.txt"
    txt_file.write_text(
        "API_KEY=sk-ant-api03-abc123DEF456ghi789JKL012mno345PQR678 was used"
    )
    json_file = tmp_path / "meta.json"
    json_file.write_text('{"hf": "hf_AAA1BBB2CCC3DDD4EEE5FFF6GGG7HHH8II"}')
    import sys
    saved = sys.argv
    try:
        sys.argv = ["redact.py", str(tmp_path), "--in-place"]
        redact_main()
    finally:
        sys.argv = saved
    assert "tskey-auth" not in log_file.read_text()
    assert "<REDACTED_TAILSCALE_AUTH_KEY>" in log_file.read_text()
    assert "sk-ant-api03-abc123" not in txt_file.read_text()
    assert "<REDACTED_API_KEY>" in txt_file.read_text()
    assert "<REDACTED_HF_TOKEN>" in json_file.read_text()
