"""Tests for vllm.log → per-request hit metrics extraction (followup #6 close)."""
from skillcacher.bench.log_metrics import (
    parse_vllm_log_for_hits,
    aggregate_log_hits,
    PerRequestHits,
)

# Sample lines from a real cacheblend pod's vllm.log (cacheblend_proof.sh
# MODE=permuted; req 1 = warmup, req 2 = different question + permuted passages).
SAMPLE_LOG = """\
(EngineCore pid=2092) [32;20m[2026-05-06 03:02:09,456] LMCache INFO:[0m Reqid: chatcmpl-8d9881c4f28ca7e1-98dda571, Total tokens 1839, Inference Engine computed tokens: 0, LMCache hit tokens: 0, need to load: 0 [3m(vllm_v1_adapter.py:1324:lmcache.integration.vllm.vllm_v1_adapter)[0m
(EngineCore pid=2092) [32;20m[2026-05-06 03:02:09,806] LMCache INFO:[0m [req_id=chatcmpl-8d9881c4f28ca7e1-98dda571] Stored 1782 out of total 1792 tokens. size: 0.2447 GB
(EngineCore pid=2092) [32;20m[2026-05-06 03:02:13,292] LMCache INFO:[0m Reqid: chatcmpl-9ccbe0e81ee82e9b-b4fa0173, Total tokens 1848, Inference Engine computed tokens: 0, LMCache hit tokens: 1708, need to load: 0 [3m(vllm_v1_adapter.py:1324:lmcache.integration.vllm.vllm_v1_adapter)[0m
"""


def test_parser_extracts_one_record_per_request(tmp_path):
    log = tmp_path / "vllm.log"
    log.write_text(SAMPLE_LOG)
    rows = parse_vllm_log_for_hits(log)
    assert len(rows) == 2
    assert rows[0].req_id == "chatcmpl-8d9881c4f28ca7e1-98dda571"
    assert rows[0].total_tokens == 1839
    assert rows[0].hit_tokens == 0
    assert rows[1].req_id == "chatcmpl-9ccbe0e81ee82e9b-b4fa0173"
    assert rows[1].total_tokens == 1848
    assert rows[1].hit_tokens == 1708


def test_parser_dedupes_repeated_req_ids(tmp_path):
    """lmcache sometimes logs the lookup result line multiple times as the
    engine progresses through layerwise prefill; we keep first-only."""
    dup_line = SAMPLE_LOG.splitlines()[0]
    log = tmp_path / "vllm.log"
    log.write_text(dup_line + "\n" + dup_line + "\n")
    rows = parse_vllm_log_for_hits(log)
    assert len(rows) == 1


def test_parser_ignores_unrelated_lines(tmp_path):
    log = tmp_path / "vllm.log"
    log.write_text(
        "INFO some unrelated chatter\n"
        "ERROR EngineCore failed to start\n"
        "Stored 1782 out of total 1792 tokens\n"  # store line, not lookup
    )
    assert parse_vllm_log_for_hits(log) == []


def test_parser_handles_missing_file(tmp_path):
    assert parse_vllm_log_for_hits(tmp_path / "does_not_exist.log") == []


def test_aggregate_overall_hit_rate():
    rows = [
        PerRequestHits("a", total_tokens=1839, computed_tokens=0, hit_tokens=0),
        PerRequestHits("b", total_tokens=1848, computed_tokens=0, hit_tokens=1708),
    ]
    agg = aggregate_log_hits(rows)
    assert agg.n_requests == 2
    assert agg.total_hit_tokens == 1708
    assert agg.prompt_token_count == 1839 + 1848
    # overall_hit_rate = 1708 / 3687 ≈ 0.463
    assert abs(agg.overall_hit_rate - 1708 / (1839 + 1848)) < 1e-9
    # No span info → skill_* zeroed (followup #6 docs this).
    assert agg.skill_hit_tokens == 0
    assert agg.skill_total_tokens == 0


def test_aggregate_empty():
    agg = aggregate_log_hits([])
    assert agg.n_requests == 0
    assert agg.overall_hit_rate == 0.0
    assert agg.total_hit_tokens == 0
