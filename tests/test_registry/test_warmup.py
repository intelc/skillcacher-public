import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from skillcacher.tagging.skill_prefix_index import SkillPrefixIndex
from skillcacher.registry.span_registry import SpanRegistry
from skillcacher.registry.warmup import pre_seed_skills, pre_seed_statistical_spans


TEST_TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    d = tmp_path / "skills"
    d.mkdir()
    s = d / "alpha"
    s.mkdir()
    (s / "SKILL.md").write_text(
        "---\nname: alpha\n---\n\n# Alpha\n\n" + ("body. " * 200)
    )
    return d


@pytest.mark.asyncio
async def test_pre_seed_registers_each_anchor(skills_dir, tmp_path):
    idx = SkillPrefixIndex([skills_dir], anchor_bytes=[1024])
    registry = SpanRegistry(tmp_path / "spans.sqlite")
    registry.init_schema()
    fake_controller = MagicMock()
    fake_controller.pin = AsyncMock(return_value=True)
    fake_warmup = AsyncMock()

    await pre_seed_skills(idx, registry, fake_controller, TEST_TOKENIZER, warmup_fn=fake_warmup)

    spans = registry.spans_referenced_by(skill_ids=["alpha"])
    assert len(spans) == 1
    assert spans[0].skill_id == "alpha"
    assert spans[0].anchor_bytes == 1024
    assert fake_controller.pin.called
    assert fake_warmup.called


@pytest.mark.asyncio
async def test_pre_seed_continues_on_pin_failure(skills_dir, tmp_path):
    idx = SkillPrefixIndex([skills_dir], anchor_bytes=[1024])
    registry = SpanRegistry(tmp_path / "spans.sqlite")
    registry.init_schema()
    fake_controller = MagicMock()
    fake_controller.pin = AsyncMock(return_value=False)  # pin fails
    fake_warmup = AsyncMock()

    # Should not raise
    await pre_seed_skills(idx, registry, fake_controller, TEST_TOKENIZER, warmup_fn=fake_warmup)
    spans = registry.spans_referenced_by(skill_ids=["alpha"])
    assert len(spans) == 1  # registered even though pin failed


@pytest.mark.asyncio
async def test_pre_seed_without_controller(skills_dir, tmp_path):
    """When the lmcache shim isn't configured, controller is None — the
    warmup prefill alone populates lmcache via vLLM's kv_transfer integration.
    oneshot pods don't run the shim, so controller=None is the
    bench's default path."""
    idx = SkillPrefixIndex([skills_dir], anchor_bytes=[1024])
    registry = SpanRegistry(tmp_path / "spans.sqlite")
    registry.init_schema()
    fake_warmup = AsyncMock()

    n = await pre_seed_skills(idx, registry, None, TEST_TOKENIZER, warmup_fn=fake_warmup)

    assert n == 1
    assert fake_warmup.called
    spans = registry.spans_referenced_by(skill_ids=["alpha"])
    assert len(spans) == 1  # still registered without the pin


# --- pre_seed_statistical_spans -------------------------------


@pytest.mark.asyncio
async def test_pre_seed_statistical_registers_each_record(tmp_path: Path):
    spans_file = tmp_path / "stat.jsonl"
    spans_file.write_text(
        json.dumps({"fingerprint": "aaa", "token_ids": [1, 2, 3], "frequency": 5, "length": 3}) + "\n"
        + json.dumps({"fingerprint": "bbb", "token_ids": [4, 5, 6, 7], "frequency": 3, "length": 4}) + "\n"
    )
    registry = SpanRegistry(tmp_path / "spans.sqlite")
    registry.init_schema()
    fake_controller = MagicMock()
    fake_controller.pin = AsyncMock(return_value=True)
    fake_warmup = AsyncMock()

    n = await pre_seed_statistical_spans(spans_file, registry, fake_controller, warmup_fn=fake_warmup)
    assert n == 2

    rows = registry.all_spans(source="statistical")
    by_id = {s.span_id: s for s in rows}
    assert "stat:aaa" in by_id and "stat:bbb" in by_id
    assert by_id["stat:aaa"].token_ids == [1, 2, 3]
    assert by_id["stat:bbb"].token_ids == [4, 5, 6, 7]
    assert all(s.skill_id == "_statistical" for s in rows)
    assert all(s.anchor_bytes == 0 for s in rows)
    assert fake_warmup.call_count == 2
    assert fake_controller.pin.call_count == 2


@pytest.mark.asyncio
async def test_pre_seed_statistical_no_op_when_file_missing(tmp_path: Path):
    registry = SpanRegistry(tmp_path / "spans.sqlite")
    registry.init_schema()
    fake_warmup = AsyncMock()
    n = await pre_seed_statistical_spans(
        tmp_path / "absent.jsonl", registry, None, warmup_fn=fake_warmup,
    )
    assert n == 0
    assert not fake_warmup.called


@pytest.mark.asyncio
async def test_pre_seed_statistical_skips_blank_lines(tmp_path: Path):
    spans_file = tmp_path / "stat.jsonl"
    spans_file.write_text(
        "\n"
        + json.dumps({"fingerprint": "aaa", "token_ids": [1, 2], "frequency": 3, "length": 2}) + "\n"
        + "   \n"
        + json.dumps({"fingerprint": "bbb", "token_ids": [3, 4], "frequency": 4, "length": 2}) + "\n"
    )
    registry = SpanRegistry(tmp_path / "spans.sqlite")
    registry.init_schema()
    n = await pre_seed_statistical_spans(spans_file, registry, None)
    assert n == 2
