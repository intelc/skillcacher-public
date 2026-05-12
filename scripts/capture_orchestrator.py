"""the harness Layer 1 orchestrator — provision pod + local proxy, run a
bash capture script against it, tear down on exit.

Wraps ``ConditionLifecycle`` (the bench's pod+proxy lifecycle) so the pod
gets cleaned up automatically even if the capture crashes mid-run. Used for
both the harness SWE-V Layer 1 captures (mode=long_session) and the harness
multi-turn /compact captures (mode=compaction).

Usage:
    # the harness SWE-V long-session capture:
    MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct DTYPE=fp8 \\
        python -m scripts.capture_orchestrator --task pylint-dev__pylint-7080

    # the harness multi-turn /compact spike:
    MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct DTYPE=fp8 \\
        python -m scripts.capture_orchestrator --mode compaction \\
            --task compaction_spike_001

The pod is brought up under condition ``prefix_cache`` by default — the
realistic baseline for capture purposes per the project spec. The local
proxy spawned by ConditionLifecycle listens on 127.0.0.1:4000 by default;
we set ``ANTHROPIC_BASE_URL`` accordingly and hand control to the chosen
bash capture script.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from skillcacher.bench.cli import _load_env_file
from skillcacher.bench.conditions import Condition, ConditionLifecycle


async def run_one_capture(
    task_id: str,
    *,
    max_budget_usd: str,
    timeout_secs: str,
    condition: str = "prefix_cache",
    mode: str = "long_session",
    skill_dir: str | None = None,
) -> int:
    """Provision pod + proxy, run capture for one task, tear down on exit.

    `mode="long_session"` calls scripts/capture_long_sessions.sh (the harness
    SWE-V flow). `mode="compaction"` calls scripts/capture_compaction.sh
    (the harness multi-turn /compact flow). `mode="skill_invocation"` calls
    scripts/capture_skill_invocation.sh (iterates a batch of
    pre-built (skill, prompt) prompts under one warm cacheblend pod).

    For `mode="skill_invocation"`, `task_id` is treated as a *batch label*
    (e.g., a timestamp) — the actual per-task IDs come from the prompt
    builder. `skill_dir` MUST be set; the prompt builder reads SKILL.md
    files under it."""
    _load_env_file()

    # Single-task TASKS_FILE points capture_long_sessions.sh at just our target.
    # Captures legitimately wait 10+ min for capacity in busy DCs. Bump the
    # outer pod-boot timeout to 1 hr by default so the queue has time to
    # actually assign a host (the harness capture work, ).
    os.environ.setdefault("WAIT_TIMEOUT_S", "3600")

    # The proxy reads SKILLCACHER_TRACE_DIR at spawn time inside
    # ConditionLifecycle.__aenter__; setting it inside the bash script later
    # is too late (the proxy is already running with the bench's default
    # `benchmark/results/_traces_<cond>/` path, so per-request token parquets
    # land outside our per-task fixture dir). Plant the per-task path here.
    #
    # the harness followup write directly to the FINAL fixture
    # location instead of staging under `_raw/` and `mv`-ing later. The
    # old two-step shape raced with `_dump_logs`: the bash script `mv`'d
    # the staging dir before __aexit__ ran, so the SCP'd vllm.log +
    # oneshot_boot.log landed in a recreated empty `_raw/<TASK>/` instead
    # of inside the moved fixture. Single-dir shape eliminates the race.
    # Default OUT_ROOT per mode mirrors what the bash scripts use; can
    # be overridden via the OUT_ROOT env var for direct invocations.
    out_root_default = {
        "long_session": "tests/fixtures/claude_code_real/swebench_verified",
        "compaction": "tests/fixtures/claude_code_real/post_compact",
        "skill_invocation": "tests/fixtures/claude_code_real/skill_invocation",
    }.get(mode, "tests/fixtures/claude_code_real/_raw")
    raw_root = Path(os.environ.get("RAW_ROOT", out_root_default))
    out_root = Path(os.environ.get("OUT_ROOT", out_root_default))
    # `skill_invocation` mode batches N (skill, prompt) pairs under a single
    # warm pod, so its trace dir is the *batch* root (named by `task_id`)
    # rather than a per-task subdir. The proxy reads SKILLCACHER_TRACE_DIR
    # once at spawn, so all N captures share the same parquet directory;
    # per-task bucketing happens post-hoc by request_id.
    if mode == "skill_invocation":
        task_raw_dir = (raw_root / task_id).resolve()
        task_raw_dir.mkdir(parents=True, exist_ok=True)
        os.environ["SKILLCACHER_TRACE_DIR"] = str(task_raw_dir / "_traces")
        Path(os.environ["SKILLCACHER_TRACE_DIR"]).mkdir(parents=True, exist_ok=True)
    else:
        task_raw_dir = (raw_root / task_id).resolve()
        task_raw_dir.mkdir(parents=True, exist_ok=True)
        os.environ["SKILLCACHER_TRACE_DIR"] = str(task_raw_dir)

    tf = None
    prompts_dir: Path | None = None
    if mode == "long_session":
        tf = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        tf.write(f"{task_id}\n")
        tf.flush()
        tf.close()
    elif mode == "skill_invocation":
        if not skill_dir:
            raise ValueError(
                "mode=skill_invocation requires --skill-dir pointing at a "
                "directory of <skill_id>/SKILL.md fixtures"
            )
        # Build the (skill, prompt) prompt files + TASKS_FILE under the
        # batch raw dir. Done BEFORE pod-up so a misconfigured prompt set
        # fails immediately rather than after a costly pod boot.
        from scripts.skill_invocation_prompts import build_prompts
        prompts = build_prompts([Path(skill_dir).expanduser().resolve()])
        if not prompts:
            raise RuntimeError(
                f"skill_invocation: no prompts built from {skill_dir!r}; "
                f"check that the dir contains <skill_id>/SKILL.md files and "
                f"that TASK_DATA in scripts.skill_invocation_prompts covers them"
            )
        prompts_dir = task_raw_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        tf = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        for ip in prompts:
            (prompts_dir / f"{ip.task_id}.txt").write_text(ip.prompt)
            tf.write(f"{ip.task_id}\n")
        tf.flush()
        tf.close()
        print(
            f"[capture-orch] skill_invocation: built {len(prompts)} prompts "
            f"under {prompts_dir}",
            file=sys.stderr,
        )

    rc: int = -1
    try:
        cond = Condition(name=condition)
        # dump vllm.log to the per-task raw dir BEFORE the pod
        # is deleted, so we have authoritative `LMCache hit tokens: N` lines
        # for retrospective metrics scraping. (Llama doesn't emit
        # `cache_read_input_tokens` in its response body, so without
        # vllm.log the per-request hit count is unrecoverable.)
        log_dump_path = task_raw_dir / "vllm.log"
        async with ConditionLifecycle(cond, log_dump_path=log_dump_path):
            env = os.environ.copy()
            env["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:4000"
            env["TIMEOUT_SECS"] = timeout_secs
            env["MAX_BUDGET_USD"] = max_budget_usd

            # the harness followup tell the bash scripts to use the
            # SAME directory for staging and final output so their `mv`
            # short-circuits (with the same-path guard added in those
            # scripts). This keeps the orchestrator's log_dump_path
            # pointing at a directory that won't be moved away.
            env["RAW_ROOT"] = str(raw_root)
            env["OUT_ROOT"] = str(out_root)
            if mode == "long_session":
                env["TASKS_FILE"] = tf.name
                script = "scripts/capture_long_sessions.sh"
            elif mode == "compaction":
                env["TASK_ID"] = task_id
                script = "scripts/capture_compaction.sh"
            elif mode == "skill_invocation":
                env["TASKS_FILE"] = tf.name
                env["PROMPT_DIR"] = str(prompts_dir)
                env["RAW_ROOT"] = str(task_raw_dir)
                script = "scripts/capture_skill_invocation.sh"
            else:
                raise ValueError(f"unknown mode: {mode!r}")

            print(
                f"[capture-orch] pod up, proxy at 127.0.0.1:4000 — running "
                f"{mode} capture for {task_id}",
                file=sys.stderr,
            )
            r = subprocess.run(["bash", script], env=env)
            rc = r.returncode
    finally:
        if tf is not None:
            Path(tf.name).unlink(missing_ok=True)

    # the harness followup redact the SCP'd logs in place. The
    # orchestrator's `_dump_logs` runs inside __aexit__ and lands
    # vllm.log + oneshot_boot.log under `task_raw_dir` — those files
    # come from the pod and contain proxy URLs / Tailscale auth-keys,
    # so they need the same redact pass as everything else. Doing it
    # here means captures are publication-ready out of the box rather
    # than waiting for `publish_claudecode_trace.py --apply` later.
    try:
        from scripts.redact import redact_file
        for log_name in ("vllm.log", "oneshot_boot.log", "proxy.log"):
            p = task_raw_dir / log_name
            if p.exists() and p.is_file():
                redact_file(p, in_place=True)
    except Exception as e:
        print(f"[capture-orch] post-dump redact failed: {e}", file=sys.stderr)

    return rc


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--task",
        required=True,
        help="SWE-Bench Verified instance_id (e.g., pylint-dev__pylint-7080)",
    )
    p.add_argument(
        "--max-budget-usd",
        default="2",
        help="claude -p --max-budget-usd cap (default 2)",
    )
    p.add_argument(
        "--timeout-secs",
        default="600",
        help="hard wall-clock cap on the claude -p invocation (default 600)",
    )
    p.add_argument(
        "--condition",
        default="prefix_cache",
        choices=["no_cache", "prefix_cache", "cacheblend"],
        help="bench condition for the capture pod (default prefix_cache; "
        "use cacheblend to exercise pre-seed + lmcache retrieval for "
        "Tier-3 skill_hit_rate validation)",
    )
    p.add_argument(
        "--mode",
        default="long_session",
        choices=["long_session", "compaction", "skill_invocation"],
        help="capture flow: long_session calls capture_long_sessions.sh "
        "(the harness SWE-V); compaction calls capture_compaction.sh (the harness "
        "multi-turn /compact spike); skill_invocation calls "
        "capture_skill_invocation.sh (batch of (skill, prompt) "
        "pairs under one warm cacheblend pod, --task is the batch label)",
    )
    p.add_argument(
        "--skill-dir",
        default=None,
        help="(skill_invocation mode only) directory of <skill_id>/SKILL.md "
        "fixtures to drive prompt building; required when --mode=skill_invocation",
    )
    args = p.parse_args(argv[1:])

    return asyncio.run(
        run_one_capture(
            args.task,
            max_budget_usd=args.max_budget_usd,
            timeout_secs=args.timeout_secs,
            condition=args.condition,
            mode=args.mode,
            skill_dir=args.skill_dir,
        )
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv))
