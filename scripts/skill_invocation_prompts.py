"""build (skill, prompt) pairs that explicitly invoke a skill body.

For each SKILL.md under the given dirs, render the body the way CC does
(frontmatter and !`cmd` markers stripped — see
`skillcacher.tagging.skill_prefix_index.render_skill_body`), take the
leading ANCHOR_BYTES of the rendered body, and embed that verbatim in three
prompts. Each prompt:

- mentions the skill_id by name
- contains a verbatim ≥1024-byte slice from the rendered body (the
  smallest anchor in `SkillPrefixIndex`, so the prefix index DOES find a
  match when the proxy parses the request)
- asks the model to apply the skill to specific task data

The three per-skill tasks differ in the trailing task-data block; the
verbatim slice is constant per skill, so all three prompts trigger the
same anchor match in the proxy. Single-condition cacheblend wiring proof —
see the harness design spec §2.

CLI usage:
    python -m scripts.skill_invocation_prompts \\
        --skill-dir tests/fixtures/test_skills \\
        --out /tmp/skill_invocation_prompts

Writes one `<task_id>.txt` per prompt and a `TASKS_FILE` listing all task
IDs (one per line). The capture orchestrator's `skill_invocation` mode
consumes these.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from skillcacher.tagging.skill_prefix_index import render_skill_body

# 1280 bytes gives us comfortable headroom over the smallest 1024-byte
# anchor in case downstream tooling normalizes whitespace and shaves a
# byte or two off the prompt-side copy.
ANCHOR_BYTES = 1280


# Per-skill task-data blocks. Three per skill so we get 3 (skill, prompt)
# pairs each. The intro text varies the wording per prompt_id so they
# read like distinct turns rather than literal duplicates, but the
# verbatim quote section in the middle is constant — that is what the
# prefix index matches.
TASK_DATA: dict[str, list[tuple[str, str]]] = {
    "weather-format": [
        (
            "hourly_obs",
            "Apply the weather-format skill to this hourly observation set:\n"
            "  Station: KBOS\n"
            "  Date: 2025-04-12\n"
            "  Hourly readings (hour,temp_C,humidity_pct,wind_dir,wind_kph,precip_mm):\n"
            "    00,8,72,NW,11.4,0\n    03,7,75,NW,12.1,0\n    06,8,78,W,9.3,0.2\n"
            "    09,11,68,W,15.6,0\n    12,15,55,SW,18.9,0\n    15,17,49,SW,21.2,0\n"
            "    18,14,58,SW,16.4,0.6\n    21,11,65,W,12.8,0.1\n"
            "Produce the report.",
        ),
        (
            "daily_summary",
            "Apply the weather-format skill to this daily summary:\n"
            "  Location: Reykjavík, IS  Provider: Veðurstofa Íslands  Date: 2025-11-04\n"
            "  high_C=4 low_C=-2 mean_humidity_pct=82\n"
            "  prevailing_wind=NE mean_wind_kph=27.5 precip_mm_24h=8.4\n"
            "  notes: persistent overcast, brief snow showers in the late afternoon.\n"
            "Produce the report.",
        ),
        (
            "noaa_tuple",
            "Apply the weather-format skill to this NOAA-style tuple:\n"
            "  ('Tucson, AZ', 'KTUS', '2025-07-22', 41, 27, 18, 'SE', 9.7, 0.0,\n"
            "   'clear and very hot, well above seasonal norm')\n"
            "Tuple fields are: city_region, station, date, high_C, low_C,\n"
            "mean_humidity_pct, prevailing_wind, mean_wind_kph, precip_mm_24h,\n"
            "narrative_hint. Produce the report.",
        ),
    ],
    "commit-msg-style": [
        (
            "bugfix",
            "Apply the commit-msg-style skill to this change:\n"
            "  Area: auth\n"
            "  Diff summary: rotate_key() previously raised KeyError when the\n"
            "  incoming JWT lacked a `kid` header; the rotation policy expects\n"
            "  to fall back to the active key in that case.\n"
            "  Fix: catch KeyError and fall through to the active key path.\n"
            "  Closes issue #4421.\n"
            "Produce the commit message.",
        ),
        (
            "feature",
            "Apply the commit-msg-style skill to this change:\n"
            "  Area: query planner\n"
            "  Diff summary: add a planner shortcut that skips predicate\n"
            "  pushdown when the query contains a window aggregate, since\n"
            "  the pushdown is a no-op in that case and the analysis pass\n"
            "  is the dominant cost on hot-path dashboards.\n"
            "  Refs issue #5102.\n"
            "Produce the commit message.",
        ),
        (
            "perf",
            "Apply the commit-msg-style skill to this change:\n"
            "  Area: ingest pipeline\n"
            "  Diff summary: replace per-row JSON parse in the hot loop with\n"
            "  a streaming parser that processes batches of 1024 rows; on the\n"
            "  benchmark workload (`bench/ingest_p99.py`) end-to-end p99\n"
            "  drops from 412ms to 178ms.\n"
            "Produce the commit message.",
        ),
    ],
    "bash-safety": [
        (
            "rm_rf",
            "Apply the bash-safety skill to this command:\n"
            "  command: rm -rf $DIR/build\n"
            "  context: $DIR is set by a Makefile recipe. The Makefile is\n"
            "  invoked from a developer's laptop, not in CI. The variable\n"
            "  has been seen to expand to an empty string on a fresh clone\n"
            "  before `make configure` has run.\n"
            "What should the user check before running this?",
        ),
        (
            "drop_table",
            "Apply the bash-safety skill to this operation:\n"
            "  command: psql -c 'DROP TABLE legacy_audit_events;'\n"
            "  context: the staging database. The table has rows that were\n"
            "  not migrated to the new schema; legal has not yet signed off\n"
            "  on the retention plan for the un-migrated subset.\n"
            "What should the user check before running this?",
        ),
        (
            "force_push",
            "Apply the bash-safety skill to this command:\n"
            "  command: git push --force origin shared/release-2025q4\n"
            "  context: the developer rebased their local branch to drop a\n"
            "  commit that accidentally included a vendored binary. Three\n"
            "  other developers have the branch checked out.\n"
            "What should the user check before running this?",
        ),
    ],
    "python-import-order": [
        (
            "tangled",
            "Apply the python-import-order skill to this file head:\n"
            "  from src.app import settings\n"
            "  import os\n"
            "  from typing import Iterable\n"
            "  import requests\n"
            "  from collections import OrderedDict\n"
            "  import json\n"
            "  from src.app.cache import LRUCache\n"
            "  from pydantic import BaseModel\n"
            "Produce the corrected import block.",
        ),
        (
            "type_checking",
            "Apply the python-import-order skill to this file head:\n"
            "  import os\n"
            "  from typing import TYPE_CHECKING\n"
            "  if TYPE_CHECKING:\n"
            "      from src.app.session import Session\n"
            "      from src.app.user import User\n"
            "  import json\n"
            "  import boto3\n"
            "  from src.app import settings\n"
            "Produce the corrected import block.",
        ),
        (
            "multi_from",
            "Apply the python-import-order skill to this file head:\n"
            "  from collections import OrderedDict, defaultdict, namedtuple\n"
            "  import os, sys\n"
            "  from typing import Iterable, Optional, Sequence, Tuple\n"
            "  import requests\n"
            "  from src.app.models import User, Session, AuditEvent\n"
            "Produce the corrected import block.",
        ),
    ],
    "markdown-table": [
        (
            "csv",
            "Apply the markdown-table skill to this CSV:\n"
            "  name,age,role\n"
            "  Alice,34,engineer\n"
            "  Bob,29,designer\n"
            "  Carol,41,manager\n"
            "Produce the rendered table.",
        ),
        (
            "list_of_dicts",
            "Apply the markdown-table skill to this list of dicts:\n"
            "  [{'sku': 'A-100', 'qty': 4, 'unit_price': 12.5},\n"
            "   {'sku': 'B-204', 'qty': 1, 'unit_price': 99.0},\n"
            "   {'sku': 'C-77',  'qty': 8, 'unit_price': 3.25}]\n"
            "Produce the rendered table.",
        ),
        (
            "list_of_tuples",
            "Apply the markdown-table skill to this list of tuples; the\n"
            "user-supplied column names are: timestamp, source, message.\n"
            "  [('2025-04-12T08:14:01Z', 'auth', 'login OK'),\n"
            "   ('2025-04-12T08:14:02Z', 'cache', 'evicted 3 entries'),\n"
            "   ('2025-04-12T08:14:05Z', 'auth', 'rotate signal received')]\n"
            "Produce the rendered table.",
        ),
    ],
}


@dataclass
class InvocationPrompt:
    skill_id: str
    prompt_id: str
    task_id: str  # "<skill_id>__<prompt_id>", filename-safe
    prompt: str


def _safe_token(s: str) -> str:
    return s.replace("/", "_").replace("-", "_")


def build_prompts(
    skill_dirs: list[Path], *, anchor_bytes: int = ANCHOR_BYTES
) -> list[InvocationPrompt]:
    """For each SKILL.md under skill_dirs, build 3 prompts. Skills missing
    from TASK_DATA are skipped with a warning. Each prompt contains a
    verbatim slice of `anchor_bytes` from the rendered body."""
    out: list[InvocationPrompt] = []
    seen: set[str] = set()
    for d in skill_dirs:
        for skill_md in sorted(d.rglob("SKILL.md")):
            skill_id = skill_md.parent.name
            if skill_id in seen:
                continue
            seen.add(skill_id)
            rendered = render_skill_body(skill_md)
            if len(rendered) < anchor_bytes:
                print(
                    f"[skill-prompts] skill {skill_id} rendered to "
                    f"{len(rendered)} bytes, less than anchor_bytes "
                    f"{anchor_bytes}; skipping (won't match prefix index).",
                    file=sys.stderr,
                )
                continue
            tasks = TASK_DATA.get(skill_id)
            if not tasks:
                print(
                    f"[skill-prompts] no TASK_DATA entry for {skill_id}; "
                    f"skipping (add to TASK_DATA in this module).",
                    file=sys.stderr,
                )
                continue
            anchor_text = rendered[:anchor_bytes].decode("utf-8")
            for prompt_id, task_block in tasks:
                prompt = (
                    f"You have access to the `{skill_id}` skill. Its body "
                    f"begins with the following text — quoted verbatim so "
                    f"you can apply it precisely:\n\n"
                    f"---BEGIN {skill_id}---\n"
                    f"{anchor_text}"
                    f"---END {skill_id}---\n\n"
                    f"{task_block}\n"
                )
                out.append(
                    InvocationPrompt(
                        skill_id=skill_id,
                        prompt_id=prompt_id,
                        task_id=f"{_safe_token(skill_id)}__{_safe_token(prompt_id)}",
                        prompt=prompt,
                    )
                )
    return out


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skill-dir", action="append", required=True,
                   help="Directory containing <skill_id>/SKILL.md files. Repeatable.")
    p.add_argument("--out", required=True,
                   help="Output directory: writes <task_id>.txt per prompt and TASKS_FILE.")
    p.add_argument("--anchor-bytes", type=int, default=ANCHOR_BYTES,
                   help=f"Bytes of skill body to embed verbatim (default {ANCHOR_BYTES}).")
    args = p.parse_args(argv[1:])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = build_prompts(
        [Path(d) for d in args.skill_dir],
        anchor_bytes=args.anchor_bytes,
    )
    if not prompts:
        print("[skill-prompts] no prompts built — check --skill-dir paths and TASK_DATA coverage", file=sys.stderr)
        return 2

    tasks_file = out_dir / "TASKS_FILE"
    with tasks_file.open("w") as tf:
        for ip in prompts:
            (out_dir / f"{ip.task_id}.txt").write_text(ip.prompt)
            tf.write(f"{ip.task_id}\n")
    print(f"[skill-prompts] wrote {len(prompts)} prompts + TASKS_FILE under {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
