"""Per-request JSON checkpoint store. Resumable bench runs survive
spot preemption and inter-condition pod cycling."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


class CheckpointStore:
    def __init__(self, run_root: Path):
        self.run_root = Path(run_root)
        self.run_root.mkdir(parents=True, exist_ok=True)

    def _condition_dir(self, condition: str) -> Path:
        d = self.run_root / condition
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_request(self, condition: str, index: int, record: dict) -> None:
        path = self._condition_dir(condition) / f"{index:06d}.json"
        path.write_text(json.dumps(record))

    def load_request(self, condition: str, index: int) -> dict | None:
        path = self._condition_dir(condition) / f"{index:06d}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def next_index(self, condition: str) -> int:
        """Highest contiguous index + 1 (re-runs gaps).

        Treating gaps as 'redo' rather than 'skip' is conservative — a gap
        means a partial write or a corrupted record."""
        d = self._condition_dir(condition)
        idx = 0
        while (d / f"{idx:06d}.json").exists():
            idx += 1
        return idx

    def iter_completed(self, condition: str) -> Iterator[dict]:
        d = self._condition_dir(condition)
        for path in sorted(d.glob("*.json")):
            yield json.loads(path.read_text())
