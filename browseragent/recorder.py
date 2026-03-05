"""Record execution paths and persist them for replay."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RecordedStep:
    """A single tool invocation captured during an LLM-driven run."""

    step_index: int
    tool_name: str
    tool_args: dict[str, Any]
    tool_result: str  # Truncated for storage efficiency

    # Semantic identifiers extracted from snapshot context.
    # Used by the replayer to re-map ephemeral `ref` values.
    element_descriptor: dict[str, str] | None = None


@dataclass
class ExecutionPath:
    """An ordered sequence of steps that accomplish a browser task."""

    url: str
    task: str
    steps: list[RecordedStep] = field(default_factory=list)
    created_at: str = ""
    optimized: bool = False
    final_response: str = ""  # The LLM's formatted final answer

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionPath":
        steps = [RecordedStep(**s) for s in data.pop("steps", [])]
        return cls(steps=steps, **data)


class PathStore:
    """Manages persistence of ExecutionPath objects as JSON files."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(
            base_dir or os.path.expanduser("~/.browseragent/paths")
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(url: str, task: str) -> str:
        """Deterministic hash from (url, task)."""
        raw = f"{url}|{task}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _path_file(self, url: str, task: str) -> Path:
        return self.base_dir / f"{self._make_key(url, task)}.json"

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def has_path(self, url: str, task: str) -> bool:
        return self._path_file(url, task).exists()

    def save_path(
        self,
        execution_path: ExecutionPath,
        *,
        keep_unoptimized: bool = True,
    ) -> Path:
        """Persist an ExecutionPath. Optionally keeps the pre-optimization copy."""
        filepath = self._path_file(execution_path.url, execution_path.task)

        # If we're saving an optimized version and an unoptimized one already
        # exists, keep the original as a backup.
        if keep_unoptimized and execution_path.optimized and filepath.exists():
            backup = filepath.with_suffix(".unoptimized.json")
            if not backup.exists():
                filepath.rename(backup)

        filepath.write_text(
            json.dumps(execution_path.to_dict(), indent=2),
            encoding="utf-8",
        )
        return filepath

    def load_path(self, url: str, task: str) -> ExecutionPath:
        filepath = self._path_file(url, task)
        data = json.loads(filepath.read_text(encoding="utf-8"))
        return ExecutionPath.from_dict(data)

    def delete_path(self, url: str, task: str) -> None:
        for suffix in (".json", ".unoptimized.json"):
            f = self.base_dir / f"{self._make_key(url, task)}{suffix}"
            f.unlink(missing_ok=True)

    def clear_all(self) -> int:
        """Delete every stored path. Returns the count of files removed."""
        count = 0
        for f in self.base_dir.glob("*.json"):
            f.unlink()
            count += 1
        return count


class StepRecorder:
    """Callback-style recorder attached to an agent run.

    Usage::

        recorder = StepRecorder(url, task)
        # ... inside agent loop ...
        recorder.record(tool_name, tool_args, tool_result)
        # ... after completion ...
        path = recorder.finish()
    """

    def __init__(self, url: str, task: str) -> None:
        self._path = ExecutionPath(url=url, task=task)
        self._index = 0

    def record(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: str,
        element_descriptor: dict[str, str] | None = None,
    ) -> None:
        """Append a step."""
        step = RecordedStep(
            step_index=self._index,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result[:2000],  # Truncate long results
            element_descriptor=element_descriptor,
        )
        self._path.steps.append(step)
        self._index += 1

    def set_final_response(self, content: str) -> None:
        """Store the LLM's formatted final answer."""
        self._path.final_response = content

    def finish(self) -> ExecutionPath:
        """Return the completed ExecutionPath."""
        return self._path

    @property
    def steps(self) -> list[RecordedStep]:
        return self._path.steps
