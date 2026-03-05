"""Deterministic replay of a stored execution path without LLM involvement."""

from __future__ import annotations

import json
import re
from typing import Any, AsyncGenerator

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool

from browseragent.recorder import ExecutionPath, RecordedStep


def extract_text_content(result: Any) -> str:
    """Extract clean text from an MCP tool result.

    Playwright MCP tools return structured results like::

        [{'type': 'text', 'text': '### Snapshot\\n...'}]

    This helper pulls out the actual text content. Falls back to
    ``str(result)`` if the format is unexpected.
    """
    if result is None:
        return ""

    # Already a plain string
    if isinstance(result, str):
        return result

    # List of content blocks (standard MCP format)
    if isinstance(result, list):
        parts = []
        for item in result:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            else:
                parts.append(str(item))
        if parts:
            return "\n".join(parts)

    return str(result)


class RefMapper:
    """Maps element descriptors from a recorded run to fresh ``ref`` values.

    Playwright MCP snapshots produce output like::

        - ref=42  button "Submit"
        - ref=17  textbox "Email"

    The mapper parses snapshot text, builds a lookup of
    ``(role, name/text) → ref``, and substitutes old refs in tool args.
    """

    # Pattern: captures ref=<number> followed by role and quoted/unquoted name
    _REF_LINE_RE = re.compile(
        r"ref=(\d+)\s+(\w+)\s+\"?([^\"\\n]*)\"?", re.IGNORECASE
    )

    def __init__(self) -> None:
        # (role_lower, name_lower) → current ref string
        self._current_map: dict[tuple[str, str], str] = {}

    def update_from_snapshot(self, snapshot_text: str) -> None:
        """Parse a ``browser_snapshot`` result and refresh the ref map."""
        self._current_map.clear()
        for match in self._REF_LINE_RE.finditer(snapshot_text):
            ref_val, role, name = match.group(1), match.group(2), match.group(3)
            key = (role.lower().strip(), name.lower().strip())
            self._current_map[key] = ref_val

    def substitute_ref(
        self,
        tool_args: dict[str, Any],
        element_descriptor: dict[str, str] | None,
    ) -> dict[str, Any]:
        """Return a copy of *tool_args* with the ``ref`` value replaced.

        Uses the *element_descriptor* (recorded during the original run) to
        look up the equivalent element in the current snapshot.
        """
        if "ref" not in tool_args or element_descriptor is None:
            return dict(tool_args)

        role = element_descriptor.get("role", "").lower().strip()
        name = element_descriptor.get("name", "").lower().strip()

        key = (role, name)
        new_ref = self._current_map.get(key)

        updated = dict(tool_args)
        if new_ref is not None:
            updated["ref"] = new_ref
        return updated


def _build_element_descriptor(
    tool_name: str,
    tool_args: dict[str, Any],
    snapshot_text: str,
) -> dict[str, str] | None:
    """Extract an element descriptor for the ref used in *tool_args*.

    Searches the most recent *snapshot_text* for the matching ref line and
    returns ``{"role": ..., "name": ...}`` so the replayer can relocate the
    element in future runs.
    """
    ref = tool_args.get("ref")
    if ref is None:
        return None

    ref_str = str(ref)
    for match in RefMapper._REF_LINE_RE.finditer(snapshot_text):
        if match.group(1) == ref_str:
            return {"role": match.group(2), "name": match.group(3)}
    return None


async def replay_path(
    execution_path: ExecutionPath,
    tools: list[BaseTool],
) -> AsyncGenerator[tuple[int, str, str, bool], None]:
    """Replay a stored execution path by calling MCP tools directly.

    Yields ``(step_index, tool_name, tool_result, success)`` tuples.

    If any step fails, the generator yields the failure and stops —
    the caller should fall back to full LLM agent mode.
    """
    # Build a tool lookup by name
    tool_map: dict[str, BaseTool] = {t.name: t for t in tools}
    ref_mapper = RefMapper()

    for step in execution_path.steps:
        tool = tool_map.get(step.tool_name)
        if tool is None:
            yield (step.step_index, step.tool_name, f"Tool '{step.tool_name}' not found", False)
            return

        # Substitute refs for the current session
        args = ref_mapper.substitute_ref(step.tool_args, step.element_descriptor)

        try:
            result = await tool.ainvoke(args)
            result_str = extract_text_content(result)

            # If this was a snapshot, update the ref mapper
            if step.tool_name == "browser_snapshot":
                ref_mapper.update_from_snapshot(result_str)

            yield (step.step_index, step.tool_name, result_str, True)

        except Exception as exc:
            yield (step.step_index, step.tool_name, f"Error: {exc}", False)
            return
