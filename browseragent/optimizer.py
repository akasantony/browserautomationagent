"""LLM-based optimization of recorded execution paths."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from browseragent.recorder import ExecutionPath, RecordedStep

OPTIMIZER_SYSTEM_PROMPT = """\
You are an expert at optimizing browser automation workflows.

You will receive a recorded sequence of Playwright browser tool calls that
successfully completed a task.  Your job is to analyse the sequence and
return an **optimised** version that:

1. Removes redundant or unnecessary steps (e.g. duplicate snapshots taken
   back-to-back, retries of failed actions that were later repeated
   successfully, unnecessary navigations).
2. Preserves every step that is **essential** for the task to succeed —
   ordering, navigation, form filling, data extraction, etc.
3. Keeps all `browser_snapshot` calls that are needed before any action
   that uses `ref` values (these are mandatory).
4. Does NOT reorder steps — only remove unnecessary ones.

Return ONLY a JSON array of the optimised steps.  Each step must have the
exact keys: `step_index` (re-numbered from 0), `tool_name`, `tool_args`,
`tool_result`, and `element_descriptor`.  Do not add explanation text —
respond with the JSON array only.
"""


async def optimize_path(
    path: ExecutionPath,
    *,
    model_name: str = "gpt-4o",
    api_key: str = "",
) -> ExecutionPath:
    """Use an LLM to prune redundant steps from a recorded path.

    Returns a new ``ExecutionPath`` with ``optimized=True``.
    If the LLM determines the path is already optimal it is returned as-is
    (still marked optimized).
    """
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        temperature=0,
    )

    # Build a concise representation of steps for the LLM
    steps_for_llm = []
    for s in path.steps:
        steps_for_llm.append(
            {
                "step_index": s.step_index,
                "tool_name": s.tool_name,
                "tool_args": s.tool_args,
                "tool_result": s.tool_result[:500],  # Further truncate for prompt
                "element_descriptor": s.element_descriptor,
            }
        )

    human_content = (
        f"Task: {path.task}\n"
        f"URL: {path.url}\n\n"
        f"Recorded steps ({len(steps_for_llm)} total):\n"
        f"```json\n{json.dumps(steps_for_llm, indent=2)}\n```"
    )

    response = await llm.ainvoke(
        [
            SystemMessage(content=OPTIMIZER_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]
    )

    # Parse the LLM response
    optimized_steps = _parse_optimized_steps(response.content, path.steps)

    return ExecutionPath(
        url=path.url,
        task=path.task,
        steps=optimized_steps,
        created_at=path.created_at,
        optimized=True,
        final_response=path.final_response,
    )


def _parse_optimized_steps(
    llm_response: str,
    original_steps: list[RecordedStep],
) -> list[RecordedStep]:
    """Parse the LLM JSON response into RecordedStep objects.

    Falls back to the original steps if parsing fails.
    """
    try:
        # Strip markdown code fences if present
        cleaned = llm_response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        data: list[dict[str, Any]] = json.loads(cleaned)

        steps = []
        for i, item in enumerate(data):
            steps.append(
                RecordedStep(
                    step_index=i,
                    tool_name=item["tool_name"],
                    tool_args=item.get("tool_args", {}),
                    tool_result=item.get("tool_result", ""),
                    element_descriptor=item.get("element_descriptor"),
                )
            )
        return steps

    except (json.JSONDecodeError, KeyError, TypeError):
        # If parsing fails, return the originals unchanged
        return original_steps
