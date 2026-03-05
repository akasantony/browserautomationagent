"""LangGraph ReAct agent wired to Playwright MCP browser tools."""

from __future__ import annotations

import contextlib
import shutil
from typing import Any, AsyncGenerator, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from browseragent.config import Settings

# Type alias for the recording callback.
# Signature: (tool_name, tool_args, tool_result) -> None
RecordCallback = Callable[[str, dict[str, Any], str], None]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert browser automation agent. You control a real Chromium browser
through Playwright MCP tools.

## Your mission
1. Navigate to the target URL.
2. If login is required and credentials are provided below, fill the login form
   and submit it.
3. Navigate to the relevant page as described in the user's task.
4. Extract the requested data.

## Auth credentials (use ONLY when a login form is present)
- Username: {auth_username}
- Password: {auth_password}

## CRITICAL workflow — follow this EXACT order:
1. Call `browser_navigate` to go to the URL.
2. Call `browser_snapshot` to get a FRESH snapshot with current `ref` values.
3. Use ONLY the `ref` values from the MOST RECENT `browser_snapshot` result.
4. To fill a form field, call `browser_click` on the input field's ref FIRST,
   then call `browser_type` with the text to enter and the same ref.
5. After filling all fields, call `browser_click` on the submit/login button ref.
6. Call `browser_snapshot` again to see the result page.

## IMPORTANT rules:
- ONLY make ONE tool call per turn. NEVER call multiple tools in parallel.
  Browser actions must be strictly sequential.
- NEVER use `browser_fill_form`. Always use `browser_click` + `browser_type`
  for each field individually.
- NEVER reuse `ref` values from a previous snapshot or from `browser_navigate`.
  Always take a fresh `browser_snapshot` first.
- After EVERY navigation or page change, call `browser_snapshot` before doing
  anything else.
- If a tool call fails, call `browser_snapshot` to get fresh refs before retrying.
- When done, return extracted data as a well-formatted JSON object in your final
  message. Do NOT wrap it in a tool call — just respond with the JSON.
- Never invent data. Only return what is actually on the page.
"""


def _find_npx() -> str:
    """Return the full path to npx, or raise if not found."""
    npx = shutil.which("npx")
    if npx is None:
        raise SystemExit(
            "npx is not installed. Install Node.js (>=18) to use the Playwright MCP server."
        )
    return npx


def _build_mcp_config(settings: Settings) -> dict[str, Any]:
    """Return the MCP client configuration dict for Playwright."""
    npx_path = _find_npx()
    mcp_args = ["@playwright/mcp@latest", "--browser", "chromium"]
    if settings.headless:
        mcp_args.append("--headless")
    return {
        "playwright": {
            "command": npx_path,
            "args": mcp_args,
            "transport": "stdio",
        }
    }


@contextlib.asynccontextmanager
async def open_mcp_session(
    settings: Settings,
) -> AsyncGenerator[list[BaseTool], None]:
    """Context manager that starts a Playwright MCP session and yields tools.

    This is shared between the LLM agent (record mode) and the replayer so
    both use the same session-construction logic.
    """
    client = MultiServerMCPClient(_build_mcp_config(settings))

    async with client.session("playwright") as session:
        tools = await load_mcp_tools(session)
        yield tools


async def run_agent_stream(
    url: str,
    task: str,
    settings: Settings,
    *,
    record_callback: RecordCallback | None = None,
) -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
    """Spin up the Playwright MCP server, build the LangGraph agent, and
    stream step-by-step events.

    Yields ``(node_name, chunk_data)`` tuples as the agent works.
    The final chunk from the ``"agent"`` node contains the answer.

    If *record_callback* is provided it is called after every tool execution
    with ``(tool_name, tool_args, tool_result)`` so the caller can record
    the execution path.
    """
    async with open_mcp_session(settings) as tools:
        # --- LLM ---
        llm = ChatOpenAI(
            model=settings.model_name,
            api_key=settings.openai_api_key,
            temperature=0,
        )

        # --- Graph ---
        def call_model(state: MessagesState) -> dict:
            # parallel_tool_calls=False forces one tool call per turn
            response = llm.bind_tools(tools, parallel_tool_calls=False).invoke(state["messages"])
            return {"messages": response}

        builder = StateGraph(MessagesState)
        builder.add_node("agent", call_model)
        builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", tools_condition)
        builder.add_edge("tools", "agent")

        graph = builder.compile()

        # --- Build messages ---
        system_msg = SystemMessage(
            content=SYSTEM_PROMPT_TEMPLATE.format(
                auth_username=settings.auth_username or "(not provided)",
                auth_password=settings.auth_password or "(not provided)",
            )
        )
        human_msg = HumanMessage(
            content=f"Target URL: {url}\n\nTask: {task}"
        )

        # Track the most recent AI tool calls so we can pair them with
        # their ToolMessage results for the recording callback.
        pending_tool_calls: dict[str, dict[str, Any]] = {}

        # --- Stream ---
        async for chunk in graph.astream(
            {"messages": [system_msg, human_msg]},
            config={"recursion_limit": 60},
            stream_mode="updates",
        ):
            for node_name, node_data in chunk.items():
                messages = node_data.get("messages", [])
                if not isinstance(messages, list):
                    messages = [messages]

                # Capture pending tool calls from the agent node
                if node_name == "agent":
                    for msg in messages:
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for tc in msg.tool_calls:
                                pending_tool_calls[tc["id"]] = {
                                    "name": tc["name"],
                                    "args": tc.get("args", {}),
                                }

                # When tools respond, fire the recording callback
                if node_name == "tools" and record_callback is not None:
                    for msg in messages:
                        if isinstance(msg, ToolMessage):
                            tc_id = msg.tool_call_id
                            tc_info = pending_tool_calls.pop(tc_id, None)
                            if tc_info:
                                record_callback(
                                    tc_info["name"],
                                    tc_info["args"],
                                    str(msg.content),
                                )

                yield node_name, node_data
