"""CLI entry point for the browser automation agent."""

from __future__ import annotations

import argparse
import asyncio
import json

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from browseragent.config import Settings

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic browser automation powered by Gemini + LangGraph + Playwright MCP",
    )
    parser.add_argument("--url", help="Target website URL")
    parser.add_argument("--task", help="Natural-language task / extraction prompt")
    parser.add_argument(
        "--headless",
        action="store_true",
        default=None,
        help="Run browser in headless mode (default)",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        default=False,
        help="Run browser in visible (headed) mode",
    )
    # Record-Then-Replay flags
    parser.add_argument(
        "--no-replay",
        action="store_true",
        default=False,
        help="Force a fresh LLM run even if a stored path exists",
    )
    parser.add_argument(
        "--clear-paths",
        action="store_true",
        default=False,
        help="Delete all stored paths and exit",
    )
    parser.add_argument(
        "--show-path",
        action="store_true",
        default=False,
        help="Print the stored path for the given URL+task without executing",
    )
    return parser.parse_args()


def _truncate(text: str, max_len: int = 300) -> str:
    """Truncate long text for display."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"… (+{len(text) - max_len} chars)"


def _log_agent_step(step_num: int, messages: list) -> str | None:
    """Log an agent (LLM) step. Returns final content if no tool calls."""
    from langchain_core.messages import AIMessage

    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue

        # Check for tool calls
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc["name"]
                tool_args = tc.get("args", {})
                # Format args nicely, truncating long values
                formatted_args = {}
                for k, v in tool_args.items():
                    v_str = str(v)
                    formatted_args[k] = _truncate(v_str, 80)

                args_str = json.dumps(formatted_args, indent=2) if formatted_args else "(no args)"
                console.print(
                    Panel(
                        f"[bold yellow]🔧 Tool Call:[/bold yellow] [cyan]{tool_name}[/cyan]\n"
                        f"[dim]{args_str}[/dim]",
                        title=f"[bold]Step {step_num} · Agent → Tool[/bold]",
                        border_style="yellow",
                    )
                )
            return None  # Agent is calling tools, not done yet

        # No tool calls → final answer
        if msg.content:
            return msg.content

    return None


def _log_tool_step(step_num: int, messages: list) -> None:
    """Log a tool execution result."""
    from langchain_core.messages import ToolMessage

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        tool_name = msg.name or "unknown"
        content = str(msg.content)
        console.print(
            Panel(
                f"[bold green]✅ Result from:[/bold green] [cyan]{tool_name}[/cyan]\n"
                f"[dim]{_truncate(content, 500)}[/dim]",
                title=f"[bold]Step {step_num} · Tool Result[/bold]",
                border_style="green",
            )
        )


def _display_result(final_content: str | None, step_count: int) -> None:
    """Display the final result and done banner."""
    console.print()
    console.rule("[bold]Result[/bold]")
    console.print()

    if final_content:
        try:
            cleaned = final_content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                cleaned = "\n".join(lines)
            data = json.loads(cleaned)
            console.print_json(json.dumps(data, indent=2))
        except (json.JSONDecodeError, ValueError):
            console.print(Markdown(final_content))
    else:
        console.print("[yellow]No final response from agent.[/yellow]")

    console.print()
    console.print(
        Panel(
            f"[bold green]✅ Done — {step_count} steps[/bold green]",
            border_style="green",
        )
    )


# -----------------------------------------------------------------------
# Record mode: LLM-driven run with step recording
# -----------------------------------------------------------------------

async def _run_record_mode(url: str, task: str, settings: Settings) -> None:
    """Execute the LLM agent, record the path, optimise, and persist."""
    from browseragent.agent import run_agent_stream
    from browseragent.recorder import PathStore, StepRecorder
    from browseragent.replayer import _build_element_descriptor

    step_recorder = StepRecorder(url, task)
    last_snapshot_text: list[str] = [""]  # mutable container for closure

    def _on_tool_executed(
        tool_name: str,
        tool_args: dict,
        tool_result: str,
    ) -> None:
        # Build element descriptor from the most recent snapshot
        descriptor = _build_element_descriptor(
            tool_name, tool_args, last_snapshot_text[0]
        )
        step_recorder.record(tool_name, tool_args, tool_result, descriptor)

        # Keep track of latest snapshot for ref-descriptor lookups
        if tool_name == "browser_snapshot":
            last_snapshot_text[0] = tool_result

    console.rule("[bold]Agent Trace (Record Mode)[/bold]")
    console.print()

    step_num = 0
    final_content = None

    async for node_name, node_data in run_agent_stream(
        url, task, settings, record_callback=_on_tool_executed
    ):
        step_num += 1
        messages = node_data.get("messages", [])
        if not isinstance(messages, list):
            messages = [messages]

        if node_name == "agent":
            result = _log_agent_step(step_num, messages)
            if result is not None:
                final_content = result
        elif node_name == "tools":
            _log_tool_step(step_num, messages)

    _display_result(final_content, step_num)

    # --- Persist the recorded path ---
    if not step_recorder.steps:
        console.print("[yellow]No steps recorded — nothing to save.[/yellow]")
        return

    # Store the LLM's final formatted answer so replay can display it
    if final_content:
        step_recorder.set_final_response(final_content)

    execution_path = step_recorder.finish()
    store = PathStore(settings.paths_dir)

    # Optimise if enabled
    if settings.optimize_paths:
        console.print()
        console.print("[bold cyan]🔬 Optimising recorded path via LLM…[/bold cyan]")
        from browseragent.optimizer import optimize_path

        execution_path = await optimize_path(
            execution_path,
            model_name=settings.model_name,
            api_key=settings.openai_api_key,
        )
        original_count = len(step_recorder.steps)
        optimized_count = len(execution_path.steps)
        if optimized_count < original_count:
            console.print(
                f"[green]✂  Optimised: {original_count} → {optimized_count} steps[/green]"
            )
        else:
            console.print("[green]✅ Path is already optimal[/green]")

    filepath = store.save_path(execution_path)
    console.print(f"[bold green]💾 Path saved:[/bold green] {filepath}")


# -----------------------------------------------------------------------
# Replay mode: deterministic execution of stored path
# -----------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are a data extraction assistant. You are given a browser page snapshot
(in YAML-like format) and a user's task description.

Your job:
1. Analyse the current page snapshot carefully.
2. Extract the data requested in the task.
3. Return a well-formatted JSON object with the extracted data.

Important rules:
- Never invent data. Only return what is actually visible on the page.
- If the exact requested data is not found on the page, still return a JSON
  object describing what IS visible — for example the page title, any headings,
  status messages, error messages, or other relevant content you can see.
- Include a "status" field indicating whether the task data was found or not.
- Do NOT wrap the response in a tool call — just respond with the JSON.
"""


async def _extract_data_from_snapshot(
    snapshot_text: str,
    task: str,
    url: str,
    settings: "Settings",
) -> str:
    """Use a single LLM call to extract/format data from the current page snapshot."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.openai_api_key,
        temperature=0,
    )

    response = await llm.ainvoke(
        [
            SystemMessage(content=EXTRACTION_PROMPT),
            HumanMessage(
                content=(
                    f"Task: {task}\n"
                    f"Current page URL: {url}\n\n"
                    f"Current page snapshot:\n```\n{snapshot_text[:8000]}\n```"
                )
            ),
        ]
    )
    return response.content


async def _run_replay_mode(url: str, task: str, settings: Settings) -> None:
    """Replay a stored execution path, then extract fresh data from the page."""
    from browseragent.agent import open_mcp_session
    from browseragent.recorder import PathStore
    from browseragent.replayer import replay_path

    store = PathStore(settings.paths_dir)
    execution_path = store.load_path(url, task)

    console.print(
        Panel(
            f"[bold cyan]⏩ Replaying stored path[/bold cyan]\n"
            f"[dim]{len(execution_path.steps)} steps · "
            f"{'optimised' if execution_path.optimized else 'unoptimised'}[/dim]",
            border_style="cyan",
        )
    )
    console.rule("[bold]Replay Trace[/bold]")
    console.print()

    async with open_mcp_session(settings) as tools:
        replay_failed = False
        last_snapshot = ""
        step_count = 0

        async for step_index, tool_name, tool_result, success in replay_path(
            execution_path, tools
        ):
            step_count += 1
            if success:
                console.print(
                    Panel(
                        f"[bold green]✅[/bold green] [cyan]{tool_name}[/cyan]\n"
                        f"[dim]{_truncate(tool_result, 300)}[/dim]",
                        title=f"[bold]Replay Step {step_index}[/bold]",
                        border_style="blue",
                    )
                )
                # Track the latest snapshot for data extraction
                if tool_name == "browser_snapshot":
                    last_snapshot = tool_result
            else:
                console.print(
                    Panel(
                        f"[bold red]❌ FAILED:[/bold red] [cyan]{tool_name}[/cyan]\n"
                        f"[dim]{tool_result}[/dim]",
                        title=f"[bold]Replay Step {step_index}[/bold]",
                        border_style="red",
                    )
                )
                replay_failed = True
                break

        if replay_failed:
            console.print()
            console.print(
                "[bold yellow]⚠  Replay failed — falling back to LLM agent "
                "(re-recording)…[/bold yellow]"
            )
            console.print()
            await _run_record_mode(url, task, settings)
            return

        # --- Always take a fresh snapshot for data extraction ---
        # The recorded path may not include snapshot steps (optimizer may
        # have removed them), and we need the CURRENT page state anyway.
        console.print()
        console.print("[dim]Taking a fresh snapshot for data extraction…[/dim]")
        snapshot_tool = next(
            (t for t in tools if t.name == "browser_snapshot"), None
        )
        if snapshot_tool:
            from browseragent.replayer import extract_text_content
            raw_result = await snapshot_tool.ainvoke({})

            # Debug: show what type and format the tool returns
            console.print(f"[dim]Snapshot result type: {type(raw_result).__name__}[/dim]")

            last_snapshot = extract_text_content(raw_result)
            console.print(f"[dim]Extracted snapshot length: {len(last_snapshot)} chars[/dim]")
            if last_snapshot:
                console.print(f"[dim]Snapshot preview: {last_snapshot[:200]}…[/dim]")

        # --- Extract current data via a single LLM call ---
        console.print()
        console.print("[bold cyan]🔍 Extracting fresh data from current page…[/bold cyan]")

        if last_snapshot:
            extracted = await _extract_data_from_snapshot(
                last_snapshot, task, url, settings
            )
            _display_result(extracted, step_count)
        else:
            console.print("[yellow]No snapshot available for extraction.[/yellow]")


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------

async def async_main() -> None:
    args = parse_args()

    console.print(
        Panel(
            "[bold cyan]🤖 BrowserAgent[/bold cyan]\n"
            "[dim]Gemini + LangGraph + Playwright MCP[/dim]",
            border_style="cyan",
        )
    )

    # --- Load settings ---
    settings = Settings.from_env()
    settings.validate()

    # Headless / visible override from CLI flags
    if args.visible:
        settings.headless = False
    elif args.headless:
        settings.headless = True

    # --- Handle --clear-paths ---
    if args.clear_paths:
        from browseragent.recorder import PathStore

        store = PathStore(settings.paths_dir)
        count = store.clear_all()
        console.print(f"[bold green]🗑  Cleared {count} stored path(s).[/bold green]")
        return

    # --- Get URL and task ---
    url = args.url or Prompt.ask("[bold green]🌐 Target URL[/bold green]")
    task = args.task or Prompt.ask("[bold green]📝 Task / prompt[/bold green]")

    console.print()
    console.print(f"[bold]URL:[/bold]  {url}")
    console.print(f"[bold]Task:[/bold] {task}")
    console.print(f"[bold]Mode:[/bold] {'headless' if settings.headless else 'visible'}")
    console.print(f"[bold]Model:[/bold] {settings.model_name}")
    if settings.auth_username:
        console.print(f"[bold]Auth:[/bold]  {settings.auth_username} / {'*' * len(settings.auth_password)}")

    # --- Handle --show-path ---
    if args.show_path:
        from browseragent.recorder import PathStore

        store = PathStore(settings.paths_dir)
        if store.has_path(url, task):
            path = store.load_path(url, task)
            console.print()
            console.rule("[bold]Stored Path[/bold]")
            console.print_json(json.dumps(path.to_dict(), indent=2))
        else:
            console.print("[yellow]No stored path found for this URL + task.[/yellow]")
        return

    # --- Decide: replay or record ---
    console.print()

    from browseragent.recorder import PathStore

    store = PathStore(settings.paths_dir)
    use_replay = (
        settings.replay_enabled
        and not args.no_replay
        and store.has_path(url, task)
    )

    if use_replay:
        console.print("[bold cyan]📂 Stored path found — using replay mode[/bold cyan]")
        await _run_replay_mode(url, task, settings)
    else:
        if args.no_replay:
            console.print("[bold yellow]🔄 --no-replay flag set — forcing LLM agent[/bold yellow]")
        elif not store.has_path(url, task):
            console.print("[bold cyan]🆕 No stored path — running LLM agent (will record)[/bold cyan]")
        await _run_record_mode(url, task, settings)


def main() -> None:
    """Synchronous wrapper for the async entry point."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
