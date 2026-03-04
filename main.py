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

    print(settings)

    # Headless / visible override from CLI flags
    if args.visible:
        settings.headless = False
    elif args.headless:
        settings.headless = True

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
    console.print()
    console.rule("[bold]Agent Trace[/bold]")
    console.print()

    # --- Stream agent steps ---
    from browseragent.agent import run_agent_stream

    step_num = 0
    final_content = None

    async for node_name, node_data in run_agent_stream(url, task, settings):
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

    # --- Display final result ---
    console.print()
    console.rule("[bold]Result[/bold]")
    console.print()

    if final_content:
        # Try to parse as JSON for pretty-printing
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
            f"[bold green]✅ Done — {step_num} steps[/bold green]",
            border_style="green",
        )
    )


def main() -> None:
    """Synchronous wrapper for the async entry point."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

