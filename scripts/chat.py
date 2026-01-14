#!/usr/bin/env python3
"""
=============================================================================
Multi-Agent Banking Expert System - Interactive CLI
=============================================================================

An interactive command-line interface for querying the multi-agent banking
expert system. Similar to Claude Code or OpenCode, this tool provides:

- Persistent conversation history within a session
- Rich formatted output with agent attribution
- Session management (new session, clear history)
- Cache performance metrics display
- Markdown rendering in terminal

USAGE:
------
    uv run python scripts/chat.py                    # Default server
    uv run python scripts/chat.py --url http://...   # Custom server URL

COMMANDS:
---------
    /new      - Start a new conversation session
    /clear    - Clear conversation history (keep session)
    /stats    - Show cache performance statistics
    /agents   - Show available agent information
    /history  - Show conversation history
    /help     - Show this help message
    /exit     - Exit the CLI

=============================================================================
"""

import argparse
import uuid

import httpx

# Try to import rich for pretty output, fall back to basic if not available
try:
    from rich.console import Console
    from rich.table import Table
    from rich.theme import Theme

    RICH_AVAILABLE = True
    custom_theme = Theme(
        {
            "info": "cyan",
            "warning": "yellow",
            "error": "red bold",
            "success": "green",
            "agent": "magenta",
        }
    )
    console = Console(theme=custom_theme)
except ImportError:
    RICH_AVAILABLE = False
    console = None


class BankingCLI:
    """Interactive CLI for the Multi-Agent Banking Expert System."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session_id: str = str(uuid.uuid4())
        self.history: list[dict] = []
        self.client = httpx.Client(timeout=120.0)  # Long timeout for LLM

    def print_banner(self) -> None:
        """Print welcome banner."""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║     🏦 Multi-Agent Banking Expert System                     ║
║     ─────────────────────────────────────────────────────    ║
║     Type your banking questions below.                       ║
║     Use /help for available commands.                        ║
╚══════════════════════════════════════════════════════════════╝
        """
        if RICH_AVAILABLE:
            console.print(banner, style="bold cyan")
            console.print(f"Session: [dim]{self.session_id[:8]}...[/dim]")
            console.print(f"Server:  [dim]{self.base_url}[/dim]\n")
        else:
            print(banner)
            print(f"Session: {self.session_id[:8]}...")
            print(f"Server:  {self.base_url}\n")

    def print_help(self) -> None:
        """Print help message."""
        if RICH_AVAILABLE:
            table = Table(title="Available Commands", show_header=True)
            table.add_column("Command", style="cyan")
            table.add_column("Description")
            table.add_row("/new", "Start a new conversation session")
            table.add_row("/clear", "Clear conversation history (keep session)")
            table.add_row("/stats", "Show cache performance statistics")
            table.add_row("/agents", "Show which agents were used")
            table.add_row("/history", "Show conversation history")
            table.add_row("/health", "Check server health")
            table.add_row("/help", "Show this help message")
            table.add_row("/exit, /quit", "Exit the CLI")
            console.print(table)
        else:
            print("\nAvailable Commands:")
            print("  /new     - Start a new conversation session")
            print("  /clear   - Clear conversation history")
            print("  /stats   - Show cache performance statistics")
            print("  /agents  - Show which agents were used")
            print("  /history - Show conversation history")
            print("  /health  - Check server health")
            print("  /help    - Show this help message")
            print("  /exit    - Exit the CLI\n")

    def new_session(self) -> None:
        """Start a new session."""
        self.session_id = str(uuid.uuid4())
        self.history = []
        if RICH_AVAILABLE:
            console.print(f"✨ New session started: [cyan]{self.session_id[:8]}...[/cyan]")
        else:
            print(f"✨ New session started: {self.session_id[:8]}...")

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []
        if RICH_AVAILABLE:
            console.print("🗑️  Conversation history cleared.")
        else:
            print("🗑️  Conversation history cleared.")

    def show_history(self) -> None:
        """Display conversation history."""
        if not self.history:
            print("No conversation history yet.")
            return

        if RICH_AVAILABLE:
            for i, turn in enumerate(self.history, 1):
                console.print(f"\n[bold]Turn {i}[/bold]")
                console.print(f"[cyan]You:[/cyan] {turn['query'][:100]}...")
                console.print(f"[green]Assistant:[/green] {turn['response'][:100]}...")
        else:
            for i, turn in enumerate(self.history, 1):
                print(f"\nTurn {i}")
                print(f"You: {turn['query'][:100]}...")
                print(f"Assistant: {turn['response'][:100]}...")

    def check_health(self) -> bool:
        """Check server health."""
        try:
            response = self.client.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                if RICH_AVAILABLE:
                    console.print(
                        f"✅ Server healthy (v{data.get('version', '?')})", style="success"
                    )
                else:
                    print(f"✅ Server healthy (v{data.get('version', '?')})")
                return True
            else:
                if RICH_AVAILABLE:
                    console.print(f"❌ Server returned {response.status_code}", style="error")
                else:
                    print(f"❌ Server returned {response.status_code}")
                return False
        except Exception as e:
            if RICH_AVAILABLE:
                console.print(f"❌ Cannot connect to server: {e}", style="error")
            else:
                print(f"❌ Cannot connect to server: {e}")
            return False

    def show_stats(self) -> None:
        """Show cache statistics."""
        try:
            response = self.client.get(f"{self.base_url}/cache/stats")
            if response.status_code == 200:
                data = response.json()
                if RICH_AVAILABLE:
                    table = Table(title="Cache Performance Statistics")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Value", style="green")
                    table.add_row("Status", data.get("status", "N/A"))
                    table.add_row("Grade", data.get("grade", "N/A"))
                    table.add_row("Interpretation", data.get("interpretation", "N/A"))

                    if "total_requests" in data:
                        table.add_row("Total Requests", str(data.get("total_requests", 0)))
                        table.add_row(
                            "Cold Cache TTFT", f"{data.get('cold_cache_ttft_seconds', 0):.2f}s"
                        )
                        table.add_row(
                            "Avg Warm TTFT", f"{data.get('avg_warm_ttft_seconds', 0):.2f}s"
                        )
                        table.add_row(
                            "Cache Hit Rate", f"{data.get('inferred_cache_hit_rate', 0):.1%}"
                        )
                        table.add_row(
                            "Prefix Aligned", "✅" if data.get("prefix_alignment_ok") else "❌"
                        )

                    console.print(table)
                else:
                    print("\nCache Statistics:")
                    for k, v in data.items():
                        print(f"  {k}: {v}")
        except Exception as e:
            if RICH_AVAILABLE:
                console.print(f"❌ Error fetching stats: {e}", style="error")
            else:
                print(f"❌ Error fetching stats: {e}")

    def query_stream(self, user_input: str) -> dict | None:
        """Send a streaming query to the API and display tokens in real-time."""
        import json

        payload = {
            "query": user_input,
            "session_id": self.session_id,
        }

        try:
            # Start streaming request
            url = f"{self.base_url}/api/v1/query/stream"

            if RICH_AVAILABLE:
                console.print("\n[dim]⏳ Waiting for first token...[/dim]", end="")
            else:
                print("\n⏳ Waiting for first token...", end="", flush=True)

            full_response = ""
            ttft = None
            agents = []
            total_time = 0.0

            with httpx.stream("POST", url, json=payload, timeout=120.0) as response:
                if response.status_code != 200:
                    if RICH_AVAILABLE:
                        console.print(f"\n❌ API error: {response.status_code}", style="error")
                    else:
                        print(f"\n❌ API error: {response.status_code}")
                    return None

                # Parse SSE stream
                # Use Live display for rich markdown rendering during streaming
                live = None

                try:
                    for line in response.iter_lines():
                        if not line:
                            continue

                        # Parse event type
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                            continue

                        # Parse data
                        if line.startswith("data:"):
                            data_str = line[5:].strip()
                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            if event_type == "metadata":
                                ttft = data.get("ttft", 0)
                                agents = data.get("agents", [])

                                # Clear "waiting" message and show TTFT
                                if RICH_AVAILABLE:
                                    # Move cursor back and clear line
                                    console.print("\r" + " " * 50, end="\r")
                                    console.print(
                                        f"[green]⚡ TTFT: {ttft:.3f}s[/green] | Agent: [magenta]{', '.join(agents)}[/magenta]"
                                    )
                                    console.print("\n[bold]Response:[/bold]")

                                    # Start Live display for markdown rendering
                                    from rich.live import Live
                                    from rich.markdown import Markdown

                                    live = Live(
                                        Markdown(""), console=console, refresh_per_second=10
                                    )
                                    live.start()
                                else:
                                    print(f"\r⚡ TTFT: {ttft:.3f}s | Agent: {', '.join(agents)}")
                                    print("\nResponse:")

                            elif event_type == "token":
                                content = data.get("content", "")
                                full_response += content
                                # Update Live display or print token
                                if RICH_AVAILABLE and live:
                                    live.update(Markdown(full_response))
                                elif RICH_AVAILABLE:
                                    # Fallback if live failed to start (rare)
                                    console.print(content, end="", markup=False)
                                else:
                                    print(content, end="", flush=True)

                            elif event_type == "done":
                                if live:
                                    live.stop()
                                    live = None

                                total_time = data.get("total_time", 0)
                                if RICH_AVAILABLE:
                                    console.print(f"\n\n[dim]Total time: {total_time:.2f}s[/dim]")
                                else:
                                    print(f"\n\nTotal time: {total_time:.2f}s")

                            elif event_type == "error":
                                if live:
                                    live.stop()
                                    live = None

                                error = data.get("error", "Unknown error")
                                if RICH_AVAILABLE:
                                    console.print(f"\n❌ Error: {error}", style="error")
                                else:
                                    print(f"\n❌ Error: {error}")
                                return None
                finally:
                    # Ensure live is stopped if loop exits abnormally
                    if live:
                        live.stop()

            # Add to history
            self.history.append(
                {
                    "query": user_input,
                    "response": full_response,
                    "agents": agents,
                    "elapsed": total_time,
                    "ttft": ttft,
                }
            )

            return {
                "response": full_response,
                "agents_used": agents,
                "ttft_seconds": ttft,
                "_elapsed": total_time,
                "compliance_passed": True,
            }

        except httpx.TimeoutException:
            if RICH_AVAILABLE:
                console.print("\n❌ Request timed out (120s limit)", style="error")
            else:
                print("\n❌ Request timed out (120s limit)")
            return None
        except Exception as e:
            if RICH_AVAILABLE:
                console.print(f"\n❌ Error: {e}", style="error")
            else:
                print(f"\n❌ Error: {e}")
            return None

    def query(self, user_input: str) -> dict | None:
        """Send a query - uses streaming by default."""
        return self.query_stream(user_input)

    def display_response(self, data: dict) -> None:
        """Display response summary (streaming already shows the content)."""
        # Streaming already displays the response, just show summary if needed
        pass

    def run(self) -> None:
        """Main REPL loop."""
        self.print_banner()

        # Check server health on startup
        if not self.check_health():
            if RICH_AVAILABLE:
                console.print("\n[warning]⚠️  Server not responding. Start it with:[/warning]")
                console.print("[dim]   uv run uvicorn src.main:app --port 8000[/dim]\n")
            else:
                print("\n⚠️  Server not responding. Start it with:")
                print("   uv run uvicorn src.main:app --port 8000\n")

        while True:
            try:
                # Get user input
                if RICH_AVAILABLE:
                    user_input = console.input("\n[bold cyan]You>[/bold cyan] ").strip()
                else:
                    user_input = input("\nYou> ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    cmd = user_input.lower()

                    if cmd in ("/exit", "/quit", "/q"):
                        if RICH_AVAILABLE:
                            console.print("👋 Goodbye!", style="info")
                        else:
                            print("👋 Goodbye!")
                        break

                    elif cmd == "/help":
                        self.print_help()

                    elif cmd == "/new":
                        self.new_session()

                    elif cmd == "/clear":
                        self.clear_history()

                    elif cmd == "/history":
                        self.show_history()

                    elif cmd == "/stats":
                        self.show_stats()

                    elif cmd == "/health":
                        self.check_health()

                    elif cmd == "/agents":
                        if self.history:
                            last = self.history[-1]
                            agents = last.get("agents", [])
                            if RICH_AVAILABLE:
                                console.print(
                                    f"Last query used: [magenta]{', '.join(agents)}[/magenta]"
                                )
                            else:
                                print(f"Last query used: {', '.join(agents)}")
                        else:
                            print("No queries sent yet.")

                    else:
                        if RICH_AVAILABLE:
                            console.print(
                                f"Unknown command: {cmd}. Use /help for available commands.",
                                style="warning",
                            )
                        else:
                            print(f"Unknown command: {cmd}. Use /help for available commands.")

                    continue

                # Send query to API
                result = self.query(user_input)
                if result:
                    self.display_response(result)

            except KeyboardInterrupt:
                if RICH_AVAILABLE:
                    console.print("\n👋 Goodbye!", style="info")
                else:
                    print("\n👋 Goodbye!")
                break
            except EOFError:
                break

        self.client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive CLI for the Multi-Agent Banking Expert System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/chat.py
  uv run python scripts/chat.py --url http://192.168.1.100:8000
        """,
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the API server (default: http://localhost:8000)",
    )

    args = parser.parse_args()

    cli = BankingCLI(base_url=args.url)
    cli.run()


if __name__ == "__main__":
    main()
