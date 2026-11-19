from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from mcp import Client
from mcp.server import MCPServer

from rag import ITSupportRAG

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

MODEL = "claude-haiku-5-5"

mcp_server = MCPServer(name="IT Helpdesk MCP Server", version="1.0.0")

TICKETS: dict[str, dict] = {
    "IT-101": {"ticket_id": "IT-101", "employee": "Rahul", "issue": "Laptop running slowly",
               "description": "System takes too long to start", "priority": "Medium",
               "status": "Open", "created_at": "2026-11-17 09:12"},
    "IT-102": {"ticket_id": "IT-102", "employee": "Priya", "issue": "VPN not connecting",
               "description": "Unable to connect to office VPN", "priority": "High",
               "status": "In Progress", "created_at": "2026-11-18 14:40"},
    "IT-103": {"ticket_id": "IT-103", "employee": "Arun", "issue": "Printer not working",
               "description": "Printer is not detected by laptop", "priority": "Low",
               "status": "Resolved", "created_at": "2026-11-16 11:05"},
}


def _next_ticket_id() -> str:
    numbers = [int(t.split("-")[1]) for t in TICKETS if t.startswith("IT-")]
    return f"IT-{max(numbers, default=100) + 1}"


@mcp_server.tool(name="create_support_ticket")
def create_support_ticket(employee: str, issue: str, description: str,
                          priority: str = "Medium") -> str:
    """Create a new IT support ticket. priority must be Low, Medium or High.
    Returns the created ticket as JSON."""
    priority = priority.capitalize()
    if priority not in {"Low", "Medium", "High"}:
        priority = "Medium"
    ticket_id = _next_ticket_id()
    TICKETS[ticket_id] = {
        "ticket_id": ticket_id,
        "employee": employee,
        "issue": issue,
        "description": description,
        "priority": priority,
        "status": "Open",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    return json.dumps(TICKETS[ticket_id])


@mcp_server.tool(name="check_ticket_status")
def check_ticket_status(ticket_id: str) -> str:
    """Retrieve the status and details of an existing IT support ticket by its ID
    (for example IT-101). Returns the ticket as JSON, or an error message."""
    ticket = TICKETS.get(ticket_id.strip().upper())
    if ticket is None:
        return json.dumps({"error": f"Ticket {ticket_id} not found."})
    return json.dumps(ticket)


# ---------------------------------------------------------------------------
# 2. Local tools (tool calling) + RAG
# ---------------------------------------------------------------------------

def get_current_datetime() -> str:
    now = datetime.now().astimezone()
    return now.strftime("%A, %d %B %Y, %H:%M:%S %Z")


LOCAL_TOOL_DEFS = [
    {
        "name": "search_it_docs",
        "description": (
            "Search the company's internal IT Support FAQ (vector database). Use this FIRST "
            "for any IT question: passwords, Wi-Fi, network, VPN, software installation, "
            "laptops, printers, email, security, backups, mobile devices."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The IT question to look up."}},
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "get_current_datetime",
        "description": "Get the current local date and time.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
]

# Anthropic server-side web search: runs on Anthropic's side, no local code needed.
WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": 3}

SYSTEM_PROMPT = """You are the IT Helpdesk Assistant for company employees.

How to help:
- For IT questions, call search_it_docs first and base your answer on the internal FAQ.
- If the FAQ does not cover the problem, or the user asks for more options, use web_search \
and say the extra steps came from the web.
- Give troubleshooting as short numbered steps.
- If the issue isn't resolved, or the user asks for one, offer to create a support ticket. \
Before calling create_support_ticket you need the employee's name, a short issue title, a \
description, and a priority (Low/Medium/High). Ask for whatever is missing; infer priority \
from impact if the user doesn't say.
- Use check_ticket_status when the user asks about an existing ticket (IDs look like IT-101).
- After creating or checking a ticket, show the ticket ID, issue, priority and status.
- Use get_current_datetime when the user asks about the date/time or you need it.
- Stay on IT support topics; be friendly and concise."""


@dataclass
class ToolEvent:
    """A record of one tool call, for display in the UI."""
    name: str
    input: dict
    output: str
    source: str  # "rag" | "mcp" | "local" | "web"


@dataclass
class AgentReply:
    text: str
    events: list[ToolEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 3. Agent: Claude decides which tools to call
# ---------------------------------------------------------------------------

class HelpdeskAgent:
    """Holds one conversation and runs the tool-calling loop for each user turn."""

    def __init__(self, rag: ITSupportRAG | None = None):
        self.client = anthropic.AsyncAnthropic()
        self.rag = rag or ITSupportRAG()
        # Full API history, including tool_use / tool_result / thinking blocks,
        # passed back unchanged on every request.
        self.messages: list[dict] = []
        self.session_id = uuid.uuid4().hex[:8]

    @staticmethod
    def _mcp_tool_defs(tools) -> list[dict]:
        return [
            {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
            for t in tools
        ]

    async def _run_tool(self, mcp: Client, mcp_names: set[str], name: str,
                        args: dict) -> tuple[str, str, bool]:
        """Execute one tool call. Returns (output, source, is_error)."""
        if name in mcp_names:
            result = await mcp.call_tool(name, args)
            text = "\n".join(getattr(c, "text", "") for c in result.content)
            return text, "mcp", bool(result.is_error)
        if name == "search_it_docs":
            return self.rag.search_as_text(args["query"]), "rag", False
        if name == "get_current_datetime":
            return get_current_datetime(), "local", False
        return f"Unknown tool: {name}", "local", True

    async def achat(self, user_text: str, max_iterations: int = 8) -> AgentReply:
        self.messages.append({"role": "user", "content": user_text})
        events: list[ToolEvent] = []

        async with Client(mcp_server) as mcp:
            mcp_tools = (await mcp.list_tools()).tools
            mcp_names = {t.name for t in mcp_tools}
            tools = LOCAL_TOOL_DEFS + self._mcp_tool_defs(mcp_tools) + [WEB_SEARCH_TOOL]

            for _ in range(max_iterations):
                response = await self.client.beta.messages.create(
                    model=MODEL,
                    max_tokens=16000,
                    system=SYSTEM_PROMPT,
                    tools=tools,
                    messages=self.messages,
                    output_config={"effort": "medium"},
                    # If a safety classifier declines, the server retries on a fallback model.
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                )
                self.messages.append({"role": "assistant", "content": response.content})

                # Record server-side web searches for the UI.
                for block in response.content:
                    if block.type == "server_tool_use" and block.name == "web_search":
                        events.append(ToolEvent("web_search", dict(block.input), "", "web"))

                if response.stop_reason == "refusal":
                    return AgentReply("Sorry, I can't help with that request.", events)
                if response.stop_reason == "pause_turn":
                    continue  # server tool paused; re-send to let it finish
                if response.stop_reason != "tool_use":
                    text = "".join(b.text for b in response.content if b.type == "text")
                    return AgentReply(text.strip(), events)

                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    try:
                        output, source, is_error = await self._run_tool(
                            mcp, mcp_names, block.name, dict(block.input))
                    except Exception as exc:  # report tool failures back to Claude
                        output, source, is_error = f"Tool error: {exc}", "local", True
                    events.append(ToolEvent(block.name, dict(block.input), output, source))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                        "is_error": is_error,
                    })
                # All results go back in a single user message.
                self.messages.append({"role": "user", "content": tool_results})

        return AgentReply("I couldn't finish that request. Please try rephrasing it.", events)

    def chat(self, user_text: str) -> AgentReply:
        """Synchronous wrapper (used by Streamlit and the CLI)."""
        return asyncio.run(self.achat(user_text))


def list_tickets() -> list[dict]:
    """All tickets currently in the in-memory store (for the UI sidebar)."""
    return list(TICKETS.values())


def main() -> None:
    agent = HelpdeskAgent()
    print("IT Helpdesk Assistant (type 'exit' to quit)")
    while True:
        user_text = input("\nYou: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue
        reply = agent.chat(user_text)
        for ev in reply.events:
            print(f"  [tool:{ev.source}] {ev.name}({ev.input})")
        print(f"\nAssistant: {reply.text}")


if __name__ == "__main__":
    main()
