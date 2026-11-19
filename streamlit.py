import json

import streamlit as st

from main import HelpdeskAgent, list_tickets
from rag import ITSupportRAG

st.set_page_config(page_title="IT Helpdesk Assistant", page_icon="🛠️", layout="wide")

SOURCE_LABELS = {
    "rag": "IT docs (RAG)",
    "mcp": "MCP ticket tool",
    "web": "Web search",
    "local": "Local tool",
}
STATUS_ICONS = {"Open": "🟡", "In Progress": "🔵", "Resolved": "🟢"}


@st.cache_resource(show_spinner="Loading IT documentation into the vector database...")
def get_rag() -> ITSupportRAG:
    return ITSupportRAG()


def render_events(events) -> None:
    """Show which tools the assistant used, with ticket details where relevant."""
    for ev in events:
        label = f"{SOURCE_LABELS.get(ev.source, ev.source)}: `{ev.name}`"
        with st.expander(label, expanded=ev.source == "mcp"):
            if ev.input:
                st.caption("Input")
                st.json(ev.input)
            if ev.source == "mcp":
                try:
                    ticket = json.loads(ev.output)
                except json.JSONDecodeError:
                    ticket = None
                if isinstance(ticket, dict) and "ticket_id" in ticket:
                    cols = st.columns(4)
                    cols[0].metric("Ticket", ticket["ticket_id"])
                    cols[1].metric("Priority", ticket["priority"])
                    cols[2].metric("Status", ticket["status"])
                    cols[3].metric("Employee", ticket["employee"])
                    st.write(f"**{ticket['issue']}**: {ticket['description']}")
                    continue
            if ev.output:
                st.caption("Result")
                st.text(ev.output)


if "agent" not in st.session_state:
    st.session_state.agent = HelpdeskAgent(rag=get_rag())
    st.session_state.chat = []  # [{"role", "content", "events"}]

with st.sidebar:
    st.header("🎫 Support tickets")
    for t in list_tickets():
        st.markdown(
            f"{STATUS_ICONS.get(t['status'], '⚪')} **{t['ticket_id']}**: {t['issue']}  \n"
            f"<small>{t['employee']} · {t['priority']} · {t['status']}</small>",
            unsafe_allow_html=True,
        )
    st.divider()
    st.header("Try asking")
    st.markdown(
        "- How do I reset my password?\n"
        "- My Wi-Fi keeps disconnecting\n"
        "- Create a ticket: my laptop won't charge\n"
        "- What's the status of IT-102?\n"
        "- What's today's date?"
    )
    st.divider()
   
    

st.title("🛠️ IT Helpdesk Assistant")
st.caption("Ask about passwords, Wi-Fi, software, laptops and more. I can also create and check support tickets.")

if not st.session_state.chat:
    with st.chat_message("assistant"):
        st.write("Hi! I'm your IT Helpdesk Assistant. What can I help you with today?")

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("events"):
            render_events(msg["events"])

if prompt := st.chat_input("Describe your IT issue..."):
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = st.session_state.agent.chat(prompt)
                text, events = reply.text, reply.events
            except Exception as exc:
                text, events = f"⚠️ Something went wrong: {exc}", []
        st.markdown(text)
        render_events(events)

    st.session_state.chat.append({"role": "assistant", "content": text, "events": events})
    st.rerun() 
