"""
Streamlit chat front-end for the ReAct agent defined in agent.py.

This is intentionally a *thin* UI layer: all the agent logic (state, nodes,
edges) still lives in agent.py exactly as before. This file only has to
solve UI-specific problems: rendering a chat, showing tool-call steps live
while the agent works, and keeping conversation history across reruns
(Streamlit re-executes this whole script top-to-bottom on every interaction,
so anything that must survive a rerun goes in st.session_state).

Run with:  streamlit run streamlit_app.py   (from inside this folder)
"""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

st.set_page_config(page_title="ReAct Agent", page_icon="🤖", layout="centered")

# config.py raises EnvironmentError if a required API key is missing. In a
# terminal that's a fine crash; in a web app it's a wall of traceback the
# user didn't ask for, so we catch it once here and show a clean message.
try:
    from agent import SYSTEM_PROMPT, build_agent_graph
except EnvironmentError as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()


@st.cache_resource
def get_agent_app():
    """Build the compiled LangGraph app once per Streamlit server process.

    Rebuilding it on every rerun would recreate the ChatGroq client and tool
    objects on every keystroke/interaction, which is wasteful and unnecessary
    since none of it depends on user input.
    """
    return build_agent_graph()


def init_session_state():
    if "lc_messages" not in st.session_state:
        # The full conversation history handed to the graph on every turn.
        # Starts with just the system prompt; grows with each Human/AI/Tool
        # message as the conversation progresses.
        st.session_state.lc_messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if "turns" not in st.session_state:
        # A UI-friendly summary per user turn, used only for rendering
        # history back out after a rerun: {"user", "steps", "answer"}.
        st.session_state.turns = []


def render_steps(steps: list[dict]) -> None:
    """Render a past turn's recorded tool-call trace inside an expander."""
    if not steps:
        return
    n_calls = sum(1 for s in steps if s["type"] == "call")
    with st.expander(f"🔧 Reasoning ({n_calls} tool call{'s' if n_calls != 1 else ''})"):
        for step in steps:
            if step["type"] == "call":
                st.markdown(f"**Called `{step['name']}`** with `{step['args']}`")
            else:
                st.markdown(f"**`{step['name']}`** returned:")
                st.code(step["content"], language=None)


def run_agent_turn(user_input: str) -> None:
    """Run one full ReAct loop for a new user message, streaming steps live."""
    app = get_agent_app()
    st.session_state.lc_messages.append(HumanMessage(content=user_input))

    steps: list[dict] = []
    final_answer = ""

    with st.status("Agent is thinking...", expanded=True) as status:
        for update in app.stream({"messages": st.session_state.lc_messages}, stream_mode="values"):
            last_message = update["messages"][-1]

            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                for call in last_message.tool_calls:
                    st.write(f"🔧 Calling **{call['name']}** with `{call['args']}`")
                    steps.append({"type": "call", "name": call["name"], "args": call["args"]})
            elif isinstance(last_message, ToolMessage):
                preview = str(last_message.content)[:500]
                st.write(f"↩️ **{last_message.name}** returned:")
                st.code(preview, language=None)
                steps.append({"type": "result", "name": last_message.name, "content": preview})
            elif isinstance(last_message, AIMessage) and last_message.content:
                final_answer = last_message.content

            # Keep the full message history in sync with what the graph has
            # produced so far, so the next turn continues from here.
            st.session_state.lc_messages = update["messages"]

        status.update(label="Done", state="complete", expanded=False)

    st.write(final_answer)
    st.session_state.turns.append({"user": user_input, "steps": steps, "answer": final_answer})


def main() -> None:
    init_session_state()

    st.title("🤖 ReAct Agent")
    st.caption("Groq (openai/gpt-oss-120b) + LangGraph · calculator, Wikipedia, web search")

    with st.sidebar:
        st.header("About")
        st.write(
            "This agent decides on its own whether to answer directly or "
            "call a tool first. Watch the **Reasoning** section under each "
            "answer to see which tools it used."
        )
        st.subheader("Tools")
        st.markdown("- 🧮 **calculator** — arithmetic\n- 📚 **wikipedia_lookup** — stable facts\n- 🌐 **web_search** — current info")
        st.divider()
        if st.button("🗑️ Clear conversation"):
            st.session_state.lc_messages = [SystemMessage(content=SYSTEM_PROMPT)]
            st.session_state.turns = []
            st.rerun()
        st.caption("Chat history resets when you refresh the page or restart the app — this project doesn't persist memory to disk (see README, section 7).")

    # Replay history from previous reruns.
    for turn in st.session_state.turns:
        with st.chat_message("user"):
            st.write(turn["user"])
        with st.chat_message("assistant"):
            render_steps(turn["steps"])
            st.write(turn["answer"])

    # Handle a new message for this run.
    prompt = st.chat_input("Ask me anything...")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            run_agent_turn(prompt)


if __name__ == "__main__":
    main()
