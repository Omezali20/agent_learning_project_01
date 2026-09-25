"""
Terminal chat loop for the ReAct agent.

Uses app.stream(..., stream_mode="values") instead of app.invoke() so we can
print each intermediate step (tool calls, tool results) as they happen, not
just the final answer. Watching these steps is the whole point of this
project - it's how you build intuition for what the agent is actually doing.

Run with:  python main.py   (from inside this folder)
"""

import sys

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent import SYSTEM_PROMPT, build_agent_graph

# Windows terminals often default to a legacy code page (cp1252) that can't
# encode every Unicode character an LLM might output (curly quotes, em dashes,
# narrow spaces, etc.), which crashes print() mid-conversation. Force UTF-8
# on stdout so that never happens.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    app = build_agent_graph()
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    print("=== ReAct Agent (Groq + LangGraph) ===")
    print("Tools available: calculator, wikipedia_lookup, web_search")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        messages.append(HumanMessage(content=user_input))

        for step in app.stream({"messages": messages}, stream_mode="values"):
            last_message = step["messages"][-1]

            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                for call in last_message.tool_calls:
                    print(f"  -> calling tool: {call['name']}({call['args']})")
            elif isinstance(last_message, ToolMessage):
                preview = str(last_message.content)[:200]
                print(f"  <- {last_message.name} returned: {preview}")
            elif isinstance(last_message, AIMessage) and last_message.content:
                print(f"\nAgent: {last_message.content}\n")

            messages = step["messages"]


if __name__ == "__main__":
    main()
