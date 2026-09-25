"""
The agent's "brain": a LangGraph StateGraph implementing the ReAct
(Reason -> Act -> Observe -> repeat) pattern.

We build this graph by hand instead of using LangGraph's `create_react_agent`
shortcut on purpose - the point of this project is to see every moving part.
Once this makes sense, the shortcut is a one-line drop-in replacement for
everything in `build_agent_graph()`.

See ../README.md for the full explanation of *why* this shape (state, nodes,
conditional edges) is the right way to model an agent.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from config import GROQ_API_KEY, MODEL_NAME, TEMPERATURE
from tools import TOOLS

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to three tools: a calculator, "
    "a Wikipedia lookup, and a live web search. "
    "Use a tool whenever it would make your answer more accurate than relying "
    "on your own memory - always use the calculator for math, wikipedia_lookup "
    "for stable background facts, and web_search for anything current or "
    "time-sensitive (news, prices, recent events). "
    "If you already know the answer confidently and no tool is needed, answer "
    "directly without calling a tool. "
    "When you have the information you need, give a clear, direct final answer."
)


class AgentState(TypedDict):
    """The state LangGraph threads through every node in the graph.

    `messages` is the entire conversation so far (system prompt, human
    messages, the AI's replies, and tool results). `Annotated[..., add_messages]`
    tells LangGraph *how* to merge a node's output into the existing state:
    instead of overwriting the list, `add_messages` appends new messages to it
    (and updates a message in place if one with a matching id is returned).
    This is what makes the state accumulate into a running transcript rather
    than each node needing to know and resend the full history itself.
    """

    messages: Annotated[list[AnyMessage], add_messages]


llm = ChatGroq(model=MODEL_NAME, temperature=TEMPERATURE, api_key=GROQ_API_KEY)

# bind_tools tells the LLM what tools exist (name, description, argument
# schema) so it can request a tool call in its response. Binding does NOT
# execute anything - it only makes tool-calling *possible*. Actually running
# the tool is ToolNode's job below.
llm_with_tools = llm.bind_tools(TOOLS)


def call_agent(state: AgentState) -> dict:
    """The 'agent' node: ask the LLM what to do next given the conversation so far.

    The response is either:
      - a normal AIMessage with `content` (a final answer), or
      - an AIMessage with `tool_calls` populated (the model wants a tool run
        before it can answer).
    Either way we just return it; the graph's routing logic decides what
    happens next.

    Tool-calling models occasionally generate a malformed tool call (e.g.
    missing a required argument) - this isn't a bug in our tools, it's the
    LLM provider rejecting the model's own output before it ever reaches
    ToolNode. This is an external API call at a real system boundary, so we
    catch failures here and turn them into a plain final answer instead of
    letting the whole program crash mid-conversation.
    """
    try:
        response = llm_with_tools.invoke(state["messages"])
    except Exception as exc:
        response = AIMessage(
            content=(
                "I ran into an error trying to use a tool for that "
                f"({exc.__class__.__name__}). Could you rephrase, or ask me "
                "something else?"
            )
        )
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """The router: decide whether to run tools or stop, based on the last message.

    This is the "Act" decision point of the ReAct loop, made in code rather
    than by another LLM call - if the model's last reply requested tool calls,
    go execute them; otherwise the model considers itself done, so end the graph run.
    """
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def build_agent_graph():
    """Wire the nodes and edges together into a runnable graph.

    Graph shape:

        START -> agent --(tool_calls requested?)--> tools -> agent -> ... -> END
                    \\_____________(no)_____________________________________/

    - "agent" node: the LLM reasons and either answers or requests tool calls.
    - "tools" node: LangGraph's prebuilt ToolNode executes whichever tools
      the last AI message requested, and appends their results as
      ToolMessages back into state["messages"].
    - The edge from "tools" back to "agent" is what makes this a *loop*: the
      model gets to see tool results and reason again, possibly calling more
      tools, until it produces a final answer with no tool_calls.
    """
    graph = StateGraph(AgentState)

    graph.add_node("agent", call_agent)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()
