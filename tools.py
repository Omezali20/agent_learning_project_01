"""
Tools the agent can choose to call.

IMPORTANT teaching point: the LLM never sees this Python code. It only sees each
tool's *name* and *docstring* (LangChain turns the docstring into the tool's
"description" field, which gets sent to the model). The model decides which tool
to call, and with what arguments, purely from that name + description text.
Vague or missing docstrings are the #1 cause of an agent picking the wrong tool
or never calling one it should - so treat these docstrings as prompts, not comments.
"""

import ast
import operator

import wikipedia
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

import config  # noqa: F401  (importing runs load_dotenv() + validates keys are set)

# ---------------------------------------------------------------------------
# Tool 1: Calculator
# ---------------------------------------------------------------------------
# We deliberately do NOT use Python's built-in eval() here. eval() executes
# arbitrary code - if a user (or a prompt-injected tool result) ever got a
# string like "__import__('os').system('...')" into this function, eval()
# would run it. Instead we parse the expression into an AST and only allow
# a fixed whitelist of arithmetic operators, so the tool can compute math and
# nothing else. This is a small example of a much bigger rule: any tool an
# agent can call is a real capability with real blast radius, so scope it down
# to exactly what it needs to do.

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression component: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the numeric result.

    Use this for ANY math - addition, subtraction, multiplication, division,
    powers, percentages written as arithmetic, etc. Do not compute math
    yourself; call this tool instead, since LLMs are unreliable at arithmetic.

    Args:
        expression: A plain arithmetic expression, e.g. "12 * (4 + 3)" or
            "340 * 0.15". Supports + - * / % ** and parentheses only.
    """
    try:
        tree = ast.parse(expression, mode="eval").body
        result = _safe_eval(tree)
        return str(result)
    except Exception as exc:
        return f"Could not evaluate '{expression}': {exc}"


# ---------------------------------------------------------------------------
# Tool 2: Wikipedia lookup
# ---------------------------------------------------------------------------
# Written directly against the `wikipedia` PyPI package rather than pulling in
# langchain-community's WikipediaQueryRun: langchain-community was sunset and
# archived in 2026, so we avoid depending on it. This also shows that a tool
# is just a regular Python function - no special integration is required to
# make one.


@tool
def wikipedia_lookup(query: str) -> str:
    """Look up a concise factual summary of a topic, person, place, or event on Wikipedia.

    Best for stable background knowledge (biographies, historical facts,
    definitions, geography). Not useful for anything that changes day to day -
    use web_search for that instead.

    Args:
        query: A short search term, e.g. "Marie Curie" or "Great Barrier Reef".
    """
    try:
        return wikipedia.summary(query, sentences=3, auto_suggest=True)
    except wikipedia.exceptions.DisambiguationError as exc:
        options = ", ".join(exc.options[:5])
        return f"'{query}' is ambiguous. Try one of: {options}"
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{query}'."
    except Exception as exc:
        return f"Wikipedia lookup failed: {exc}"


# ---------------------------------------------------------------------------
# Tool 3: Live web search
# ---------------------------------------------------------------------------
# TavilySearch is a search engine built specifically to return LLM-friendly
# results (clean text, not raw HTML). Unlike the two tools above, this one is
# a ready-made LangChain tool object, not a function we wrote - it already has
# a name and description built in, so we can use it as-is. It reads the
# TAVILY_API_KEY environment variable automatically (already loaded by config).
#
# Its default registered name is "tavily_search" - we rename it to
# "web_search" so it reads as a capability ("search the web") rather than a
# vendor name, and so it matches what agent.py's system prompt calls it.
# This name (not the Python variable name) is what the LLM actually sees.

web_search = TavilySearch(max_results=3, topic="general")
web_search.name = "web_search"


# The full toolbox handed to the agent. Order doesn't affect behavior - the
# model picks whichever tool's description best matches what it needs.
TOOLS = [calculator, wikipedia_lookup, web_search]
