# Project 1 — A ReAct Tool-Using Agent

Your first agentic AI project: an LLM (served by Groq) that can decide, on its
own, when to do math with a calculator, when to look something up on
Wikipedia, and when to search the live web — instead of just guessing from
memory. This is the simplest possible "real" agent, and almost every more
advanced agent pattern (multi-agent systems, RAG agents, coding agents) is
this same loop with more nodes added.

---

## 1. What you'll learn

- The **ReAct pattern** (Reason → Act → Observe → repeat) that underlies most tool-using agents.
- Why **LangGraph** models an agent as an explicit state machine instead of a hidden loop.
- How an LLM "calls a tool" (it doesn't — it *requests* one, your code executes it).
- Why a tool's docstring is effectively a prompt, not documentation.
- Basic secret management (`.env`) and safe tool design (why we didn't use `eval()`).
- How to watch an agent's reasoning step-by-step instead of only seeing the final answer.

---

## 2. Why an agent, and why this shape?

A plain LLM call is: **prompt in → text out**. It cannot do anything the
model doesn't already "know" from training, and it's often unreliable at
tasks like arithmetic or knowing today's date.

An **agent** adds a loop around the LLM call: after the model responds, your
code checks *"did it ask to use a tool?"* If yes, you run the tool, feed the
result back in, and ask the model again. This repeats until the model is
satisfied and gives a final answer. That loop — not anything mystical — is
what "agentic" means here.

### Why LangGraph instead of plain LangChain

Older LangChain (`AgentExecutor`) hides this loop inside a single function
call — convenient, but you can't easily see or control what's happening
between steps. **LangGraph** makes the loop an explicit, inspectable graph:
you define the *state* that flows through it, the *nodes* that transform that
state, and the *edges* (including conditional ones) that decide what runs
next. This buys you three things that matter a lot as agents get more
complex (later projects will lean on all three):

1. **Transparency** — you can print/log the state at every node.
2. **Control** — you can insert extra steps (validation, human approval, retries) anywhere in the graph.
3. **Resumability** — the graph can be paused, checkpointed, and resumed (used heavily once we add memory in a later project).

This is also architecturally close to how production agent tools are built
(this very CLI you're using, Claude Code, is built on a similar explicit
agent-loop idea) — so the mental model transfers.

### The graph, visually

```
        ┌─────────┐
 START ─▶  agent   │──── no tool_calls? ────▶ END
        └────┬────┘
             │ tool_calls present
             ▼
        ┌─────────┐
        │  tools   │
        └────┬────┘
             │
             └──────────────▶ back to agent
```

- **`agent` node** — calls the LLM (`ChatGroq`, bound to the 3 tools) with the
  full conversation so far. The model replies either with a normal answer, or
  with `tool_calls` (a structured request like `calculator(expression="340*0.15")`).
- **`should_continue` (the diamond above)** — plain Python, not another LLM
  call: *"does the last message have `tool_calls`?"* If yes → route to
  `tools`. If no → route to `END`.
- **`tools` node** — LangGraph's prebuilt `ToolNode`. It reads the pending
  `tool_calls`, actually executes the matching Python function(s), and
  appends the result(s) as `ToolMessage`s.
- **`tools → agent` edge** — always taken. This is what makes it a loop: the
  model sees the tool's result and reasons again, possibly requesting another
  tool call (e.g. "look up population" → "now calculate 15% of it"), until it
  has enough to answer directly.

### The state

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

Every node receives the current `state` and returns a partial update.
`Annotated[..., add_messages]` tells LangGraph *how* to merge that update:
**append** new messages to the list (rather than replacing it). This is the
entire "memory" mechanism within a single run — the whole transcript (system
prompt, your question, the model's tool requests, the tools' answers, and the
final reply) accumulates in one place, and every node sees the full history
each time it runs.

> Note: this is memory *within one conversation/run only*. Close the program
> and it's gone — there's no persistence to disk yet. That's deliberate for
> Project 1 (see "What's deliberately left out" below).

### Walking through a real example

Question: *"What is the population of Japan according to Wikipedia, and what is 15% of that number?"*

1. `agent`: model reads the question, decides it needs a fact first → returns `tool_calls=[wikipedia_lookup(query="Japan population")]`.
2. `should_continue`: tool_calls present → route to `tools`.
3. `tools`: runs `wikipedia_lookup`, gets back a summary containing a population figure → appended as a `ToolMessage`.
4. `agent`: model reads the summary, now needs to compute 15% of that number → returns `tool_calls=[calculator(expression="123000000 * 0.15")]`.
5. `should_continue`: tool_calls present → route to `tools` again.
6. `tools`: runs `calculator`, returns the result → appended as a `ToolMessage`.
7. `agent`: model now has both facts, composes a final answer with no `tool_calls`.
8. `should_continue`: no tool_calls → route to `END`.

Run this exact question in the notebook (`notebook.ipynb`, Example 5) or the CLI to see it happen live.

---

## 3. Project structure

```
LLM Practical Implementation using Python/     <- repo root
├── .env                          # secrets (GROQ_API_KEY, TAVILY_API_KEY) — gitignored
├── .gitignore
├── requirements.txt              # shared across all numbered projects
├── LLM Intro.ipynb               # your earlier notebook
└── 01_react_agent_with_tools/
    ├── config.py                 # loads + validates secrets, model settings
    ├── tools.py                  # calculator, wikipedia_lookup, web_search
    ├── agent.py                  # the LangGraph state machine (the core of this project)
    ├── main.py                   # terminal chat loop
    ├── streamlit_app.py          # web chat UI (same agent, browser front-end)
    ├── list_groq_models.py       # utility: list models available to your Groq key
    ├── notebook.ipynb            # same project, step-by-step for experimentation
    └── README.md                 # this file
```

Why split into these specific files (a pattern worth reusing in later
projects): **config** (secrets/settings), **tools** (capabilities), **agent**
(reasoning/orchestration), and **entry point** (how a human interacts with
it) are four genuinely different concerns. Keeping them in separate files
means you can, for example, swap in a new tool without touching the graph
logic, or swap Groq for another provider by editing only `config.py`/`agent.py`.

---

## 4. Setup

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/keys) (you already have one)
- A free [Tavily API key](https://app.tavily.com) (you already added one)

### Steps

```bash
# 1. From the repo root, create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Confirm .env at the repo root contains both keys:
#    GROQ_API_KEY = "..."
#    TAVILY_API_KEY = "..."
```

**Never commit `.env`.** It's already listed in `.gitignore`. If you ever
push this folder to GitHub, double check `git status` doesn't show `.env`
before committing — a leaked key on a public repo will get scraped and
abused within minutes, even if you delete it in a later commit (git history
keeps it).

---

## 5. How to run

**Terminal chat:**
```bash
cd 01_react_agent_with_tools
python main.py
```
Try asking it something needing math, something needing Wikipedia, and
something needing current news, and watch the printed `-> calling tool: ...`
/ `<- ... returned: ...` lines.

**Notebook (recommended for learning):**
Open `notebook.ipynb` and run the cells top to bottom. It builds the same
agent piece by piece with explanations, including printing the graph as a
Mermaid diagram you can visualize at https://mermaid.live.

**Web UI (Streamlit):**
```bash
cd 01_react_agent_with_tools
streamlit run streamlit_app.py
```
Opens a chat interface in your browser. Every answer has a collapsible
**🔧 Reasoning** section showing exactly which tools were called, with what
arguments, and what they returned — the same information `main.py` prints to
the terminal, just rendered as UI instead of text. See section 10 below for
how it's wired up.

---

## 6. Design decisions worth understanding

- **No `eval()` for the calculator.** `eval()` runs arbitrary Python — a
  genuinely dangerous thing to expose to text an LLM (or a tool result, or a
  user) controls. `tools.py` parses the expression into an AST and only
  allows a fixed set of arithmetic operators. General rule: every tool you
  give an agent is a real capability with real blast radius — scope it to
  exactly what's needed, nothing more.
- **Custom Wikipedia tool instead of a prebuilt LangChain integration.**
  `langchain-community` (which used to host a `WikipediaQueryRun` tool) was
  sunset/archived in 2026. Rather than depend on an unmaintained package, we
  wrapped the plain `wikipedia` PyPI package in a few lines with `@tool`.
  This also demonstrates the real lesson: **any Python function can be an
  agent tool** — prebuilt integrations are just a convenience, not a
  requirement.
- **Model: `openai/gpt-oss-120b` on Groq, not `llama-3.3-70b-versatile`.** Groq's
  hosted model catalog changes over time — Llama models that used to be
  available were removed from Groq's lineup and calling them now returns a
  `404 model_not_found` error. `openai/gpt-oss-120b` (OpenAI's open-weight
  model) is what's currently available and reliably supports tool calling.
  If this ever breaks again, run `python list_groq_models.py` to see what
  your key currently has access to and update `MODEL_NAME` in `config.py`.
- **`temperature=0`.** While learning, you want the agent's tool-use
  decisions to be as consistent and repeatable as possible so you can reason
  about *why* it did something. Higher temperature adds randomness that would
  make debugging much harder at this stage.
- **`try/except` around the LLM call in `call_agent`.** While testing this
  project, the model occasionally generated a tool call with a missing/wrong
  argument (e.g. calling `web_search` without a `query`). Groq's API rejects
  that malformed request with a `400` error *before* our tools ever run - so
  no tool-side error handling can catch it. Left unhandled, that exception
  crashes the entire graph mid-conversation. We catch it at this one
  boundary (an external API call) and turn it into a plain reply asking the
  user to rephrase, instead of killing the whole session. This is a real,
  observed failure mode of tool-calling LLMs - not defensive code added "just
  in case."
- **Streaming (`app.stream(..., stream_mode="values")`) instead of `app.invoke()`.**
  `invoke()` would only give you the final answer, hiding every tool call in
  between. Streaming the state after each node lets you print the agent's
  intermediate reasoning — essential for building intuition, and for
  debugging when it picks the wrong tool.
- **Graph built by hand instead of `langgraph.prebuilt.create_react_agent`.**
  LangGraph ships a one-line helper that creates this exact graph shape.
  We built it manually here so every part (state, nodes, conditional edges)
  is visible and editable. Once this makes sense, `create_react_agent(llm, tools)`
  is a legitimate shortcut for the same thing in future projects.

---

## 7. What's deliberately left out (and why)

This is Project 1 — kept minimal on purpose so the core loop is easy to see.
Missing pieces are exactly what later projects will add:

| Missing | Why it's not here yet | Where it's coming |
|---|---|---|
| Persistent memory across sessions | Adds a checkpointer/state store — a separate concept from the reasoning loop | Project 3 (conversational agent with memory) |
| Multiple agents / role specialization | Needs the loop above to be solid first | Project 5 (multi-agent research-and-write pipeline) |
| Document retrieval (RAG) | Different concern — indexing/embeddings — from tool-calling | Project 2 (PDF/notes Q&A) |
| Retry/error-recovery logic | Real production concern, but adds noise for a first pass | Later, once basics are second nature |
| Guardrails on tool inputs (e.g. blocking huge Tavily queries, rate limiting) | Same reason | Later |

---

## 8. Troubleshooting

- **`EnvironmentError: GROQ_API_KEY is missing`** — check `.env` is at the
  repo root (one level above this folder) and the key name matches exactly.
- **`401`/auth error from Tavily** — your `TAVILY_API_KEY` is missing/invalid;
  regenerate it at https://app.tavily.com.
- **`ModuleNotFoundError`** — you likely didn't activate the virtual
  environment before running, or forgot `pip install -r requirements.txt`.
- **Agent never calls a tool when you expect it to** — check the tool's
  docstring in `tools.py`. The model chooses tools based purely on that text;
  vague descriptions lead to vague tool choices.
- **Wikipedia disambiguation errors** — `wikipedia_lookup` catches these and
  returns a list of suggested alternatives back to the model, which will
  usually retry with a more specific query on its own.
- **`404 ... model_not_found` from Groq** — the model name in `config.py` is
  no longer hosted. Run `python list_groq_models.py` and swap in a current
  model id.
- **`UnicodeEncodeError` crash mid-answer in the terminal** — Windows
  terminals sometimes default to a legacy code page that can't display every
  character an LLM outputs (curly quotes, em dashes, etc.). `main.py` already
  forces UTF-8 output to prevent this; if you still see it, your terminal
  itself isn't rendering UTF-8 — try Windows Terminal instead of the classic
  `cmd.exe` console.
- **Agent replies "I ran into an error trying to use a tool..."** — this is
  the graceful fallback in `call_agent` (see section 6). The model generated
  a malformed tool call and Groq's API rejected it before your tools ever
  ran. It's a known reliability limit of tool-calling LLMs, not a bug in this
  project - just rephrase and try again.

---

## 9. Ideas to extend this project yourself

- Add a fourth tool (e.g. a weather API, or a simple file reader).
- Add a `max_iterations` guard in `should_continue` to prevent infinite tool-call loops.
- Swap `stream_mode="values"` for `stream_mode="messages"` in `main.py` to see token-by-token streaming of the final answer.
- Run `python list_groq_models.py` to see every model your key can access, try `openai/gpt-oss-20b` in `config.py` instead of `openai/gpt-oss-120b`, and compare how often the smaller/faster model picks the wrong tool.

---

## 10. The Streamlit UI: how it's wired

`streamlit_app.py` is a thin UI layer on top of the *exact same* `agent.py` —
it doesn't duplicate any agent logic, it just renders the same conversation
loop as a web chat instead of a terminal. If you already understand
`main.py`, the only genuinely new ideas here are Streamlit-specific:

- **Streamlit reruns the whole script on every interaction.** Unlike
  `main.py`'s `while True:` loop, there is no persistent Python process
  holding your conversation in a local variable — every time you send a
  message, Streamlit re-executes `streamlit_app.py` from top to bottom.
  Anything that must survive that rerun (the message history, in our case)
  has to be stored in `st.session_state`, a dict-like object Streamlit
  preserves for you across reruns *within the same browser session*.
- **Two message representations, on purpose.** `st.session_state.lc_messages`
  holds the real LangChain message objects (System/Human/AI/Tool) — this is
  what actually gets passed back into the graph so the agent has full
  context each turn, identical in spirit to `messages` in `main.py`.
  `st.session_state.turns` is a separate, simplified list built only for
  *rendering* — plain dicts of `{user, steps, answer}` that `render_steps()`
  can redraw cheaply on every rerun without needing to inspect LangChain
  message types again.
- **`@st.cache_resource` on `get_agent_app()`.** Without this, every rerun
  (i.e. every message you send) would rebuild the `ChatGroq` client and tool
  objects from scratch. `cache_resource` tells Streamlit to build it once per
  server process and hand back the same instance every time — the correct
  cache decorator for objects like API clients (as opposed to `st.cache_data`,
  which is for cacheable *data*, not live connections).
- **`st.status(...)` for live reasoning steps.** This widget gives you a
  collapsible "in progress" box you can write into while a long-running
  operation executes, then collapse into a "Done" summary afterward — a
  direct visual equivalent of the `-> calling tool: ...` lines `main.py`
  prints live, but as a UI control the user can expand/collapse.
- **Same error handling, no extra work needed.** Because the malformed-tool-call
  fallback (see section 6) lives inside `agent.py`'s `call_agent` function,
  the Streamlit UI automatically inherits it — a bad tool call becomes a
  normal-looking assistant reply instead of crashing the Streamlit server.
- **Memory is per-browser-session, not global.** `st.session_state` is scoped
  to one browser tab's session with the Streamlit server. Two people (or two
  tabs) hitting the same running app get independent conversations — but
  refreshing the page, or restarting `streamlit run`, wipes it, same
  limitation as `main.py` (see section 7).
