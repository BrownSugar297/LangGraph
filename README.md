# LangGraph
**LangGraph** is a framework built on top of LangChain for constructing **stateful, multi-actor applications** with Large Language Models. It models complex agent workflows as **graphs** — where nodes represent computation steps and edges represent transitions — enabling robust control flow, cycles, memory, and human-in-the-loop patterns that are difficult or impossible to express with simple linear chains.

This repository documents a hands-on learning journey through LangGraph's core concepts, architectural patterns, and real-world agent systems.

---

## 📚 Table of Contents

- [Goals of This Repository](#-goals-of-this-repository)
- [Current Progress](#-current-progress)
  - [1. Introduction](#1️⃣-introduction)
  - [2. Basic Reflection System](#2️⃣-basic-reflection-system)
  - [3. Reflexion Agent System](#3️⃣-reflexion-agent-system)
  - [4. State System](#4️⃣-state-system)
  - [5. ReAct Agent](#5️⃣-react-agent)
  - [6. Chatbots](#6️⃣-chatbots)
  - [7. Human in the Loop](#7️⃣-human-in-the-loop)
  - [8. RAG Agent](#8️⃣-rag-agent)
  - [9. Multi-Agent Architecture](#9️⃣-multi-agent-architecture)
  - [10. Streaming](#🔟-streaming)
- [Installation & Setup](#-installation--setup)
- [Technologies Used](#-technologies-used)
- [Contribution](#-contribution)

---

## 🎯 Goals of This Repository

- Understand **LangGraph's graph-based execution model** from the ground up.
- Build increasingly sophisticated **agent systems** — from simple reflection loops to full multi-agent architectures.
- Explore real-world patterns like **Human-in-the-Loop**, **RAG agents**, and **streaming** through practical, well-documented examples.
- Serve as an **educational reference** for anyone learning agentic AI systems.

---

## 🗂️ Current Progress

---

### 1️⃣ Introduction

**Folder:** `1_Introduction/`

An introduction to LangGraph's foundational concepts — how it differs from LangChain's linear chains and why graph-based execution unlocks more powerful agentic behavior.

**Topics covered:**
- What is LangGraph and why it exists
- Core primitives: `StateGraph`, nodes, edges, and entry/finish points
- How state flows between nodes
- Compiling and invoking a basic graph

**Example — Minimal LangGraph:**
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    message: str

def greet(state: AgentState) -> AgentState:
    return {"message": f"Hello, {state['message']}!"}

graph = StateGraph(AgentState)
graph.add_node("greet", greet)
graph.set_entry_point("greet")
graph.add_edge("greet", END)

app = graph.compile()
print(app.invoke({"message": "world"}))
```

---

### 2️⃣ Basic Reflection System

**Folder:** `2_basic_reflection_system/`

A reflection system is an agent loop where the model generates an initial response and then critiques or refines it. This module implements the simplest form of that pattern using a two-node graph.

**Topics covered:**
- Generating and critiquing outputs in a loop
- Using conditional edges to stop iteration
- Structuring feedback cycles as graphs
- Controlling loop termination logic

**Workflow:**
```
[Generate] → [Reflect] → [Generate] → ... → [END]
```

The loop continues until the reflection node determines the output is satisfactory, at which point control passes to `END`.

---

### 3️⃣ Reflexion Agent System

**Folder:** `3_reflexion_agent_system/`

An advanced evolution of the basic reflection system, inspired by the **Reflexion** research paper. The agent reasons over its previous attempts, learns from failures, and iteratively improves its answers using self-critique and external tool feedback.

**Topics covered:**
- Implementing the full Reflexion architecture
- Combining tool use with reflection loops
- Managing episodic memory across iterations
- Using structured outputs for reflection reasoning

**Reflexion Loop:**
```
[Actor] → [Evaluate] → (pass) → [END]
                      ↘ (fail) → [Reflect] → [Actor]
```

The Actor attempts a task, the Evaluator judges the result, and if unsatisfactory, the Reflector generates structured critique to guide the next attempt.

---

### 4️⃣ State System

**Folder:** `4_state_system/`

A deep dive into LangGraph's **state management** — the backbone of all graph-based workflows. This module covers how state is defined, passed, updated, and persisted across nodes.

**Topics covered:**
- Defining typed state schemas with `TypedDict` and Pydantic
- State reducers and how updates are merged
- `Annotated` fields and `operator.add` for append-only state
- Passing state between nodes without losing history
- Checkpointing state for persistence and resumption

**Example — State with message history:**
```python
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    iteration: int
```

---

### 5️⃣ ReAct Agent

**Folder:** `5_react_agent/`

Implementation of the **ReAct (Reasoning + Acting)** agent pattern inside LangGraph. The agent alternates between reasoning about its next action and executing that action using tools, creating a tight observe → think → act loop.

**Topics covered:**
- Building a ReAct loop from scratch in LangGraph
- Binding tools to LLMs and routing based on tool call output
- Implementing the `should_continue` conditional edge
- Integrating web search (DuckDuckGo) as an external tool
- Tracking intermediate reasoning steps in state

**ReAct Loop:**
```
[Reason] → (tool call) → [Act / Tool Execution] → [Reason] → ... → [END]
         ↘ (no tool call) → [END]
```

---

### 6️⃣ Chatbots

**Folder:** `6_chatbots/`

Building production-ready chatbots in LangGraph with **persistent memory** and **multi-turn conversation management**. This module demonstrates how to maintain context across messages using LangGraph's checkpointing system.

**Topics covered:**
- Building a stateful chatbot graph
- Persisting conversation history using `SqliteSaver` (checkpoint.sqlite)
- Managing thread IDs for multi-session support
- Trimming message history to fit context windows
- Adding a summarization node for long conversations

**Example — Configuring a persistent session:**
```python
config = {"configurable": {"thread_id": "user_session_1"}}
response = app.invoke({"messages": [HumanMessage(content="Hello!")]}, config=config)
```

---

### 7️⃣ Human in the Loop

**Folder:** `7_human_in_the_loop/`

One of LangGraph's most powerful features — the ability to **pause graph execution**, hand control to a human for review or input, and then **resume** from exactly where it left off.

**Topics covered:**
- Using `interrupt_before` to pause before sensitive nodes
- Resuming graphs with human-provided state updates
- Approval workflows (approve / reject / modify)
- Combining checkpointing with human interrupts for durability
- Practical use cases: tool call approval, content moderation, form filling

**Workflow:**
```
[Agent] → (tool call detected) → ⏸ PAUSE → [Human Review]
                                                  ↓ approve
                                            [Tool Execution] → [Agent]
```

---

### 8️⃣ RAG Agent

**Folder:** `8_rag_agent/`

A fully graph-based **Retrieval-Augmented Generation (RAG) agent** built in LangGraph. Unlike simple RAG chains, this implementation uses graph edges to conditionally route queries — checking whether retrieval is necessary, validating retrieved context, and falling back gracefully.

**Topics covered:**
- Integrating vector stores (FAISS / Chroma) into a LangGraph node
- Conditional routing: retrieve vs. answer directly
- Document grading nodes to filter irrelevant context
- Query rewriting when retrieval quality is low
- Full RAG pipeline as a stateful graph

**RAG Graph:**
```
[Classify Query] → (needs retrieval) → [Retrieve] → [Grade Docs]
                 ↘ (direct answer)  → [Generate]      ↓ (relevant)
                                                   [Generate]
                                                       ↓ (irrelevant)
                                                   [Rewrite Query] → [Retrieve]
```

---

### 9️⃣ Multi-Agent Architecture

**Folder:** `9_multi_agent_architecture/`

Implementation of **multi-agent systems** where specialized agents collaborate, delegate tasks, and communicate through a supervisor or network topology. This is the most advanced module in the repository.

**Topics covered:**
- Supervisor agent pattern: a routing LLM that delegates to specialist agents
- Building specialized sub-agents (researcher, coder, analyst, etc.)
- Agent-to-agent communication via shared state
- Hierarchical multi-agent graphs (supervisor of supervisors)
- Avoiding cycles and managing termination in multi-agent loops

**Supervisor Architecture:**
```
                    [Supervisor]
                   /      |      \
          [Researcher] [Coder] [Analyst]
                   \      |      /
                    [Supervisor]
                         ↓
                        END
```

The Supervisor LLM decides which specialist to invoke next based on the current state of the task.

---

### 🔟 Streaming

**Folder:** `10_streaming/`

Streaming LangGraph outputs in real time — covering both token-level streaming from LLMs and event-level streaming from graph nodes.

**Topics covered:**
- Streaming tokens using `.astream()` and `stream_mode="messages"`
- Streaming intermediate node outputs with `stream_mode="updates"`
- Streaming full state snapshots with `stream_mode="values"`
- Combining streaming with human-in-the-loop patterns
- Building responsive UIs with streamed agent output

**Example — Streaming node updates:**
```python
async for event in app.astream(inputs, stream_mode="updates"):
    for node, state in event.items():
        print(f"[{node}] → {state}")
```

---

## ⚙️ Installation & Setup

**1. Clone this repository:**
```bash
git clone https://github.com/BrownSugar297/LangGraph.git
cd LangGraph
```

**2. Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate          # Windows
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Add your API keys to a `.env` file:**
```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
```

**5. Run any notebook:**
```bash
jupyter notebook
```
Navigate to the relevant module folder and open the `.ipynb` file.

---

## 🧰 Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.11** | Core programming language |
| **LangGraph** `0.3.21` | Graph-based agent orchestration framework |
| **LangChain** `0.3.18` | LLM tooling, prompts, and chain primitives |
| **LangChain-OpenAI** | OpenAI GPT model integration |
| **LangChain-Google-GenAI** | Google Gemini model integration |
| **LangChain-Groq** | Groq LPU inference integration |
| **LangGraph Checkpoint SQLite** | Persistent state storage for checkpointing |
| **LangSmith** | Tracing, debugging, and observability |
| **OpenAI API** | GPT-4 / GPT-4o language models |
| **Google Generative AI** | Gemini models |
| **Groq** | Fast inference for open-source models |
| **Pydantic v2** | Data validation and structured schemas |
| **aiosqlite** | Async SQLite for checkpoint persistence |
| **Jupyter Notebook** | Interactive development environment |
| **python-dotenv** | API key and environment management |

---

## 💡 Contribution

Contributions are always welcome! 🙌 If you'd like to improve this repository, add new LangGraph examples, or fix issues:

**1. Fork the repository**

**2. Create a new branch for your feature or fix:**
```bash
git checkout -b feature/your-feature-name
```

**3. Commit your changes:**
```bash
git commit -m "Add: detailed example for multi-agent supervisor pattern"
```

**4. Push to your fork:**
```bash
git push origin feature/your-feature-name
```

**5. Open a Pull Request** — describe what you've added and why it improves the repo.

---

> **Note:** This repository is a structured learning reference. Each folder is self-contained and can be explored independently. It is recommended to follow the numbered order for the best learning progression.
>
> ## 👨‍💻 Author

**Ashikur Rahman**
