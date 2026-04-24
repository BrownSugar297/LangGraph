from typing import List
import time
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph

from chains import generation_chain, reflection_chain

load_dotenv()

REFLECT = "reflect"
GENERATE = "generate"

graph = MessageGraph()


# ✅ Generate Node
def generate_node(state: List[BaseMessage]):
    response = generation_chain.invoke({
        "messages": state
    })
    return state + [response]


# ✅ Reflect Node
def reflect_node(state: List[BaseMessage]):
    time.sleep(1)  # optional (Groq is fast, small delay is enough)

    response = reflection_chain.invoke({
        "messages": state
    })

    return state + [HumanMessage(content=response.content)]


# Add nodes
graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

graph.set_entry_point(GENERATE)


# ✅ Loop control (IMPORTANT)
MAX_ITER = 2

def should_continue(state: List[BaseMessage]):
    iteration = len(state) // 2

    if iteration >= MAX_ITER:
        return END
    return REFLECT


# Edges
graph.add_conditional_edges(GENERATE, should_continue)
graph.add_edge(REFLECT, GENERATE)


# Compile
app = graph.compile()


# Debug view
print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()


# Run
response = app.invoke([
    HumanMessage(content="AI Agents taking over content creation")
])

print("\n\nFinal Output:\n")
print(response[-1].content)