"""Create the langgraph object that will run the assistant"""

from langgraph.graph import StateGraph
from langgraph.types import Command
from src.models import State
from src.nodes import node_main_assistant, node_rag_assist

def create_rag_graph() -> StateGraph:
    """Builds and returns a LangGraph state graph for the Help Centre assistant"""

    workflow = StateGraph(State)

    workflow.add_node("node_main_assistant", node_main_assistant)
    workflow.set_entry_point("node_main_assistant")

    workflow.add_node("route_next_step", route_next_step)
    workflow.add_node("node_rag_assist", node_rag_assist)

    graph = workflow.compile()

    return graph

def route_next_step(state: State) -> Command:
    """Route the next step based on the state"""

    last_message = state["messages"][-1]
    
    if "tool_calls" not in last_message.additional_kwargs:
        return Command(goto="__end__")

    return Command(goto="node_rag_assist")

    