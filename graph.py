from langgraph.graph import StateGraph, END
from state import AgentState
from agents import planner_agent, researcher_agent, writer_agent, critic_agent, planner_researcher_agent

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_agent)
    graph.add_node("researcher", researcher_agent)
    graph.add_node("writer", writer_agent)
    graph.add_node("critic", critic_agent)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "critic")
    graph.add_edge("critic", END)

    return graph.compile()


def build_merged_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner_researcher", planner_researcher_agent)
    graph.add_node("writer", writer_agent)
    graph.add_node("critic", critic_agent)

    graph.set_entry_point("planner_researcher")
    graph.add_edge("planner_researcher", "writer")
    graph.add_edge("writer", "critic")
    graph.add_edge("critic", END)

    return graph.compile()