import time
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from llm import llm

def planner_agent(state: AgentState) -> AgentState:
    start = time.time()
    messages = [
        SystemMessage(content="You are a planner. Given a task, produce a concise step-by-step plan. Be brief."),
        HumanMessage(content=f"Task: {state['task']}")
    ]
    response = llm.invoke(messages)
    state["plan"] = response.content
    state["latency"]["planner"] = time.time() - start
    return state

def researcher_agent(state: AgentState) -> AgentState:
    start = time.time()
    messages = [
        SystemMessage(content="You are a researcher. Given a plan, gather and summarize relevant facts needed to complete it."),
        HumanMessage(content=f"Plan: {state['plan']}")
    ]
    response = llm.invoke(messages)
    state["research"] = response.content
    state["latency"]["researcher"] = time.time() - start
    return state

def writer_agent(state: AgentState) -> AgentState:
    start = time.time()
    messages = [
        SystemMessage(content="You are a writer. Using the plan and research, produce a complete, well-structured response."),
        HumanMessage(content=f"Plan: {state['plan']}\n\nResearch: {state['research']}")
    ]
    response = llm.invoke(messages)
    state["draft"] = response.content
    state["latency"]["writer"] = time.time() - start
    return state

def critic_agent(state: AgentState) -> AgentState:
    start = time.time()
    messages = [
        SystemMessage(content="You are a critic. Review the draft and provide a quality score from 1-10 with specific feedback."),
        HumanMessage(content=f"Draft: {state['draft']}")
    ]
    response = llm.invoke(messages)
    state["critique"] = response.content
    state["final_output"] = state["draft"]
    state["latency"]["critic"] = time.time() - start
    return state

def planner_researcher_agent(state: AgentState) -> AgentState:
    start = time.time()
    messages = [
        SystemMessage(content="You are a planner and researcher. Given a task, produce a step-by-step plan and gather all relevant facts needed to complete it."),
        HumanMessage(content=f"Task: {state['task']}")
    ]
    response = llm.invoke(messages)
    state["plan"] = response.content
    state["research"] = response.content
    state["latency"]["planner_researcher"] = time.time() - start
    return state

