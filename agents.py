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

def controller_agent(state: AgentState) -> str:
    messages = [
        SystemMessage(content="You are a router. Given a task, reply with only one word: 'simple' if the task needs no research, or 'complex' if it requires research and fact gathering."),
        HumanMessage(content=f"Task: {state['task']}")
    ]
    response = llm.invoke(messages)
    decision = response.content.strip().lower()
    state["latency"]["controller"] = 0
    return "simple" if "simple" in decision else "complex"

def summarize_context(text: str) -> str:
    messages = [
        SystemMessage(content="Summarize the following in 3 sentences maximum. Be concise and keep only the most important facts."),
        HumanMessage(content=text)
    ]
    response = llm.invoke(messages)
    return response.content

def researcher_compressed_agent(state: AgentState) -> AgentState:
    start = time.time()
    messages = [
        SystemMessage(content="You are a researcher. Given a plan, gather and summarize relevant facts needed to complete it."),
        HumanMessage(content=f"Plan: {state['plan']}")
    ]
    response = llm.invoke(messages)
    full_research = response.content
    state["research"] = summarize_context(full_research)
    state["latency"]["researcher"] = time.time() - start
    return state
