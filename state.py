from typing import TypedDict, Optional

class AgentState(TypedDict):
    task: str
    plan: Optional[str]
    research: Optional[str]
    draft: Optional[str]
    critique: Optional[str]
    final_output: Optional[str]
    token_usage: dict
    latency: dict
    