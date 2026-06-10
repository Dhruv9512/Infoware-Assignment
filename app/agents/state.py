from typing import Annotated, List, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    The central state object passed between all nodes in the LangGraph workflow.
    """
    messages: Annotated[List[AnyMessage], add_messages]
    user_id: str
    session_id: str
    tools_called: List[str]