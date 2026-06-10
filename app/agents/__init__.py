from app.agents.graph import SalesAgentGraph
from app.agents.state import AgentState

# Instantiate the graph once at startup for high-performance dependency injection
sales_agent_graph = SalesAgentGraph()

__all__ = [
    "sales_agent_graph",
    "AgentState",
]