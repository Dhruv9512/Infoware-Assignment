from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.core.get_llm import LLMFactory
from app.core.logging import get_logger
from app.agents.state import AgentState
from app.prompt.prompts import SALES_AGENT_SYSTEM_PROMPT
from app.tools import all_platform_tools

logger = get_logger(__name__)


class SalesAgentGraph:
    """
    Encapsulated graph execution class.
    Manages the state machine, LLM dynamic initialization, and routing logic
    safely across concurrent asynchronous web requests.
    """

    def __init__(self):
        self.llm = LLMFactory.get_tool_calling_llm(all_platform_tools)
        self.executor = self._build_graph()

    # -------------------------------------------------------------------------
    # Nodes
    # -------------------------------------------------------------------------

    def _call_model(self, state: AgentState):
        """Primary reasoning node — LLM decides the next action."""
        messages = state.get("messages", [])

        if messages and not isinstance(messages[0], SystemMessage):
            system_msg = SystemMessage(
                content=SALES_AGENT_SYSTEM_PROMPT.format(user_id=state["user_id"])
            )
            messages = [system_msg] + messages

        response = self.llm.invoke(messages)

        tools_called = state.get("tools_called", [])
        if hasattr(response, "tool_calls") and response.tool_calls:
            tools_called = tools_called + [tc["name"] for tc in response.tool_calls]

        return {"messages": [response], "tools_called": tools_called}

    # -------------------------------------------------------------------------
    # Edges
    # -------------------------------------------------------------------------

    def _should_continue(self, state: AgentState):
        """Routes to tools if the LLM issued a tool call, otherwise ends."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    # -------------------------------------------------------------------------
    # Graph
    # -------------------------------------------------------------------------

    def _build_graph(self):
        """Constructs, connects, and compiles the StateGraph workflow."""
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(all_platform_tools))

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", self._should_continue, {"tools": "tools", END: END})
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def ainvoke(self, state: dict, config: dict = None):
        """Exposes async graph invocation to external services."""
        return await self.executor.ainvoke(state, config)