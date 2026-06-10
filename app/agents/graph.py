from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.core.get_llm import get_llm
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
        # Initialize the LLM and bind our tools the moment the class is instantiated
        self.llm = get_llm().bind_tools(all_platform_tools)
        # Build and compile the execution graph
        self.executor = self._build_graph()

    def _call_model(self, state: AgentState):
        """The primary reasoning node where the LLM decides what to do next."""
        messages = state.get("messages", [])
        
        # If this is the very first message in the loop, inject the system prompt
        if len(messages) > 0 and not isinstance(messages[0], SystemMessage):
            system_msg = SystemMessage(
                content=SALES_AGENT_SYSTEM_PROMPT.format(
                    user_id=state["user_id"] # Removed session_id here!
                )
            )
            messages = [system_msg] + messages

        # Invoke the LLM
        response = self.llm.invoke(messages)
        
        # Track tools if the LLM decided to call any during this specific turn
        tools_called_this_turn = state.get("tools_called", [])
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                tools_called_this_turn.append(tool_call["name"])

        return {"messages": [response], "tools_called": tools_called_this_turn}

    def _should_continue(self, state: AgentState):
        """Edge condition: Determines if the LLM called a tool or is finished answering."""
        last_message = state["messages"][-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return END

    def _build_graph(self):
        """Constructs, connects, and compiles the StateGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add our encapsulated class methods as the nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(all_platform_tools))

        # Graph eges
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "tools": "tools",
                END: END
            }
        )
        workflow.add_edge("tools", "agent")

        # Compile and return the executable application
        return workflow.compile()

    async def ainvoke(self, state: dict, config: dict = None):
        """
        Public execution method. Exposes the async invoke of the compiled graph 
        so external services can interact with it cleanly.
        """
        return await self.executor.ainvoke(state, config)