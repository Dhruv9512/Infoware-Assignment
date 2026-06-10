import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage, AIMessage

from app.core.get_llm import LLMFactory
from app.core.logging import get_logger
from app.models.schemas import ChatRequest, ChatResponse
from app.memory.sqlite_backend import SQLAlchemyMemoryStore
from app.prompt.prompts import SUMMARTY_COMPRESSION_PROMPT
from app.agents import sales_agent_graph
from app.services.eval_service import build_eval_service

logger = get_logger(__name__)


class ChatService:
    """
    Core orchestrator for the conversational loop. 
    Manages state hydration, graph execution, evaluation triggers, and database persistence.
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.memory_store = SQLAlchemyMemoryStore(db_session)
        self.llm_runner = LLMFactory.get_llm()
        
    async def _update_user_long_term_summary(self, user_id: str, new_user_msg: str, new_agent_reply: str) -> None:
        """
        Background execution step that extracts key customer preferences 
        and updates the long-term profile summary.
        """
        # 1. Fetch the existing summary string from the database
        current_summary = await self.memory_store.get_user_summary(user_id) or "No prior facts established."
        
        SUMMARTY_COMPRESSION = SUMMARTY_COMPRESSION_PROMPT.format(
            current_summary=current_summary,
            new_user_msg=new_user_msg,
            new_agent_reply=new_agent_reply
        )
        try:
            # We use the underlying graph's LLM component to execute a fast invocation pass
            extracted_response = await self.llm_runner.ainvoke(SUMMARTY_COMPRESSION)
            updated_summary_text = extracted_response.content.strip()
            
            # 3. Commit the newly compressed summary block directly back to the DB layer
            await self.memory_store.update_user_summary(user_id, updated_summary_text)
            logger.info("user_summary_compressed_and_stored", user_id=user_id)
            
        except Exception as e:
            logger.error("memory_summarization_pass_failed", error=str(e), user_id=user_id)

    async def process_chat_request(self, user_id: str, request: ChatRequest) -> ChatResponse:
        """Executes the full conversational pipeline for an incoming web payload."""
        
        # 1. Establish Session Context
        session_id = request.session_id or str(uuid.uuid4())
        logger.info("processing_chat_turn", user_id=user_id, session_id=session_id)

        # 2. Hydrate Graph State with recent conversation history
        history_records = await self.memory_store.get_conversation_history(user_id, limit=6)
        messages = []
        
        for record in history_records:
            if record["role"] == "user":
                messages.append(HumanMessage(content=record["content"]))
            elif record["role"] == "assistant":
                messages.append(AIMessage(content=record["content"]))

        # Append the brand new incoming user query
        messages.append(HumanMessage(content=request.message))

        # 3. Execute the LangGraph Agent Node Engine
        state_input = {
            "messages": messages,
            "user_id": user_id,
            "session_id": session_id,
            "tools_called": []
        }
        
        result_state = await sales_agent_graph.ainvoke(state_input)
        
        # Extract the final artifacts from the graph state
        raw_content = result_state["messages"][-1].content
        
        # Flatten multi-modal lists into standard strings for the API response
        if isinstance(raw_content, list):
            extracted_text = []
            for block in raw_content:
                if isinstance(block, dict) and "text" in block:
                    extracted_text.append(block["text"])
                elif isinstance(block, str):
                    extracted_text.append(block)
            final_message_text = " ".join(extracted_text) if extracted_text else str(raw_content)
        else:
            final_message_text = str(raw_content)
            
        tools_invoked = result_state.get("tools_called", [])
        
        # Remove duplicate tool calls
        tools_invoked = list(dict.fromkeys(tools_invoked))

        # 4. Run the strict LLM Self-Evaluation Block
        eval_block = await build_eval_service(
            user_message=request.message,
            agent_response=final_message_text,
            tools_called=tools_invoked
        )

        # 5. Persist the granular turn payload back to the database history
        await self.memory_store.save_message_turn(
            user_id=user_id,
            session_id=session_id,
            role="user",
            content=request.message,
            tools_called=[],
            catalog_context="",
            evaluation={}
        )
        
        await self.memory_store.save_message_turn(
            user_id=user_id,
            session_id=session_id,
            role="assistant",
            content=final_message_text,
            tools_called=tools_invoked,
            catalog_context="", 
            evaluation=eval_block.model_dump()
        )

        # 6. RUN THE AUTOMATED COMPRESSION PASS (Fulfills the summary requirement!)
        # This compiles the newly extracted context information into long-term storage
        await self._update_user_long_term_summary(
            user_id=user_id, 
            new_user_msg=request.message, 
            new_agent_reply=final_message_text
        )

        # 7. Return the strict Pydantic payload required by the assignment specs
        return ChatResponse(
            response=final_message_text,
            session_id=session_id,
            tools_called=tools_invoked,
            eval=eval_block
        )


async def build_chat_service(db_session, user_id, request):
    """Refactored async factory function matching structural API endpoint layers cleanly."""
    obj = ChatService(db_session)
    return await obj.process_chat_request(user_id=user_id, request=request)