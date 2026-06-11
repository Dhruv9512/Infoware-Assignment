import uuid
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.get_llm import LLMFactory
from app.core.logging import get_logger
from app.models.schemas import ChatRequest, ChatResponse
from app.memory.sqlite_backend import SQLAlchemyMemoryStore
from app.prompt.prompts import SUMMARTY_COMPRESSION_PROMPT
from app.agents import sales_agent_graph
from app.services.eval_service import build_eval_service
from app.db.database import get_db
from langchain_core.messages import HumanMessage, AIMessage

logger = get_logger(__name__)


async def _background_persist_and_summarize(
    user_id: str,
    session_id: str,
    user_message: str,
    final_message_text: str,
    tools_invoked: list,
    eval_block,
):
    """
    Runs DB persistence and memory compression in the background
    using its own independent session — safe after request closes.
    """
    async for db in get_db():
        try:
            memory_store = SQLAlchemyMemoryStore(db)

            await memory_store.save_message_turn(
                user_id=user_id,
                session_id=session_id,
                user_message=user_message,
                agent_message=final_message_text,
                tools_called=tools_invoked,
                catalog_context="",
                evaluation=eval_block.model_dump(),
            )

            current_summary = await memory_store.get_user_summary(user_id) or "No prior facts established."
            compression_prompt = SUMMARTY_COMPRESSION_PROMPT.format(
                current_summary=current_summary,
                new_user_msg=user_message,
                new_agent_reply=final_message_text,
            )
            updated_summary = await LLMFactory.get_llm().ainvoke(compression_prompt)
            await memory_store.update_user_summary(user_id, updated_summary.content.strip())

            logger.info("background_persist_and_summarize_complete", user_id=user_id)

        except Exception as e:
            logger.error("background_persist_and_summarize_failed", error=str(e), user_id=user_id)


class ChatService:
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.memory_store = SQLAlchemyMemoryStore(db_session)
        self.llm_runner = LLMFactory.get_llm()

    async def process_chat_request(self, user_id: str, request: ChatRequest) -> ChatResponse:
        session_id = request.session_id or str(uuid.uuid4())
        logger.info("processing_chat_turn", user_id=user_id, session_id=session_id)

        # ── 1. Hydrate message history ────────────────────────────────────────
        history = await self.memory_store.get_conversation_history(
            user_id=user_id, 
            session_id=session_id
        )
        
        messages = []
        for r in history:
            messages.append(HumanMessage(content=r["user_message"]))
            messages.append(AIMessage(content=r["agent_message"]))

        messages.append(HumanMessage(content=request.message))

        # ── 2. Run the agent ──────────────────────────────────────────────────
        result_state = await sales_agent_graph.ainvoke({
            "messages": messages,
            "user_id": user_id,
            "session_id": session_id,
            "tools_called": [],
        })

        raw_content = result_state["messages"][-1].content
        if isinstance(raw_content, list):
            final_message_text = " ".join(
                block["text"] if isinstance(block, dict) and "text" in block else str(block)
                for block in raw_content
            ) or str(raw_content)
        else:
            final_message_text = str(raw_content)

        tools_invoked = list(dict.fromkeys(result_state.get("tools_called", [])))

        # ── 3. Evaluate ───────────────────────────────────────────────────────
        eval_block = await build_eval_service(
            user_message=request.message,
            agent_response=final_message_text,
            tools_called=tools_invoked,
        )

        # ── 4. Persist + summarize in background ──────────────────────────────
        asyncio.create_task(
            _background_persist_and_summarize(
                user_id=user_id,
                session_id=session_id,
                user_message=request.message,
                final_message_text=final_message_text,
                tools_invoked=tools_invoked,
                eval_block=eval_block,
            )
        )

        return ChatResponse(
            response=final_message_text,
            session_id=session_id,
            tools_called=tools_invoked,
            eval=eval_block,
        )


async def build_chat_service(db_session, user_id, request):
    obj = ChatService(db_session)
    return await obj.process_chat_request(user_id=user_id, request=request)