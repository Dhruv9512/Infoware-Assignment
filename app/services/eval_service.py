from langchain_core.prompts import ChatPromptTemplate

from app.core.get_llm import LLMFactory
from app.core.logging import get_logger
from app.models.schemas import FeedbackLoopBlock
from app.prompt.prompts import EVAL_SYSTEM_PROMPT, EVAL_HUMAN_PROMPT

logger = get_logger(__name__)


class EvaluationService:
    """
    Encapsulated service for grading agent responses.
    Compiles the LLM chain once upon initialization for high performance.
    """

    def __init__(self):
        self.llm = LLMFactory.build_structured_eval_llm()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", EVAL_SYSTEM_PROMPT),
            ("human", EVAL_HUMAN_PROMPT),
        ])
        self.chain = self.prompt | self.llm

    async def evaluate_turn(
        self,
        user_message: str,
        agent_response: str,
        tools_called: list,
    ) -> FeedbackLoopBlock:
        """Grades the agent's last response against a strict rubric."""
        logger.info("eval_service_invoked", tools_count=len(tools_called))

        try:
            result: FeedbackLoopBlock = await self.chain.ainvoke({
                "user_message": user_message,
                "agent_response": agent_response,
                "tools_called": tools_called,
            })

            if result.flagged:
                logger.warning("agent_response_flagged_by_eval", reasoning=result.reasoning)

            return result

        except Exception as e:
            logger.error("eval_service_failed", error=str(e))
            return FeedbackLoopBlock(
                groundedness=0.0,
                relevance=0.0,
                confidence=0.0,
                flagged=True,
                reasoning=f"Evaluation execution failed: {str(e)}",
            )


async def build_eval_service(
    user_message: str,
    agent_response: str,
    tools_called: list,
) -> FeedbackLoopBlock:
    """Factory wrapper — instantiates the service and awaits a single evaluation turn."""
    return await EvaluationService().evaluate_turn(
        user_message=user_message,
        agent_response=agent_response,
        tools_called=tools_called,
    )