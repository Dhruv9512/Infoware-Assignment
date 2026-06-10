from langchain_core.prompts import ChatPromptTemplate
from app.core.get_llm import LLMFactory
from app.core.logging import get_logger
from app.models.schemas import FeedbackLoopBlock
# Import the newly separated prompt templates
from app.prompt.prompts import EVAL_SYSTEM_PROMPT, EVAL_HUMAN_PROMPT

logger = get_logger(__name__)


class EvaluationService:
    """
    Encapsulated service for grading agent responses.
    Compiles the LLM chain once upon initialization for high performance.
    """
    
    def __init__(self):
        self.llm = LLMFactory.build_structured_eval_llm()
        self.prompt = self._build_prompt()
        self.chain = self.prompt | self.llm

    def _build_prompt(self):
        """Constructs the strict rubric prompt using external configurations."""
        return ChatPromptTemplate.from_messages([
            ("system", EVAL_SYSTEM_PROMPT),
            ("human", EVAL_HUMAN_PROMPT)
        ])

    async def evaluate_turn(self, user_message: str, agent_response: str, tools_called: list) -> FeedbackLoopBlock:
        """Executes the self-evaluation LLM chain to grade the agent's output."""
        logger.info("eval_service_invoked", tools_count=len(tools_called))
        
        try:
            result: FeedbackLoopBlock = await self.chain.ainvoke({
                "user_message": user_message,
                "agent_response": agent_response,
                "tools_called": tools_called
            })
            
            if result.flagged:
                logger.warning("agent_response_flagged_by_eval", reasoning=result.reasoning)
                
            return result
            
        except Exception as e:
            logger.error("eval_service_failed", error=str(e))
            # Fallback safeguard to ensure the API never crashes if the formatting model fails
            return FeedbackLoopBlock(
                groundedness=0.0,
                relevance=0.0,
                confidence=0.0,
                flagged=True,
                reasoning=f"Evaluation execution failed: {str(e)}"
            )


async def build_eval_service(user_message, agent_response, tools_called):
    """
    Asynchronous factory wrapper to trigger structured valuations safely.
    Ensures coroutines are awaited properly before returning to the chat service orchestrator.
    """
    obj = EvaluationService()
    return await obj.evaluate_turn(
        user_message=user_message,
        agent_response=agent_response,
        tools_called=tools_called
    )