from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseMemoryStore(ABC):
    """
    Abstract Base Class defining the Repository interface for the memory layer.
    Ensures that the agent orchestration loops are completely decoupled from 
    the underlying database technology.
    """

    @abstractmethod
    async def get_conversation_history(
        self, 
        user_id: str, 
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the turn-by-turn raw historical messages for a user across all sessions.
        Used to back the history API endpoint and populate audit trails.
        """
        pass

    @abstractmethod
    async def get_user_summary(self, user_id: str) -> Optional[str]:
        """
        Retrieves the long-term compressed memory profile summary for a given user.
        Used to fetch cross-session context before an agent turn begins.
        """
        pass

    @abstractmethod
    async def save_message_turn(
        self, 
        user_id: str, 
        session_id: str, 
        role: str, 
        content: str,
        tools_called: List[str],
        catalog_context: str,
        evaluation: Dict[str, Any]
    ) -> None:
        """
        Persists a single conversational turn, including its tool execution 
        logs and required self-evaluation metrics.
        """
        pass

    @abstractmethod
    async def update_user_summary(self, user_id: str, new_summary: str) -> None:
        """
        Overwrites or updates the compressed long-term profile facts for a user.
        """
        pass

    @abstractmethod
    async def clear_user_memory(self, user_id: str) -> None:
        """
        Completely deletes all rows (history and summary) tied to a user.
        Fulfills the GDPR-style wipe requirement.
        """
        pass