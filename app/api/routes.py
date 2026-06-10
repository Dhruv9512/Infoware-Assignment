from fastapi import APIRouter
from app.api import chat, system

api_router = APIRouter()

# Group the AI chat routes under the /chat prefix
api_router.include_router(chat.router, prefix="/chat", tags=["AI Conversational Agent"])
# Expose the system routes at the root level
api_router.include_router(system.router, tags=["System & Health"])