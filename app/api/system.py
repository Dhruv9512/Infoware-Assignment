import json
import os
from fastapi import APIRouter
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/health")
async def health_check():
    """Service health check endpoint."""
    return {
        "status": "healthy",
        "environment": settings.ENV,
        "llm_provider": settings.LLM_PROVIDER
    }


@router.get("/catalog")
async def get_catalog():
    """Returns the raw product/pricing catalog the agent uses to ground its answers."""
    catalog_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../catalog.json"))
    
    if os.path.exists(catalog_path):
        with open(catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    return {"error": "System fault: catalog.json is missing from the root directory."}