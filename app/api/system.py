from http.client import HTTPException
import json
import os
from fastapi import APIRouter
from app.models.schemas import CatalogResponse
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


@router.get("/catalog", response_model=CatalogResponse)
async def get_catalog():
    """Returns the product/pricing catalog the agent uses to ground its answers."""
    catalog_path = os.path.abspath(os.path.join(os.getcwd(), "catalog.json"))

    if not os.path.exists(catalog_path):
        raise HTTPException(
            status_code=404,
            detail=f"catalog.json not found at: {catalog_path}"
        )

    with open(catalog_path, "r", encoding="utf-8") as f:
        return CatalogResponse(**json.load(f))