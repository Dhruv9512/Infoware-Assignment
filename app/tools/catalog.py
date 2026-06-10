import json
import os
from typing import Any, Dict, List
from langchain_core.tools import tool
from app.core.logging import get_logger

logger = get_logger(__name__)

# Load the catalog once into memory on startup
CATALOG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../catalog.json"))
_CATALOG_CACHE: Dict[str, Any] = {"plans": []}

try:
    if os.path.exists(CATALOG_PATH):
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            _CATALOG_CACHE = json.load(f)
        logger.info("catalog_loaded_successfully", path=CATALOG_PATH, plan_count=len(_CATALOG_CACHE.get("plans", [])))
    else:
        logger.error("catalog_file_missing", path=CATALOG_PATH)
except Exception as e:
    logger.error("catalog_load_failed", error=str(e))


@tool
def search_catalog(query: str) -> str:
    """
    Semantic or keyword search over the product catalog JSON.
    CRITICAL INSTRUCTION: Pass a single, simple keyword (like 'Enterprise' or 'pricing'). 
    Do NOT call this tool multiple times for the same concept. One query is enough to find the data.

    Args:
        query: The keyword or feature string to search for (e.g., 'SSO', 'storage', 'pricing').
    """
    logger.info("tool_search_catalog_invoked", query=query)
    
    plans: List[Dict[str, Any]] = _CATALOG_CACHE.get("plans", [])
    
    if not plans:
        return "System Error: Product catalog is currently empty or unavailable."

    query_lower = query.lower()
    matched_plans = []

    # Iterates through all available plans without hardcoding assumptions
    for plan in plans:
        # 1. Search against the plan name
        if query_lower in str(plan.get("name", "")).lower():
            matched_plans.append(plan)
            continue
            
        # 2. Search against the price string
        if query_lower in str(plan.get("price", "")).lower():
            matched_plans.append(plan)
            continue
            
        # 3. Dynamic search across the entire features array
        features = plan.get("features", [])
        if any(query_lower in str(f).lower() for f in features):
            matched_plans.append(plan)
            continue

    # Catch-all for broad queries: If the user asks generally about "pricing" or "plans", 
    if query_lower in ["pricing", "plan", "cost", "features", "all"]:
        matched_plans = plans

    # If we found matches, return them cleanly formatted for the LLM
    if matched_plans:
        return json.dumps({
            "matched_plans": matched_plans, 
            "context_note": "Sourced from official catalog specs."
        }, indent=2)

    # If no match is found, dynamically list whatever plans actually exist in the JSON
    available_plan_names = [str(p.get("name", "Unknown")) for p in plans]
    fallback_message = (
        f"No direct plan features found matching '{query}'. "
        f"Available plans are: {', '.join(available_plan_names)}."
    )
    return fallback_message