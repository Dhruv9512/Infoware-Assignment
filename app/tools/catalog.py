import json
import os
from typing import Any, Dict, List

from langchain_core.tools import tool

from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Catalog bootstrap ─────────────────────────────────────────────────────────

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

# ── Broad-match keywords that return the full catalog ─────────────────────────

_BROAD_MATCH_KEYWORDS = {"pricing", "plan", "cost", "features", "all"}

# ─────────────────────────────────────────────────────────────────────────────


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

    if query_lower in _BROAD_MATCH_KEYWORDS:
        matched_plans = plans
    else:
        matched_plans = []
        for plan in plans:
            if query_lower in str(plan.get("name", "")).lower():
                matched_plans.append(plan)
            elif query_lower in str(plan.get("price", "")).lower():
                matched_plans.append(plan)
            elif any(query_lower in str(f).lower() for f in plan.get("features", [])):
                matched_plans.append(plan)

    if matched_plans:
        return json.dumps(
            {"matched_plans": matched_plans, "context_note": "Sourced from official catalog specs."},
            indent=2,
        )

    available = ", ".join(str(p.get("name", "Unknown")) for p in plans)
    return f"No direct plan features found matching '{query}'. Available plans are: {available}."