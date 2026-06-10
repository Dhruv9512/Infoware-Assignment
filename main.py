from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.database import engine, Base
from app.api.routes import api_router

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown lifecycle events for the platform.
    Automatically provisions database tables on boot if they do not exist.
    """
    logger.info("booting_sales_ai_platform", env=settings.ENV)
    
    # Auto-create SQLAlchemy tables on application startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_tables_verified_successfully")
        
    yield  # The application serves traffic here
    
    # Clean teardown of the database connection pool on exit
    logger.info("shutting_down_sales_ai_platform")
    await engine.dispose()


# Initialize the FastAPI Application instance
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="A Persistent Sales Assistant Agent featuring cross-session memory and structural LLM self-evaluations.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS middleware for smooth frontend/client integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to specific domains when deploying to production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the unified platform API router containing health and chat nodes
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn
    # Allows direct native script execution via `python main.py`
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)