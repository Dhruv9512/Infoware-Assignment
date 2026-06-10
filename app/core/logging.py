import logging
import sys
import structlog

def setup_logging():
    """Configures the structured logging pipeline."""
    structlog.configure(
        processors=[
            # Tracks request IDs/Context across async scopes
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # Renders output nicely in the terminal
            structlog.dev.ConsoleRenderer() 
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Catch standard library logs and route them through structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

    noisy_ai_loggers = [
        "langchain",
        "langchain_core",
        "langchain_google_genai",
        "langchain_groq",
        "openai",  
        "httpcore",
        "httpx"    
    ]
    
    for logger_name in noisy_ai_loggers:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# Automatically initialize the configuration when this module is loaded
setup_logging()

def get_logger(name: str):
    """Utility function to retrieve a bound logger instance."""
    return structlog.get_logger(name)