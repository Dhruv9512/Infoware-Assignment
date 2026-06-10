from app.core.config import get_settings
settings = get_settings()


def get_llm():
    """Initializes a strict, zero-temperature LLM dedicated entirely to JSON output formatting."""
    if settings.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=settings.GROQ_MODEL, temperature=0.0)
    elif settings.LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=settings.GEMINI_MODEL, temperature=0.0)
    else:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        llm = HuggingFaceEndpoint(repo_id=settings.HUGGINGFACE_MODEL, temperature=0.0)
        return ChatHuggingFace(llm=llm)