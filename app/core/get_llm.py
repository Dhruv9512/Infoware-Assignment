from app.core.config import get_settings

settings = get_settings()


class LLMFactory:

    @staticmethod
    def get_llm():
        """
        General purpose LLM with fallback chain.
        
        Order:
            Groq llama3-70b → Groq llama3-8b → Groq mixtral →
            Gemini Flash → Gemini Pro →
            HF Llama3-8B → HF Llama3-8B (retry)
        """
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_groq import ChatGroq
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        # ── 1. Primary: Groq llama3-70b ──────────────────────────────────────────
        primary_llm = ChatGroq(
            model=settings.GROQ_MODEL,   # llama3-70b-8192
            temperature=settings.LLM_TEMPERATURE,
            groq_api_key=settings.GROQ_API_KEY,
            max_retries=1,
        )

        fallbacks = []

        # ── 2. Groq llama3-8b ────────────────────────────────────────────────────
        fallbacks.append(
            ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=settings.LLM_TEMPERATURE,
                groq_api_key=settings.GROQ_API_KEY,
                max_retries=1,
            )
        )

        # ── 3. Groq mixtral ──────────────────────────────────────────────────────
        fallbacks.append(
            ChatGroq(
                model="llama-3.1-70b-versatile",
                temperature=settings.LLM_TEMPERATURE,
                groq_api_key=settings.GROQ_API_KEY,
                max_retries=1,
            )
        )

        # ── 4. Gemini Flash ──────────────────────────────────────────────────────
        if settings.GOOGLE_API_KEY:
            fallbacks.append(
                ChatGoogleGenerativeAI(
                    model=settings.GEMINI_MODEL,   # gemini-2.5-flash
                    temperature=settings.LLM_TEMPERATURE,
                    google_api_key=settings.GOOGLE_API_KEY,
                    max_retries=1,
                )
            )

            # ── 5. Gemini Pro ─────────────────────────────────────────────────────
            fallbacks.append(
                ChatGoogleGenerativeAI(
                    model="gemini-2.5-pro",
                    temperature=settings.LLM_TEMPERATURE,
                    google_api_key=settings.GOOGLE_API_KEY,
                    max_retries=1,
                )
            )

        # ── 6 & 7. HuggingFace (2 attempts, same model) ──────────────────────────
        if settings.HUGGINGFACEHUB_API_TOKEN:
            for _ in range(2):
                hf_endpoint = HuggingFaceEndpoint(
                    repo_id=settings.HUGGINGFACE_MODEL,   # meta-llama/Meta-Llama-3-8B-Instruct
                    temperature=settings.LLM_TEMPERATURE,
                    huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN,
                    max_new_tokens=1024,
                )
                fallbacks.append(ChatHuggingFace(llm=hf_endpoint))

        return primary_llm.with_fallbacks(
            fallbacks,
            exceptions_to_handle=(Exception,),
        )

    @staticmethod
    def get_tool_calling_llm(tools: list):
        """
        Tool-calling LLM with fallback chain.
        HuggingFace excluded — unreliable tool calling support.

        Order:
            Groq llama3-70b → Groq llama3-8b → Groq mixtral →
            Gemini Flash → Gemini Pro
        """
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_groq import ChatGroq

        # ── 1. Primary: Groq llama3-70b ──────────────────────────────────────────
        primary_llm = ChatGroq(
            model=settings.GROQ_MODEL,   # llama3-70b-8192
            temperature=settings.LLM_TEMPERATURE,
            groq_api_key=settings.GROQ_API_KEY,
            max_retries=1,
        ).bind_tools(tools)

        fallbacks = []

        # ── 2. Groq llama3-8b ────────────────────────────────────────────────────
        fallbacks.append(
            ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=settings.LLM_TEMPERATURE,
                groq_api_key=settings.GROQ_API_KEY,
                max_retries=1,
            ).bind_tools(tools)
        )

        # ── 3. Groq mixtral ──────────────────────────────────────────────────────
        fallbacks.append(
            ChatGroq(
                model="llama-3.1-70b-versatile",
                temperature=settings.LLM_TEMPERATURE,
                groq_api_key=settings.GROQ_API_KEY,
                max_retries=1,
            ).bind_tools(tools)
        )

        # ── 4. Gemini Flash ──────────────────────────────────────────────────────
        if settings.GOOGLE_API_KEY:
            fallbacks.append(
                ChatGoogleGenerativeAI(
                    model=settings.GEMINI_MODEL,
                    temperature=settings.LLM_TEMPERATURE,
                    google_api_key=settings.GOOGLE_API_KEY,
                    max_retries=1,
                ).bind_tools(tools)
            )

            # ── 5. Gemini Pro ─────────────────────────────────────────────────────
            fallbacks.append(
                ChatGoogleGenerativeAI(
                    model="gemini-2.5-pro",
                    temperature=settings.LLM_TEMPERATURE,
                    google_api_key=settings.GOOGLE_API_KEY,
                    max_retries=1,
                ).bind_tools(tools)
            )

        # Note: HuggingFace intentionally excluded — tool calling not reliable

        return primary_llm.with_fallbacks(
            fallbacks,
            exceptions_to_handle=(Exception,),
        )

    @staticmethod
    def build_structured_eval_llm():
        """
        Structured output LLM for self-evaluation scoring.
        HuggingFace excluded — does NOT support with_structured_output().

        Order:
            Groq llama3-70b → Groq llama3-8b → Groq mixtral →
            Gemini Flash → Gemini Pro
        """
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_groq import ChatGroq
        from app.models.schemas import FeedbackLoopBlock

        # ── 1. Primary: Groq llama3-70b ──────────────────────────────────────────
        primary_eval = ChatGroq(
            model=settings.GROQ_MODEL,   # llama3-70b-8192
            temperature=0.0,             # zero temp for consistent scoring
            groq_api_key=settings.GROQ_API_KEY,
            max_retries=1,
        ).with_structured_output(FeedbackLoopBlock)

        eval_fallbacks = []

        # ── 2. Groq llama3-8b ────────────────────────────────────────────────────
        eval_fallbacks.append(
            ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.0,
                groq_api_key=settings.GROQ_API_KEY,
                max_retries=1,
            ).with_structured_output(FeedbackLoopBlock)
        )

        # ── 3. Groq mixtral ──────────────────────────────────────────────────────
        eval_fallbacks.append(
            ChatGroq(
                model="llama-3.1-70b-versatile",
                temperature=0.0,
                groq_api_key=settings.GROQ_API_KEY,
                max_retries=1,
            ).with_structured_output(FeedbackLoopBlock)
        )

        # ── 4. Gemini Flash ──────────────────────────────────────────────────────
        if settings.GOOGLE_API_KEY:
            eval_fallbacks.append(
                ChatGoogleGenerativeAI(
                    model=settings.GEMINI_MODEL,
                    temperature=0.0,
                    google_api_key=settings.GOOGLE_API_KEY,
                    max_retries=1,
                ).with_structured_output(FeedbackLoopBlock)
            )

            # ── 5. Gemini Pro ─────────────────────────────────────────────────────
            eval_fallbacks.append(
                ChatGoogleGenerativeAI(
                    model="gemini-2.5-pro",
                    temperature=0.0,
                    google_api_key=settings.GOOGLE_API_KEY,
                    max_retries=1,
                ).with_structured_output(FeedbackLoopBlock)
            )

        # Note: HuggingFace intentionally excluded — with_structured_output()
        # raises NotImplementedError on HuggingFaceEndpoint

        return primary_eval.with_fallbacks(
            eval_fallbacks,
            exceptions_to_handle=(Exception,),
        )