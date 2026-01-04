from langchain_openai import ChatOpenAI
from src.config import get_settings

settings = get_settings()


def create_chat_model(temperature: float = 0.0) -> ChatOpenAI:
    """Create a LangChain v1 ChatOpenAI instance.

    Args:
        temperature: Model temperature (default: 0.0 for deterministic outputs).

    Returns:
        Configured ChatOpenAI instance.
    """

    return ChatOpenAI(
        model=settings.openai_model_name,
        api_key=settings.openai_api_key,
        temperature=temperature,
        streaming=True,
    )


def create_reasoning_model(temperature: float = 0.7) -> ChatOpenAI:
    """Create a ChatOpenAI instance for reasoning tasks.

    Args:
        temperature: Model temperature (default: 0.7 for more creative outputs).
    Returns:
        Configured ChatOpenAI instance.
    """

    return ChatOpenAI(
        model=settings.openai_reasoning_model_name,
        api_key=settings.openai_api_key,
        temperature=temperature,
        streaming=True,
    )
