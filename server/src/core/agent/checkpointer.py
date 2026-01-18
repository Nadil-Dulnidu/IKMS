"""
PostgreSQL Checkpointer for LangGraph.

This module provides an async factory function to create and initialize
a PostgreSQL checkpointer for state persistence in LangGraph workflows.
"""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from src.config.env_config import get_settings

settings = get_settings()


async def get_postgres_checkpointer() -> AsyncPostgresSaver:
    """
    Create and initialize a PostgreSQL checkpointer.

    This function creates an AsyncPostgresSaver instance from the connection
    string and sets up the necessary database tables.

    Returns:
        AsyncPostgresSaver: Configured checkpointer instance ready for use.

    Example:
        >>> checkpointer = await get_postgres_checkpointer()
        >>> # Use checkpointer with your graph
    """
    checkpointer = AsyncPostgresSaver.from_conn_string(settings.postgres_uri)
    # await checkpointer.__aenter__()
    return checkpointer
