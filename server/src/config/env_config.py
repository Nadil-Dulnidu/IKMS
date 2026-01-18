from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI Configuration
    openai_api_key: str
    openai_model_name: str
    openai_embedding_model_name: str
    openai_reasoning_model_name: str

    # File Upload Configuration
    file_upload_dir: str

    # Pinecone Configuration
    pinecone_api_key: str
    pinecone_index_name: str

    # Retrieval Configuration
    retrieval_k: int = 4

    # Clerk Configuration
    clerk_issuer: str
    clerk_jwks_url: str

    # Logging Configuration
    log_dir: str
    log_file: str
    max_bytes: int
    backup_count: int
    log_level: str
    log_dir_enable: bool

    # server
    server_port: int
    server_host: str
    server_reload: bool

    # Postgres Configuration
    postgres_uri: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Create a singleton settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the application settings."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
