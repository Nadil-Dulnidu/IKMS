import uvicorn
from src.config.env_config import get_settings

settings = get_settings()

if __name__ == "__main__":
    uvicorn.run(
        "src.api:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.server_reload,
    )
