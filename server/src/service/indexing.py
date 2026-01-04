from pathlib import Path
from src.core.retrieval import index_documents


async def index_pdf_file(file_path: Path, user_id: str) -> int:
    """Index a PDF file located at file_path into the user's namespace in the vector store."""
    return index_documents(file_path, user_id)
