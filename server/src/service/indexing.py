from pathlib import Path
from src.core.retrieval import index_documents

async def index_pdf_file(file_path: Path) -> int:
    """Index a PDF file located at file_path into the vector store."""
    return  index_documents(file_path)