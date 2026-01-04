from  src.config import get_settings
from fastapi import UploadFile, File

from pathlib import Path

settings = get_settings()

async def save_uploaded_file(file: UploadFile = File(...)) -> Path:
    """Save an uploaded file to the configured upload directory."""
    upload_dir = Path(settings.file_upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename
    contents = await file.read()
    file_path.write_bytes(contents)
    
    return file_path