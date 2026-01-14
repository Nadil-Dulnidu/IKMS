from src.config import get_settings
from fastapi import UploadFile, File
from pathlib import Path
import os
import base64
import re
from src.config.logging import get_logger

logger = get_logger(__name__)

settings = get_settings()


# async def save_uploaded_file(file: UploadFile = File(...)) -> Path:
#     """Save an uploaded file to the configured upload directory."""
#     upload_dir = Path(settings.file_upload_dir)
#     upload_dir.mkdir(parents=True, exist_ok=True)

#     file_path = upload_dir / file.filename
#     contents = await file.read()
#     file_path.write_bytes(contents)

#     return file_path


async def save_file_from_data_url(data_url: str, filename: str = None) -> Path:
    """
    Save a file from a data URL (base64-encoded).

    Args:
        data_url: Data URL in format "data:mime/type;base64,<encoded_data>"
        filename: Optional custom filename

    Returns:
        Path object pointing to the saved file

    Raises:
        ValueError: If data URL format is invalid

    Example:
        data_url = "data:application/pdf;base64,JVBERi0xLjQK..."
        file_path = await save_file_from_data_url(data_url, "document.pdf")
    """
    # Create upload directory if it doesn't exist
    upload_dir = Path(settings.file_upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Parse data URL
    # Format: data:[<mediatype>][;base64],<data>
    match = re.match(r"data:([^;]+);base64,(.+)", data_url)
    if not match:
        logger.error(
            "Invalid data URL format provided",
            extra={"action": "file_save_invalid_format"},
        )
        raise ValueError("Invalid data URL format")

    mime_type = match.group(1)
    base64_data = match.group(2)

    # Decode base64 data
    try:
        file_content = base64.b64decode(base64_data)
    except Exception as e:
        logger.error(
            f"Failed to decode base64 data: {str(e)}",
            extra={"action": "file_save_decode_error"},
            exc_info=True,
        )
        raise ValueError(f"Failed to decode base64 data: {e}")

    # Generate filename if not provided
    if not filename:
        # Extract extension from MIME type
        ext_map = {
            "application/pdf": "pdf",
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/jpg": "jpg",
            "text/plain": "txt",
        }
        ext = ext_map.get(mime_type, "bin")
        filename = f"uploaded_file.{ext}"

    # Save to disk
    file_path = upload_dir / filename
    try:
        file_path.write_bytes(file_content)
        logger.info(
            f"File saved successfully: {filename}",
            extra={
                "action": "file_save_success",
                "uploaded_filename": filename,
                "file_size": len(file_content),
            },
        )
    except Exception as e:
        logger.error(
            f"Failed to save file: {str(e)}",
            extra={"action": "file_save_error", "uploaded_filename": filename},
            exc_info=True,
        )
        raise

    return file_path
