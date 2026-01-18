"""
Vercel-specific entry point for the IKMS FastAPI application.
This file exports the FastAPI app instance for Vercel's serverless deployment.
"""

from src.api import app

# Vercel expects the ASGI app to be named 'app'
# This is already imported from src.api, so we just need to ensure it's available
__all__ = ["app"]
