from jose import jwt
from jose.exceptions import JWTError
from fastapi import HTTPException, Request, status
import httpx
from src.config.env_config import get_settings
from src.config.logging import get_logger

logger = get_logger(__name__)

settings = get_settings()

JWKS_URL = settings.clerk_jwks_url
CLERK_ISSUER = settings.clerk_issuer

_jwks_cache = None


async def get_jwks():
    global _jwks_cache
    if not _jwks_cache:
        try:
            async with httpx.AsyncClient() as client:
                res = await client.get(JWKS_URL)
                _jwks_cache = res.json()
                logger.info(
                    "JWKS fetched successfully", extra={"action": "jwks_fetch_success"}
                )
        except Exception as e:
            logger.error(
                f"Failed to fetch JWKS: {str(e)}",
                extra={"action": "jwks_fetch_error"},
                exc_info=True,
            )
            raise
    return _jwks_cache


async def verify_clerk_token(request: Request):
    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
        logger.warning(
            "Authentication failed: Missing or invalid authorization header",
            extra={"action": "auth_missing_token"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing token",
        )

    token = auth_header.replace("Bearer ", "")

    jwks = await get_jwks()

    try:
        payload = jwt.decode(
            token,
            jwks,
            algorithms=["RS256"],
            issuer=CLERK_ISSUER,
            options={"verify_aud": False},
        )

        user_id = payload.get("sub", "unknown")
        logger.info(
            "Token verified successfully",
            extra={"user": user_id, "action": "auth_success"},
        )

        return payload
    except JWTError as e:
        logger.warning(
            f"Token verification failed: {str(e)}",
            extra={"action": "auth_token_invalid"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        )
