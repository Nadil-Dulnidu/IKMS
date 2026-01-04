from jose import jwt
from jose.exceptions import JWTError
from fastapi import HTTPException, Request, status
import httpx
from src.config.env_config import get_settings

settings = get_settings()

JWKS_URL = settings.clerk_jwks_url
CLERK_ISSUER = settings.clerk_issuer

_jwks_cache = None


async def get_jwks():
    global _jwks_cache
    if not _jwks_cache:
        async with httpx.AsyncClient() as client:
            res = await client.get(JWKS_URL)
            _jwks_cache = res.json()
    return _jwks_cache


async def verify_clerk_token(request: Request):
    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
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
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        )

    return payload
