import logging

from fastapi import APIRouter, Depends, HTTPException
from backend.schemas.auth import LoginRequest, RegisterRequest, AuthResponse, UserInfo
from backend.config import get_settings, Settings
from backend.dependencies import get_current_user, get_optional_user
from supabase import create_client

logger = logging.getLogger("moneyrag.routers.auth")

router = APIRouter()


def normalize_email(email: str | None) -> str | None:
    """Fold an address to the single form the User table is keyed on.

    Matches the unique index added in migration 019 (`lower(email)`). Without
    this, "Sam@x.com" logging in after signing up as "sam@x.com" would be the
    same auth account trying to write a second spelling into the mirror, and
    the index — not the user — would be the one to notice.

    Lowercasing only: dots and +suffixes are Gmail conventions, not email
    semantics, and collapsing them would merge addresses that really are
    different people elsewhere.
    """
    return email.strip().lower() if email else email


# Both layers can refuse a second account for one address, and they word it
# differently: the auth service rejects the signup outright, while the unique
# index on lower(email) (migration 019) raises 23505 if a mirror row somehow
# gets that far. Either way the user's situation is the same, so both map to one
# clear answer instead of a stack trace.
_DUPLICATE_EMAIL_MARKERS = (
    "already registered",
    "already been registered",
    "user_already_exists",
    "duplicate key value",
    "user_email_lower_key",
    "23505",
)


def _is_duplicate_email(error: Exception) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in _DUPLICATE_EMAIL_MARKERS)


@router.post("/login")
async def login(
    body: LoginRequest | None = None,
    user: dict | None = Depends(get_optional_user), 
    settings: Settings = Depends(get_settings)
):
    try:
        # If accessed via Swagger/Postman with raw credentials, generate the token first
        # `user` is a dict from a validated token, so it always carries an email —
        # the old `getattr(user, "email", None)` read None off the dict every time
        # and re-ran the password flow even for callers who sent a good token.
        if body and body.email and body.password and not user:
            logger.debug("Generating token dynamically for Swagger UI login email=%s", body.email)
            client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
            res = client.auth.sign_in_with_password({
                "email": body.email,
                "password": body.password,
            })
            user = {"id": res.user.id, "email": body.email, "access_token": res.session.access_token}

        if not user or "access_token" not in user:
            raise HTTPException(status_code=401, detail="Must provide either Bearer token or email/password credentials")

        # Initialize an authenticated client to bypass RLS policies
        from backend.dependencies import get_supabase
        client = get_supabase(user["access_token"])
        
        email = normalize_email(user["email"])
        logger.debug("Login sync for email=%s", email)
        client.table("User").upsert({
            "id": user["id"],
            "email": email,
            "hashed_password": "managed_by_supabase_auth",
        }).execute()

        return {
            "user": {"id": user["id"], "email": email},
            "access_token": user.get("access_token"),
        }
    except Exception as e:
        logger.error("Login sync failed: %s", e, exc_info=True)
        raise HTTPException(status_code=401, detail=f"Login sync failed: {e}")


@router.post("/register")
async def register(
    body: RegisterRequest | None = None,
    user: dict | None = Depends(get_optional_user), 
    settings: Settings = Depends(get_settings)
):
    try:
        # `user` is a dict from a validated token, so it always carries an email —
        # the old `getattr(user, "email", None)` read None off the dict every time
        # and re-ran the password flow even for callers who sent a good token.
        if body and body.email and body.password and not user:
            logger.debug("Generating token dynamically for Swagger UI register email=%s", body.email)
            client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
            res = client.auth.sign_up({
                "email": body.email,
                "password": body.password,
            })
            if not res.session:
                 raise HTTPException(status_code=400, detail="Signup succeeded but no session was returned (email confirmation may be required). Cannot sync User table without JWT.")
            user = {"id": res.user.id, "email": body.email, "access_token": res.session.access_token}

        if not user or "access_token" not in user:
            raise HTTPException(status_code=400, detail="Must provide either Bearer token or email/password credentials")

        # Initialize an authenticated client to bypass RLS policies
        from backend.dependencies import get_supabase
        client = get_supabase(user["access_token"])
        
        email = normalize_email(user["email"])
        logger.debug("Register sync for email=%s", email)
        client.table("User").upsert({
            "id": user["id"],
            "email": email,
            "hashed_password": "managed_by_supabase_auth",
        }).execute()

        return {
            "user": {"id": user["id"], "email": email},
            "message": "Account created successfully",
        }
    except HTTPException:
        # Already a considered response (including the 409 below) — re-raise it
        # rather than rewrapping it as a generic 400 with the status code
        # stringified into the detail.
        raise
    except Exception as e:
        if _is_duplicate_email(e):
            logger.info("Registration refused — address already in use")
            raise HTTPException(
                status_code=409,
                detail="An account with that email already exists. Try signing in instead.",
            )
        logger.error("Registration sync failed: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=f"Signup sync failed: {e}")


@router.post("/logout")
async def logout(user: dict = Depends(get_current_user)):
    logger.info("Logout for user_id=%s", user["id"])
    return {"message": "Logged out"}
