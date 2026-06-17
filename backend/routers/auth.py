import logging

from fastapi import APIRouter, Depends, HTTPException
from backend.schemas.auth import LoginRequest, RegisterRequest, AuthResponse, UserInfo
from backend.config import get_settings, Settings
from backend.dependencies import get_current_user
from supabase import create_client

logger = logging.getLogger("moneyrag.routers.auth")

router = APIRouter()


@router.post("/login")
async def login(
    body: LoginRequest | None = None,
    user: dict | None = Depends(get_current_user), 
    settings: Settings = Depends(get_settings)
):
    try:
        # If accessed via Swagger/Postman with raw credentials, generate the token first
        if body and body.email and body.password and (not user or getattr(user, "email", None) is None):
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
        
        logger.debug("Login sync for email=%s", user["email"])
        client.table("User").upsert({
            "id": user["id"],
            "email": user["email"],
            "hashed_password": "managed_by_supabase_auth",
        }).execute()
        
        return {
            "user": {"id": user["id"], "email": user["email"]},
            "access_token": user.get("access_token"),
        }
    except Exception as e:
        logger.error("Login sync failed: %s", e, exc_info=True)
        raise HTTPException(status_code=401, detail=f"Login sync failed: {e}")


@router.post("/register")
async def register(
    body: RegisterRequest | None = None,
    user: dict | None = Depends(get_current_user), 
    settings: Settings = Depends(get_settings)
):
    try:
        if body and body.email and body.password and (not user or getattr(user, "email", None) is None):
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
        
        logger.debug("Register sync for email=%s", user["email"])
        client.table("User").upsert({
            "id": user["id"],
            "email": user["email"],
            "hashed_password": "managed_by_supabase_auth",
        }).execute()

        return {
            "user": {"id": user["id"], "email": user["email"]},
            "message": "Account created successfully",
        }
    except Exception as e:
        logger.error("Registration sync failed: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=f"Signup sync failed: {e}")


@router.post("/logout")
async def logout(user: dict = Depends(get_current_user)):
    logger.info("Logout for user_id=%s", user["id"])
    return {"message": "Logged out"}
