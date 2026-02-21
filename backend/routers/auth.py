from fastapi import APIRouter, Depends, HTTPException
from backend.schemas.auth import LoginRequest, RegisterRequest, AuthResponse, UserInfo
from backend.config import get_settings, Settings
from backend.dependencies import get_current_user
from supabase import create_client

router = APIRouter()


@router.post("/login", response_model=AuthResponse)
async def login(body: LoginRequest, settings: Settings = Depends(get_settings)):
    try:
        client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        res = client.auth.sign_in_with_password({
            "email": body.email,
            "password": body.password,
        })

        # Upsert User table (mirrors app.py lines 219-226)
        try:
            client.table("User").upsert({
                "id": res.user.id,
                "email": body.email,
                "hashed_password": "managed_by_supabase_auth",
            }).execute()
        except Exception as sync_e:
            print(f"Warning: Could not sync user: {sync_e}")

        return AuthResponse(
            user=UserInfo(id=res.user.id, email=body.email),
            access_token=res.session.access_token,
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Login failed: {e}")


@router.post("/register")
async def register(body: RegisterRequest, settings: Settings = Depends(get_settings)):
    try:
        client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        res = client.auth.sign_up({
            "email": body.email,
            "password": body.password,
        })

        if res.user:
            try:
                client.table("User").upsert({
                    "id": res.user.id,
                    "email": body.email,
                    "hashed_password": "managed_by_supabase_auth",
                }).execute()
            except Exception:
                pass

        return {
            "user": {"id": res.user.id, "email": body.email},
            "message": "Account created successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Signup failed: {e}")


@router.post("/logout")
async def logout(user: dict = Depends(get_current_user)):
    return {"message": "Logged out"}
