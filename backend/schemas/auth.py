from pydantic import BaseModel


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str


class UserInfo(BaseModel):
    id: str
    email: str


class AuthResponse(BaseModel):
    user: UserInfo
    access_token: str
