import logging

from fastapi import APIRouter, Depends, HTTPException
from backend.dependencies import get_current_user
from backend.services import conversation_service

logger = logging.getLogger("moneyrag.routers.conversations")

router = APIRouter()


@router.get("")
async def list_conversations(user: dict = Depends(get_current_user)):
    return {"conversations": await conversation_service.list_conversations(user)}


@router.post("")
async def create_conversation(user: dict = Depends(get_current_user)):
    return await conversation_service.create_conversation(user)


@router.get("/{conversation_id}/messages")
async def get_messages(conversation_id: str, user: dict = Depends(get_current_user)):
    return {"messages": await conversation_service.get_messages(user, conversation_id)}


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str, user: dict = Depends(get_current_user)):
    await conversation_service.delete_conversation(user, conversation_id)
    return {"message": "Conversation deleted"}
