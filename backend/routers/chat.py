import json
import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from backend.dependencies import get_current_user
from backend.schemas.chat import ChatRequest
from backend.services import config_service, conversation_service
from backend.services.rag_manager import rag_manager

logger = logging.getLogger("moneyrag.routers.chat")

router = APIRouter()


def _friendly_error(exc: Exception) -> str:
    """Turn a raw provider exception into a short, human message for the chat UI."""
    low = str(exc).lower()
    if any(k in low for k in ("resource_exhausted", "429", "quota", "too many requests", "rate limit")):
        return (
            "You've hit your AI provider's rate limit (the Gemini free tier allows "
            "~20 requests/day). Please wait a moment and try again, or enable billing "
            "on your Google Cloud project."
        )
    if "no longer available" in low or "not_found" in low or ("404" in low and "model" in low):
        return (
            "The selected AI model isn't available for your API key. Open Settings and "
            "switch the model to “gemini-3-flash-preview”."
        )
    if any(k in low for k in ("api key not valid", "api_key_invalid", "permission_denied", "unauthenticated", "401", "403")):
        return "Your AI provider API key looks invalid or unauthorized — please re-check it in Settings."
    if any(k in low for k in ("deadline", "timeout", "timed out")):
        return "The AI provider took too long to respond. Please try again."
    return "Something went wrong while generating a response. Please try again in a moment."


@router.post("")
async def chat(body: ChatRequest, user: dict = Depends(get_current_user)):
    logger.debug("Chat request from user_id=%s | message=%s", user["id"], body.message[:100])

    config = await config_service.get_config(user)
    if not config:
        logger.warning("No config found for user_id=%s — returning 400", user["id"])
        raise HTTPException(status_code=400, detail="Account config required. Please configure your API key first.")

    rag = await rag_manager.get_or_create(user, config)

    # --- Conversation setup: resume an existing one or start a new one ---
    if body.conversation_id:
        conversation_id = body.conversation_id
    else:
        conv = await conversation_service.create_conversation(user)
        conversation_id = conv["id"]
    logger.debug("Chat in conversation_id=%s for user_id=%s", conversation_id, user["id"])

    # Load prior turns for agent context, then persist this user message.
    prior = await conversation_service.get_messages(user, conversation_id)
    history = conversation_service.to_agent_history(prior)
    await conversation_service.add_message(user, conversation_id, "user", body.message)
    await conversation_service.set_title_from_first_message(user, conversation_id, body.message)

    async def event_generator():
        # Let the client know which conversation this maps to (esp. brand-new ones).
        yield f"event: conversation\ndata: {json.dumps({'conversation_id': conversation_id})}\n\n"

        event_count = 0
        start = time.perf_counter()
        try:
            async for event in rag.chat(body.message, history=history):
                event_count += 1

                if event["type"] == "final":
                    content = event.get("content", "")

                    charts = []
                    while "===CHART===" in content:
                        pre, rest = content.split("===CHART===", 1)
                        if "===ENDCHART===" in rest:
                            chart_json, after = rest.split("===ENDCHART===", 1)
                            charts.append(chart_json.strip())
                            content = pre + after
                        else:
                            content = pre + rest
                            break

                    images = []
                    while "===IMAGES===" in content:
                        pre, rest = content.split("===IMAGES===", 1)
                        if "===ENDIMAGES===" in rest:
                            images_json, after = rest.split("===ENDIMAGES===", 1)
                            content = pre + after
                            try:
                                images.extend(json.loads(images_json.strip()))
                            except json.JSONDecodeError:
                                logger.warning("Failed to parse images JSON")
                        else:
                            content = pre + rest
                            break

                    pending_transactions = []
                    while "===CONFIRM_TX===" in content:
                        pre, rest = content.split("===CONFIRM_TX===", 1)
                        if "===ENDCONFIRM_TX===" in rest:
                            tx_json, after = rest.split("===ENDCONFIRM_TX===", 1)
                            content = pre + after
                            try:
                                pending_transactions.append(json.loads(tx_json.strip()))
                            except json.JSONDecodeError:
                                logger.warning("Failed to parse pending transaction JSON")
                        else:
                            content = pre + rest
                            break

                    final_content = content.strip()

                    # Persist the assistant turn so it survives reloads/restarts.
                    try:
                        await conversation_service.add_message(
                            user, conversation_id, "assistant", final_content,
                            charts=charts or None,
                            images=images or None,
                            pending_transactions=pending_transactions or None,
                        )
                    except Exception as e:
                        logger.error("Failed to persist assistant message: %s", e, exc_info=True)

                    yield (
                        "event: final\ndata: "
                        + json.dumps({
                            "content": final_content,
                            "charts": charts,
                            "images": images,
                            "pendingTransactions": pending_transactions,
                        })
                        + "\n\n"
                    )
                else:
                    yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "SSE stream complete for user_id=%s conv=%s — %d events in %.1fms",
                user["id"], conversation_id, event_count, elapsed_ms,
            )
            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            logger.error("SSE stream error for user_id=%s: %s", user["id"], e, exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': _friendly_error(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
