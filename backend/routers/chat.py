import json
import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from backend.dependencies import get_current_user
from backend.schemas.chat import ChatRequest
from backend.services import config_service
from backend.services.rag_manager import rag_manager

logger = logging.getLogger("moneyrag.routers.chat")

router = APIRouter()


@router.post("/")
async def chat(body: ChatRequest, user: dict = Depends(get_current_user)):
    logger.debug("Chat request from user_id=%s | message=%s", user["id"], body.message[:100])

    logger.debug("Fetching config for user_id=%s", user["id"])
    config = await config_service.get_config(user)
    if not config:
        logger.warning("No config found for user_id=%s — returning 400", user["id"])
        raise HTTPException(status_code=400, detail="Account config required. Please configure your API key first.")
    logger.debug("Config loaded — provider=%s, model=%s", config.get("llm_provider"), config.get("decode_model"))

    logger.debug("Getting/creating RAG instance for user_id=%s", user["id"])
    rag = await rag_manager.get_or_create(user, config)
    logger.debug("RAG instance ready for user_id=%s", user["id"])

    async def event_generator():
        event_count = 0
        start = time.perf_counter()
        try:
            logger.debug("Starting SSE stream for user_id=%s", user["id"])
            async for event in rag.chat(body.message):
                event_count += 1
                event_type = event.get("type", "unknown")
                logger.debug(
                    "SSE event #%d type=%s for user_id=%s",
                    event_count, event_type, user["id"],
                )

                if event["type"] == "final":
                    content = event.get("content", "")
                    logger.debug(
                        "Final event — content length=%d chars for user_id=%s",
                        len(content), user["id"],
                    )
                    charts = []
                    # Extract chart JSON from ===CHART===...===ENDCHART=== markers
                    while "===CHART===" in content:
                        pre, rest = content.split("===CHART===", 1)
                        if "===ENDCHART===" in rest:
                            chart_json, after = rest.split("===ENDCHART===", 1)
                            charts.append(chart_json.strip())
                            content = pre + after
                            logger.debug("Extracted chart JSON (%d chars)", len(chart_json))
                        else:
                            content = pre + rest
                            logger.warning("Found ===CHART=== without matching ===ENDCHART===")
                            break

                    images = []
                    # Extract image URLs from ===IMAGES===...===ENDIMAGES=== markers
                    while "===IMAGES===" in content:
                        pre, rest = content.split("===IMAGES===", 1)
                        if "===ENDIMAGES===" in rest:
                            images_json, after = rest.split("===ENDIMAGES===", 1)
                            content = pre + after
                            try:
                                urls = json.loads(images_json.strip())
                                images.extend(urls)
                                logger.debug("Extracted %d image URLs", len(urls))
                            except json.JSONDecodeError:
                                logger.warning("Failed to parse images JSON")
                        else:
                            content = pre + rest
                            logger.warning("Found ===IMAGES=== without matching ===ENDIMAGES===")
                            break

                    logger.debug(
                        "Final response: %d charts, %d images extracted, content length=%d",
                        len(charts), len(images), len(content.strip()),
                    )
                    yield f"event: final\ndata: {json.dumps({'content': content.strip(), 'charts': charts, 'images': images})}\n\n"
                else:
                    yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "SSE stream complete for user_id=%s — %d events in %.1fms",
                user["id"], event_count, elapsed_ms,
            )
            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            logger.error(
                "SSE stream error for user_id=%s: %s", user["id"], e, exc_info=True,
            )
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
