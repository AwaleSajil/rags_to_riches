import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from backend.dependencies import get_current_user
from backend.schemas.chat import ChatRequest
from backend.services import config_service
from backend.services.rag_manager import rag_manager

router = APIRouter()


@router.post("/")
async def chat(body: ChatRequest, user: dict = Depends(get_current_user)):
    config = await config_service.get_config(user)
    if not config:
        raise HTTPException(status_code=400, detail="Account config required. Please configure your API key first.")

    rag = await rag_manager.get_or_create(user, config)

    async def event_generator():
        try:
            async for event in rag.chat(body.message):
                if event["type"] == "final":
                    content = event.get("content", "")
                    charts = []
                    # Extract chart JSON from ===CHART===...===ENDCHART=== markers
                    while "===CHART===" in content:
                        pre, rest = content.split("===CHART===", 1)
                        if "===ENDCHART===" in rest:
                            chart_json, content = rest.split("===ENDCHART===", 1)
                            charts.append(chart_json.strip())
                        else:
                            content = pre + rest
                            break
                    yield f"event: final\ndata: {json.dumps({'content': content.strip(), 'charts': charts})}\n\n"
                else:
                    yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as e:
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
