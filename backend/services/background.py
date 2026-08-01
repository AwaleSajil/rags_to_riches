"""Fire-and-forget task tracking.

`asyncio.create_task` returns a task the event loop holds only a *weak*
reference to. If the caller drops the result — which is the whole point of a
fire-and-forget — the task can be garbage collected mid-await, and the work
silently disappears part-way through. Ingestion and receipt enrichment both run
this way, so a collected task shows up as an upload that never finishes.

Keeping a strong reference until the task completes is the documented fix.
"""

import asyncio
import logging

logger = logging.getLogger("moneyrag.services.background")

_background_tasks: set[asyncio.Task] = set()


def spawn(coro, *, name: str) -> asyncio.Task:
    """Schedule `coro` and keep it referenced until it finishes.

    `name` is used for logging so an exception in a detached task is traceable
    to what started it, rather than surfacing as a bare "Task exception was
    never retrieved" with no context.
    """
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)

    def _done(finished: asyncio.Task) -> None:
        _background_tasks.discard(finished)
        if finished.cancelled():
            logger.warning("Background task '%s' was cancelled", name)
            return
        exc = finished.exception()
        if exc is not None:
            logger.error("Background task '%s' failed: %s", name, exc, exc_info=exc)

    task.add_done_callback(_done)
    return task


def pending_count() -> int:
    """How many spawned tasks are still running (used by tests/diagnostics)."""
    return len(_background_tasks)
