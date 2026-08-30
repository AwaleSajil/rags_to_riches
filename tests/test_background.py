"""Fire-and-forget tasks must survive garbage collection.

The event loop only holds a weak reference to a task, so a bare
`asyncio.create_task(...)` whose result is discarded can vanish mid-await. That
is how an upload ends up stuck at "processing" forever.
"""

import asyncio
import gc

import pytest

from backend.services import background


@pytest.mark.asyncio
async def test_spawned_task_survives_a_collection():
    started = asyncio.Event()
    finished = asyncio.Event()

    async def work():
        started.set()
        await asyncio.sleep(0.05)
        finished.set()

    background.spawn(work(), name="test:survives")   # deliberately not stored
    await started.wait()

    gc.collect()          # what would collect an unreferenced task
    await asyncio.sleep(0.2)

    assert finished.is_set(), "task was collected before it finished"


@pytest.mark.asyncio
async def test_completed_tasks_are_released():
    """The holder must not become a leak of its own."""
    before = background.pending_count()

    async def work():
        return None

    background.spawn(work(), name="test:releases")
    await asyncio.sleep(0.05)

    assert background.pending_count() == before


@pytest.mark.asyncio
async def test_failure_is_logged_and_contained(caplog):
    """A crash in detached work must not take down the caller, but must be
    visible — a silently swallowed exception is how this stays unnoticed."""
    async def boom():
        raise RuntimeError("ingestion exploded")

    with caplog.at_level("ERROR"):
        background.spawn(boom(), name="test:boom")
        await asyncio.sleep(0.05)

    assert "test:boom" in caplog.text
    assert "ingestion exploded" in caplog.text
    assert background.pending_count() == 0
