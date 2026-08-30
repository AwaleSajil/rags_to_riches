"""A shelf photo is not stored until it is confirmed.

The value of a price-tag photo is the observation it becomes, not the image. One
that was read and then walked away from used to leave a BillFile row in the Files
tab forever, and the only way to clear it was to find it and delete it by hand.

So a capture is held in memory with its bytes in a temp directory, and the row is
written at the moment the user commits. A RECEIPT is the deliberate exception: it
is a financial record whose review screen needs a durable handle, and losing one
costs far more than a stray file.
"""

import os
import tempfile
import time

import pytest

from backend.services import capture_service as cs


@pytest.fixture(autouse=True)
def clean_pending():
    cs._pending.clear()
    yield
    cs._pending.clear()


def _hold(user_id="user-1", age=0.0):
    """Put a capture in the held state, with its bytes on disk."""
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, "tag.jpg")
    with open(path, "wb") as f:
        f.write(b"jpeg")
    capture_id = f"cap-{len(cs._pending)}"
    cs._pending[capture_id] = {
        "user_id": user_id, "local_path": path, "filename": "tag.jpg",
        "kind": "price_tag", "draft": {"tags": []}, "error": None,
        "created_at": time.time() - age,
    }
    return capture_id, temp_dir


def test_discarding_leaves_nothing_behind():
    """Nothing was stored, so nothing has to be cleaned up anywhere else."""
    capture_id, temp_dir = _hold()
    assert cs.forget_pending(capture_id, "user-1") is True
    assert capture_id not in cs._pending
    assert not os.path.exists(temp_dir), "the photo's bytes must go too"


def test_discarding_another_users_capture_is_refused():
    capture_id, temp_dir = _hold(user_id="owner")
    assert cs.forget_pending(capture_id, "someone-else") is False
    assert capture_id in cs._pending
    assert os.path.exists(temp_dir)


def test_abandoned_captures_are_swept():
    """A capture nobody came back for must not keep its bytes on disk."""
    fresh, fresh_dir = _hold(age=0)
    stale, stale_dir = _hold(age=cs.PENDING_TTL_SECONDS + 1)

    assert cs.sweep_pending() == 1
    assert fresh in cs._pending and os.path.exists(fresh_dir)
    assert stale not in cs._pending and not os.path.exists(stale_dir)


def test_sweeping_is_safe_when_nothing_is_stale():
    _hold()
    assert cs.sweep_pending() == 0
    assert len(cs._pending) == 1


# Callers pass either a held capture id or a real BillFile id and must not have
# to know which. What this file used to assert was that ANY unknown id passes
# through unchanged, which is where the foreign-key violation came from — see
# test_an_expired_capture_says_so_instead_of_breaking_the_insert below, which
# replaces it.


# --- confirming stores it, exactly once --------------------------------------

class _FakeUpload:
    """Stands in for the storage upload + BillFile insert."""

    def __init__(self):
        self.calls = 0

    def __call__(self, user, local_path, filename):
        self.calls += 1
        return f"billfile-{self.calls}"


@pytest.mark.asyncio
async def test_confirming_stores_the_photo_once_per_capture(monkeypatch):
    """One photo can hold a tag per product, and the card confirms them one at a
    time with the SAME handle. Every call must resolve to the same BillFile —
    storing twice would duplicate the photo, and handing back the capture id
    would fail the observation's foreign key."""
    upload = _FakeUpload()
    monkeypatch.setattr(cs, "_upload_photo_sync", upload)
    monkeypatch.setattr(cs, "_write_draft_sync", lambda *a, **k: None)

    capture_id, temp_dir = _hold()
    user = {"id": "user-1", "access_token": "t"}

    first = await cs.materialise(user, capture_id)
    second = await cs.materialise(user, capture_id)
    third = await cs.materialise(user, capture_id)

    assert first == second == third == "billfile-1"
    assert upload.calls == 1, "the photo must be stored once, not once per tag"
    assert not os.path.exists(temp_dir), "the temp copy goes once it is in storage"


@pytest.mark.asyncio
async def test_a_confirmed_capture_is_no_longer_silently_discardable(monkeypatch):
    """Once stored there is a row and a storage object. Dropping the in-memory
    entry would orphan both, so the caller has to delete them properly."""
    monkeypatch.setattr(cs, "_upload_photo_sync", _FakeUpload())
    monkeypatch.setattr(cs, "_write_draft_sync", lambda *a, **k: None)

    capture_id, _ = _hold()
    await cs.materialise({"id": "user-1", "access_token": "t"}, capture_id)

    assert cs.forget_pending(capture_id, "user-1") is False


@pytest.mark.asyncio
async def test_materialising_someone_elses_capture_is_refused(monkeypatch):
    monkeypatch.setattr(cs, "_upload_photo_sync", _FakeUpload())
    capture_id, _ = _hold(user_id="owner")
    with pytest.raises(ValueError):
        await cs.materialise({"id": "someone-else", "access_token": "t"}, capture_id)


# --- where the photo was taken -----------------------------------------------

def test_the_place_is_held_with_the_capture():
    """Location arrives on the upload response, but the client then POLLS for the
    draft — so it has to survive on the held capture, not just in that one reply.

    It is a NAME, never coordinates: the fix and the reverse geocode happen on
    the device and only the label is sent."""
    capture_id, _ = _hold()
    cs._pending[capture_id]["location"] = "Main St, Norwalk"

    result = cs._pending_result(capture_id, cs._pending[capture_id])
    assert result["location"] == "Main St, Norwalk"


def test_a_capture_without_a_place_is_still_a_capture():
    """Declining location costs the shop name on a price and nothing else."""
    capture_id, _ = _hold()
    result = cs._pending_result(capture_id, cs._pending[capture_id])
    assert result["location"] is None
    assert result["kind"] == "price_tag"


# --- a capture that is no longer held ----------------------------------------

@pytest.mark.asyncio
async def test_an_expired_capture_says_so_instead_of_breaking_the_insert(monkeypatch):
    """A card stays on screen indefinitely; the capture behind it does not.

    materialise() used to hand an unknown id straight back on the assumption it
    was already a real BillFile. For a swept capture it is not, and the id went
    into the observation insert and violated its foreign key — surfacing a raw
    Postgres error on the card instead of "take the photo again"."""
    monkeypatch.setattr(cs, "_billfile_exists_sync", lambda user, file_id: False)

    with pytest.raises(ValueError) as caught:
        await cs.materialise({"id": "user-1", "access_token": "t"}, "swept-capture-id")
    assert "no longer available" in str(caught.value)


@pytest.mark.asyncio
async def test_an_id_that_is_already_a_stored_photo_passes_through(monkeypatch):
    """Confirming a second tag, or opening a photo from the Files tab, arrives
    here with a real BillFile id and must not be treated as expired."""
    monkeypatch.setattr(cs, "_billfile_exists_sync", lambda user, file_id: True)

    resolved = await cs.materialise({"id": "user-1", "access_token": "t"}, "real-billfile-id")
    assert resolved == "real-billfile-id"


def test_the_hold_outlasts_a_lunch_break():
    """An hour was short enough that coming back to a card later reliably hit
    the expired path."""
    assert cs.PENDING_TTL_SECONDS >= 6 * 60 * 60


# --- the same photo, captured twice -------------------------------------------
#
# Chat can pick an existing image, not just shoot a new one, so a byte-identical
# capture is reachable. It has to be caught BEFORE classify_photo runs: the
# vision call is the expensive part, and a duplicate found afterwards has
# already been paid for.

def test_the_hash_is_checked_before_the_vision_call(monkeypatch, tmp_path):
    """The saving only happens if the lookup precedes classification."""
    import asyncio, json
    from backend.services import capture_service as cs

    photo = tmp_path / "receipt.jpg"
    photo.write_bytes(b"\xff\xd8\xff\xe0 same bytes both times")

    spawned = []
    monkeypatch.setattr(cs.background, "spawn", lambda coro, name: spawned.append(name) or coro.close())
    monkeypatch.setattr(cs.config_service, "get_config", _async({"llm_provider": "google"}))
    monkeypatch.setattr(
        cs, "_billfile_by_hash_sync",
        lambda user, h: {"id": "existing-1", "filename": "r.jpg", "kind": "receipt",
                         "raw_ocr_string": json.dumps({"merchant_name": "TESCO"})},
    )

    result = asyncio.run(cs.capture_photo(
        {"id": "u1", "access_token": "t"}, str(photo), "receipt.jpg", location=None,
    ))

    assert result["file_id"] == "existing-1", "should hand back the photo already stored"
    assert result["already_have"] is True
    assert result["draft"]["merchant_name"] == "TESCO", "reuses what was already read"
    assert spawned == [], "no vision call for a photo we already have"


def test_a_new_photo_still_goes_through_classification(monkeypatch, tmp_path):
    import asyncio
    from backend.services import capture_service as cs

    photo = tmp_path / "new.jpg"
    photo.write_bytes(b"\xff\xd8\xff\xe0 never seen before")

    spawned = []
    monkeypatch.setattr(cs.background, "spawn", lambda coro, name: spawned.append(name) or coro.close())
    monkeypatch.setattr(cs.config_service, "get_config", _async({"llm_provider": "google"}))
    monkeypatch.setattr(cs, "_billfile_by_hash_sync", lambda user, h: None)

    result = asyncio.run(cs.capture_photo(
        {"id": "u1", "access_token": "t"}, str(photo), "new.jpg", location=None,
    ))

    assert result["kind"] == "processing"
    assert result.get("already_have") is None
    assert len(spawned) == 1, "the vision call must still run for a new photo"


def _async(value):
    async def _inner(*args, **kwargs):
        return value
    return _inner
