"""A capture reports the id that will still work after the poll ends.

A photo starts life as an in-memory capture with an opaque handle, and a receipt
is written to the database during classification — which mints a BillFile id.
The pending entry deliberately survives that, as the capture-id -> file-id
mapping, so the client's next poll kept finding it and getting the capture
handle back. That handle names no row: opening the review screen with it
answered "Receipt review is not available", and discarding it deleted neither
the row nor the stored photo.
"""

import pytest

from backend.services import capture_service as cs

USER = {"id": "11111111-2222-3333-4444-555555555555", "access_token": "t"}
CAPTURE_ID = "capture-abc"
BILLFILE_ID = "0d1f5f9a-1111-2222-3333-444455556666"


@pytest.fixture(autouse=True)
def clean_pending():
    cs._pending.clear()
    yield
    cs._pending.clear()


def _entry(**overrides):
    entry = {
        "user_id": USER["id"],
        "kind": "receipt",
        "draft": {"merchant_name": "Stew Leonard's"},
        "location": None,
        "local_path": "/tmp/nonexistent/photo.jpg",
        "filename": "photo.jpg",
    }
    entry.update(overrides)
    return entry


@pytest.mark.asyncio
async def test_a_stored_receipt_reports_its_billfile_id():
    """The whole bug: the review screen is opened with whatever this returns."""
    cs._pending[CAPTURE_ID] = _entry(file_id=BILLFILE_ID)
    result = await cs.get_capture(USER, CAPTURE_ID)
    assert result["file_id"] == BILLFILE_ID
    assert result["kind"] == "receipt"


@pytest.mark.asyncio
async def test_an_unstored_price_tag_still_reports_its_capture_handle():
    """A shelf price is deliberately not written until confirmed, so there is no
    other id to give — reporting one would break the confirm call."""
    cs._pending[CAPTURE_ID] = _entry(kind="price_tag", draft={"item_description": "milk"})
    result = await cs.get_capture(USER, CAPTURE_ID)
    assert result["file_id"] == CAPTURE_ID


@pytest.mark.asyncio
async def test_a_photo_still_being_read_reports_its_capture_handle():
    cs._pending[CAPTURE_ID] = _entry(kind="processing", draft={})
    result = await cs.get_capture(USER, CAPTURE_ID)
    assert result["file_id"] == CAPTURE_ID
    assert result["kind"] == "processing"


@pytest.mark.asyncio
async def test_an_extraction_error_is_still_reported_alongside_the_id():
    cs._pending[CAPTURE_ID] = _entry(kind="unknown", draft={}, error="could not read")
    result = await cs.get_capture(USER, CAPTURE_ID)
    assert result["error"] == "could not read"


@pytest.mark.asyncio
async def test_another_users_capture_is_not_readable_from_memory():
    """The in-memory shortcut must not skip the ownership check the database
    query below it enforces."""
    cs._pending[CAPTURE_ID] = _entry(user_id="99999999-8888-7777-6666-555555555555")
    with pytest.raises(Exception):
        await cs.get_capture({"id": USER["id"], "access_token": "t"}, CAPTURE_ID)


def test_discarding_a_stored_receipt_is_refused_in_memory():
    """forget_pending returning False is what sends the caller to the real
    delete, so the row and the photo actually go."""
    cs._pending[CAPTURE_ID] = _entry(file_id=BILLFILE_ID)
    assert cs.forget_pending(CAPTURE_ID, USER["id"]) is False


def test_discarding_an_unstored_capture_is_handled_in_memory(tmp_path):
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"x")
    cs._pending[CAPTURE_ID] = _entry(kind="price_tag", local_path=str(photo))
    assert cs.forget_pending(CAPTURE_ID, USER["id"]) is True
    assert CAPTURE_ID not in cs._pending


# --- the race that made the fix above look ineffective ------------------------
#
# classify_photo sets kind BEFORE materialise() has uploaded the photo and
# written its row. The client polls every 1.5s and acts the instant kind leaves
# "processing", so it reliably saw "receipt" while file_id was still unset,
# routed to the review screen with the capture handle, and got a 404.

@pytest.mark.asyncio
async def test_a_receipt_stays_processing_until_it_is_stored():
    """The window between kind being set and the row existing."""
    cs._pending[CAPTURE_ID] = _entry(kind="receipt")  # no file_id yet
    result = await cs.get_capture(USER, CAPTURE_ID)
    assert result["kind"] == "processing", "client would route with a dead handle"


@pytest.mark.asyncio
async def test_a_receipt_becomes_ready_once_it_has_a_row():
    cs._pending[CAPTURE_ID] = _entry(kind="receipt", file_id=BILLFILE_ID)
    result = await cs.get_capture(USER, CAPTURE_ID)
    assert result["kind"] == "receipt"
    assert result["file_id"] == BILLFILE_ID


@pytest.mark.asyncio
async def test_a_price_tag_is_ready_without_being_stored():
    """A shelf price is deliberately never written until the user confirms, so
    waiting for a file_id would leave it processing forever."""
    cs._pending[CAPTURE_ID] = _entry(kind="price_tag", draft={"item_description": "milk"})
    result = await cs.get_capture(USER, CAPTURE_ID)
    assert result["kind"] == "price_tag"


@pytest.mark.asyncio
async def test_an_unclassifiable_photo_is_ready_without_being_stored():
    cs._pending[CAPTURE_ID] = _entry(kind="unknown", draft={})
    result = await cs.get_capture(USER, CAPTURE_ID)
    assert result["kind"] == "unknown"


# --- a photo that is neither, claimed to be one -------------------------------
#
# The classifier gives up and the user answers "which is this?" by hand. There
# is no extracted content behind that answer, and both branches used to dead-end.

def test_an_unreadable_photo_enriches_to_an_empty_tag_list():
    """Not a crash, but not usable either: the confirm card got zero rows, so it
    rendered no fields and then complained nothing was picked."""
    assert cs.enrich_price_tag_draft({}) == {"tags": []}


def test_a_draft_that_is_not_a_dict_is_survivable():
    assert cs.enrich_price_tag_draft(None) == {"tags": []}
    assert cs.enrich_price_tag_draft([]) == {"tags": []}


def test_a_bare_single_tag_is_still_wrapped():
    """Rows written before one photo could hold several tags."""
    out = cs.enrich_price_tag_draft({"item_description": "milk", "item_subtotal_price": 3.49})
    assert len(out["tags"]) == 1
    assert out["tags"][0]["item_description"] == "milk"


# --- answering "which is this?" by hand ---------------------------------------

@pytest.mark.asyncio
async def test_calling_it_a_receipt_returns_a_NEW_id(monkeypatch, tmp_path):
    """set_kind STORES the photo, which mints a BillFile id. The caller must
    route with that, not with the capture handle it passed in — chat used the
    handle, and the review screen answered "not available"."""
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"x")
    cs._pending[CAPTURE_ID] = _entry(kind="unknown", draft={}, local_path=str(photo))

    monkeypatch.setattr(cs, "_upload_photo_sync", lambda *a, **k: BILLFILE_ID)
    monkeypatch.setattr(cs, "_write_draft_sync", lambda *a, **k: None)
    monkeypatch.setattr(cs, "_read_draft_sync", lambda *a, **k: {})

    result = await cs.set_kind(USER, CAPTURE_ID, "receipt")
    assert result["file_id"] == BILLFILE_ID
    assert result["file_id"] != CAPTURE_ID, "the whole bug"
    assert result["kind"] == "receipt"


@pytest.mark.asyncio
async def test_calling_it_a_price_tag_keeps_the_capture_handle():
    """Nothing is stored, so there is no other id — and the confirm call that
    follows resolves this handle itself."""
    cs._pending[CAPTURE_ID] = _entry(kind="unknown", draft={})
    result = await cs.set_kind(USER, CAPTURE_ID, "price_tag")
    assert result["file_id"] == CAPTURE_ID
    assert result["kind"] == "price_tag"
    assert result["draft"] == {"tags": []}


@pytest.mark.asyncio
async def test_a_nonsense_kind_is_refused():
    cs._pending[CAPTURE_ID] = _entry(kind="unknown", draft={})
    with pytest.raises(ValueError):
        await cs.set_kind(USER, CAPTURE_ID, "cat")
