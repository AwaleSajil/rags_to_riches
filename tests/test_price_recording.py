"""What the agent records must be what the store reads back.

Two silent mismatches, both invisible because every failure mode here is a
missing value rather than an exception:

  * `check_price` built its draft with `item_quantity` / `item_quantity_unit`,
    the names migration 034 renamed on "PriceObservation". `record_observation`
    reads `size_value` / `size_unit`, so every price the user mentioned in chat
    was stored with no size AND embedded without one — a 12 oz jar and a 64 oz
    jar landing on the same vector, which is the exact distinction the whole
    comparison rests on. The confirm-card path had already been fixed for this;
    the agent path had not.

  * the metadata a retrieved observation travels with never carried
    "description", and `_is_the_same_sighting` compares against exactly that —
    so it compared "" to a real name, answered False every time, and the guard
    against a sighting being quoted back as evidence about itself never fired
    outside its own tests. Its fixture invented the key the producer omitted.
"""

from datetime import datetime, timedelta, timezone

import pytest

from backend.services import price_service as ps
from backend.vector_db_client import observation_metadata


class _Recorder:
    """Enough Supabase client to capture what an insert was handed."""

    def __init__(self):
        self.inserted = None

    def table(self, _name):
        return self

    def update(self, _record):
        return self

    def eq(self, *_args):
        return self

    def execute(self):
        return type("R", (), {"data": [dict(self.inserted or {}, id="obs-1")]})()

    def insert(self, record):
        self.inserted = record
        return self


def test_a_price_mentioned_in_chat_is_stored_with_its_size(monkeypatch):
    """The whole point of check_price's `quantity` argument."""
    import mcp_server

    recorder = _Recorder()
    captured = {}
    # Bound BEFORE the patch, or the spy calls itself.
    real_record = ps.record_observation

    def _record(client, config, draft, user_id, bill_file_id=None):
        captured["draft"] = draft
        return real_record(recorder, None, draft, user_id, bill_file_id)

    monkeypatch.setattr(mcp_server, "get_current_user_id", lambda: "u1")
    monkeypatch.setattr(mcp_server, "_account_config", lambda: None)
    monkeypatch.setattr(mcp_server, "_user_client", lambda: recorder)
    monkeypatch.setattr(ps, "record_observation", _record)
    monkeypatch.setattr(ps, "enrich_observation", lambda c, cfg, obs: obs)
    monkeypatch.setattr(
        ps, "compare_price",
        lambda *a, **k: {"item": "x", "purchases": [], "prior_observations": [],
                         "cautions": [], "size": None, "shelf_price": None},
    )

    answer = mcp_server.check_price(
        item_description="CHEERIOS TSTD WHL GRN",
        shelf_price=4.29,
        quantity=12,
        quantity_unit="oz",
    )
    # check_price catches everything and returns the text, so a blown-up run
    # would otherwise read as a pass with an empty capture.
    assert not answer.startswith("Error checking price"), answer

    # The names record_observation actually reads. Asserting on the stored row
    # rather than the draft, because that is where the value went missing.
    assert captured["draft"]["size_value"] == 12
    assert captured["draft"]["size_unit"] == "oz"
    assert recorder.inserted["size_value"] == 12
    assert recorder.inserted["size_unit"] == "oz"


def test_the_size_reaches_the_embedded_text(monkeypatch):
    """A sizeless vector is what made two differently-sized jars identical."""
    seen = {}

    def _embed(description, brand, size_text, config, observed_context=None):
        seen["size_text"] = size_text
        return None, None

    monkeypatch.setattr(ps, "build_observation_embedding", _embed)
    ps.record_observation(
        _Recorder(), None,
        {"item_description": "CHEERIOS", "size_value": 12, "size_unit": "OZ"},
        "u1",
    )
    assert seen["size_text"] == "12 oz"


# --- a sighting is not evidence about itself ---------------------------------

def _row(description="Fresh Organic Garnet Yams", price=3.99, ago_seconds=5):
    """One row shaped exactly as match_price_observations returns it."""
    return {
        "id": "obs-1",
        "bill_file_id": None,
        "merchant_name": "Stop & Shop",
        "location": "Norwalk",
        "item_description": description,
        "size_value": None,
        "size_unit": None,
        "unit_quantity_subtotal": None,
        "item_subtotal_price": price,
        "item_qualitative_description": None,
        "brand_name": None,
        "enriched_info": None,
        "note": None,
        "created_at": datetime.now(timezone.utc) - timedelta(seconds=ago_seconds),
        "score": 0.91,
    }


def test_the_metadata_carries_the_key_the_self_check_reads():
    meta = observation_metadata(_row(), "u1")
    assert meta["description"] == "Fresh Organic Garnet Yams"


def test_a_sighting_seconds_old_is_recognised_as_itself():
    """The card saves the price, then the agent asks about it with no id to
    exclude — so the row being asked ABOUT came back as evidence FOR it."""
    meta = observation_metadata(_row(), "u1")
    assert ps._is_the_same_sighting(meta, "Fresh Organic Garnet Yams", 3.99) is True


def test_a_genuine_revisit_is_still_evidence():
    meta = observation_metadata(_row(ago_seconds=7 * 24 * 3600), "u1")
    assert ps._is_the_same_sighting(meta, "Fresh Organic Garnet Yams", 3.99) is False


def test_a_different_item_at_the_same_price_is_not_the_same_sighting():
    meta = observation_metadata(_row(description="Loose Red Potatoes"), "u1")
    assert ps._is_the_same_sighting(meta, "Garnet Yams", 3.99) is False


@pytest.mark.parametrize("field", [
    "description", "shelf_price", "observed_on", "size_value", "size_unit",
    "unit_price", "merchant_name", "location", "tag_says", "note", "score",
])
def test_every_field_compare_price_reads_is_emitted(field):
    """compare_price builds a prior_observation entry from these. One missing is
    not an error — it is a blank on the card, or a guard that never fires."""
    assert field in observation_metadata(_row(), "u1")
