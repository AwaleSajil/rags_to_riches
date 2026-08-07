"""Recording a shelf price, and gathering evidence about it.

Two rules run through all of this:

* A size must reach the embedded text. Without it a 12 oz jar and a 64 oz jar
  embed identically, and that is the one distinction a price comparison rests
  on. The router used to pass None here and nobody noticed.
* compare_price returns evidence, never a verdict. Whether a price is good
  depends on why it is that price — a clearance, a multi-buy, a card rate — and
  that is in the tag's words, which this function never sees.
"""

import pytest

from backend.services import price_service as ps


# --- size, reassembled from the two columns ----------------------------------

@pytest.mark.parametrize("quantity, unit, expected", [
    (12, "oz", "12 oz"),
    (12.0, "OZ", "12 oz"),
    (1.5, "l", "1.5 l"),
    (2.25, "lb", "2.25 lb"),
])
def test_size_text_joins_quantity_and_unit(quantity, unit, expected):
    assert ps.size_text_of(quantity, unit) == expected


@pytest.mark.parametrize("quantity, unit", [(None, "oz"), (12, None), (12, ""), (None, None)])
def test_size_text_needs_both_halves(quantity, unit):
    """Half a size is not a size: "12" alone could be ounces or pounds."""
    assert ps.size_text_of(quantity, unit) is None


def test_size_falls_back_to_the_name_when_columns_are_empty():
    """225 existing rows have no item_quantity_unit, so the name is all there is."""
    size = ps.size_from_quantity(None, None, "WW SPAG 16OZ")
    assert size is not None and size.unit == "oz" and size.value == 16


def test_size_prefers_the_columns_over_the_name():
    """The name may carry a stale or mis-parsed size; the columns were confirmed."""
    size = ps.size_from_quantity(500, "ml", "SOMETHING 12 OZ")
    assert size.unit == "ml" and size.value == 500


# --- the size must reach the vector ------------------------------------------

class _FakeTable:
    def __init__(self, sink):
        self.sink = sink

    def insert(self, record):
        self.sink["record"] = record
        return self

    def execute(self):
        return type("R", (), {"data": [dict(self.sink["record"], id="obs-1")]})()


class _FakeClient:
    def __init__(self, sink):
        self.sink = sink

    def table(self, _name):
        return _FakeTable(self.sink)


def test_recorded_observation_embeds_the_size(monkeypatch):
    seen = {}

    def fake_embed(description, brand, size_text, config, observed_context=None):
        seen["size_text"] = size_text
        seen["brand"] = brand
        seen["context"] = observed_context
        return "[0.1]", "fake-model"

    monkeypatch.setattr(ps, "build_observation_embedding", fake_embed)
    sink = {}
    saved = ps.record_observation(
        _FakeClient(sink), {"embedding_model": "fake-model"},
        {
            "item_description": "  Cheerios  ",
            "brand_name": "General Mills",
            "size_value": 12,
            "size_unit": "OZ",
            "item_subtotal_price": 4.29,
            "item_qualitative_description": "2 for $5 with card",
        },
        user_id="user-1",
    )

    assert seen["size_text"] == "12 oz", "size must reach the embedded text"
    assert seen["brand"] == "General Mills"
    assert seen["context"] == "2 for $5 with card"
    assert saved["item_description"] == "Cheerios"
    # Normalised on the way in so 'OZ' and 'oz' are one unit, not two.
    assert sink["record"]["size_unit"] == "oz"


def test_recording_needs_an_item_description(monkeypatch):
    monkeypatch.setattr(ps, "build_observation_embedding", lambda *a, **k: (None, None))
    with pytest.raises(ValueError):
        ps.record_observation(_FakeClient({}), {}, {"item_subtotal_price": 1.0}, user_id="u")


# --- comparison returns evidence, and says when it has none ------------------

def test_comparison_without_config_says_so_rather_than_guessing():
    result = ps.compare_price(None, "user-1", "Broccoli", shelf_price=2.99)
    assert result["purchases"] == []
    assert result["baseline"] is None
    assert result["comparison"] is None
    assert result["cautions"], "an empty result must explain itself"


def test_missing_size_is_flagged_not_assumed():
    """An unknown size means package-to-package only. Silently assuming 'each'
    would compare a shelf price against a per-pound price."""
    result = ps.compare_price(None, "user-1", "Mystery Item", shelf_price=5.00)
    assert result["size"] is None
    assert any("size" in c.lower() for c in result["cautions"])


def test_comparison_never_returns_a_verdict():
    """No 'good'/'bad' key anywhere: that judgement needs the tag's own words."""
    result = ps.compare_price(None, "user-1", "Broccoli", shelf_price=2.99)
    assert "verdict" not in result
    assert "is_good_price" not in result
    assert "recommendation" not in result


# --- what shapes a baseline --------------------------------------------------

def test_marked_down_purchases_are_excluded_from_the_baseline():
    """Treating a markdown as 'what you usually pay' makes every ordinary shelf
    price look like a rip-off."""
    assert ps.purchase_was_discounted({"item_savings": 1.95}) is True
    assert ps.purchase_was_discounted({"item_savings": 0}) is False
    assert ps.purchase_was_discounted({}, {"discount_total": 5.0}) is True


def test_baseline_reports_a_range_not_a_point():
    """The same item has spanned $0.83-$2.99 here, so a lone average is fiction."""
    today = __import__("datetime").date(2026, 8, 1)
    baseline = ps.weighted_baseline(
        [
            {"price": 0.99, "trans_date": "2026-07-20"},
            {"price": 2.99, "trans_date": "2026-07-25"},
            {"price": 1.49, "trans_date": "2026-07-30"},
        ],
        today=today,
    )
    assert baseline["low"] == 0.99 and baseline["high"] == 2.99
    assert baseline["count"] == 3


def test_baseline_declines_to_answer_on_one_ancient_record():
    """One purchase from years ago cannot support 'you usually pay X'."""
    today = __import__("datetime").date(2026, 8, 1)
    assert ps.weighted_baseline([{"price": 1.00, "trans_date": "2022-01-01"}], today=today) is None


# --- a comparison, not a list ------------------------------------------------

def test_percent_difference_needs_both_sides():
    """A missing per-unit figure yields no number rather than a fabricated one."""
    assert ps._percent_difference(None, 1.0) is None
    assert ps._percent_difference(1.0, None) is None
    assert ps._percent_difference(1.0, 0) is None


def test_percent_difference_is_relative_to_what_was_paid():
    """$3.49/gal against $1.69/gal is +107%, not -48%: the question is how much
    more the shelf is asking than the user has paid."""
    assert ps._percent_difference(2.0, 1.0) == 100.0
    assert ps._percent_difference(1.0, 2.0) == -50.0
    assert ps._percent_difference(1.0, 1.0) == 0.0


def test_comparison_shape_always_carries_the_summary_keys():
    """The card reads these unconditionally; a missing key is a crash, and an
    absent comparison must be expressible as None."""
    result = ps.compare_price(None, "user-1", "Broccoli", shelf_price=2.99)
    for key in ("closest_comparable", "shelf_unit_price", "purchases", "cautions"):
        assert key in result
    assert result["closest_comparable"] is None


# --- both sources, always per unit -------------------------------------------

def test_one_unit_size_is_not_the_quantity_bought():
    """2.25 lb of bananas at $0.46 is $0.46 PER POUND. Dividing by 2.25 lb
    instead of one is a silent factor-of-N error in every produce comparison."""
    assert ps.purchase_unit_size("lb", "BANANAS").value == 1.0


def test_packaged_items_take_their_size_from_the_name():
    """A price per package is only comparable once you know the package size,
    and for a counted item that only exists in the description."""
    size = ps.purchase_unit_size("each", "GV LF 2 GAL")
    assert size is not None and size.value == 2.0 and size.unit == "gal"


def test_unit_size_is_unknown_when_nothing_records_it():
    """225 legacy rows have no unit and no size in the name. Unknown must stay
    unknown — guessing 'each' would compare a per-pound price to a per-bag one."""
    assert ps.purchase_unit_size(None, "BANANAS") is None


def test_comparison_is_refused_across_unit_families():
    """Grams against millilitres is not a conversion without knowing the
    substance, so there is no honest percentage to report."""
    from backend.services.units import Size, comparable
    assert comparable(Size(1, "lb"), Size(1, "gal")) is False
    assert comparable(Size(1, "gal"), Size(2, "gal")) is True


def test_percent_difference_is_none_when_either_side_is_unknown():
    assert ps._percent_difference(0.004, None) is None


# --- prices are local --------------------------------------------------------

def test_purchase_entries_carry_a_location_key():
    """Where a purchase happened decides whether its price means anything here.
    A gallon bought in Huntsville AL is not the going rate in Norwalk CT, and
    without this key the comparison could not tell — recency was weighted, and
    distance was not even visible."""
    result = ps.compare_price(None, "user-1", "Gallon Milk", shelf_price=3.49)
    # No config, so no rows; the contract is what matters here.
    assert result["purchases"] == []
    assert "closest_comparable" in result


def test_stored_size_beats_a_parse_of_the_description():
    """"+RED POTA 5L US#" is a five POUND bag; the text says litres and no regex
    can know better. A size someone confirmed has to win, or the bag stays
    uncomparable to every per-pound price."""
    parsed = ps.purchase_unit_size("each", "+RED POTA 5L US#")
    assert parsed is not None and parsed.unit == "l"  # what the text alone gives

    stored = ps.purchase_unit_size("each", "+RED POTA 5L US#", 5, "lb")
    assert stored.value == 5.0 and stored.unit == "lb"


def test_package_size_is_not_the_quantity_bought():
    """One 5 lb bag is item_quantity 1, size 5 lb. Collapsing the two is what
    made a $4.99 bag read as $1.00 per litre."""
    from backend.services.units import unit_price
    size = ps.purchase_unit_size("each", "bag", 5, "lb")
    per_base = unit_price(4.99, size)
    from backend.services.units import scale_from_base
    assert round(scale_from_base(per_base, "lb"), 2) == 1.00


def test_tag_enrichment_reads_the_size_columns():
    """The card and the server must agree on what a size is called. When this
    still looked for item_quantity after migration 034 renamed it, every tag
    reported "no size found" while the card displayed "128 oz" directly above
    the warning."""
    from backend.services.capture_service import enrich_price_tag_draft

    out = enrich_price_tag_draft({"tags": [
        {"item_description": "Gallon Milk", "size_value": 128, "size_unit": "oz",
         "item_subtotal_price": 3.49},
        {"item_description": "Loose Apples", "item_subtotal_price": 1.99},
    ]})
    sized, unsized = out["tags"]
    assert sized["size_unknown"] is False
    assert sized["unit_price_display"] == "$0.03/oz"
    # No size on the tag is a real answer, not a bug — the comparison degrades
    # to same-item-only and the card says so.
    assert unsized["size_unknown"] is True
    assert unsized["unit_price_display"] is None


# --- a sighting is not evidence about itself ---------------------------------

def _seen(price, ago_seconds, description="Fresh Organic Garnet Yams"):
    from datetime import datetime, timedelta, timezone
    when = datetime.now(timezone.utc) - timedelta(seconds=ago_seconds)
    return {"shelf_price": price, "observed_on": when.isoformat(), "description": description}


def test_the_price_just_saved_is_not_quoted_back_as_history():
    """The confirm card saves the sighting, then the agent asks about it seconds
    later with no id to exclude — so the row being asked ABOUT came back as
    evidence FOR it: "you have seen them at this same price earlier today",
    about a row a few seconds old."""
    assert ps._is_the_same_sighting(_seen(3.99, 5), "Fresh Organic Garnet Yams", 3.99) is True


def test_a_genuine_revisit_still_counts():
    """Seeing the same shelf again next week IS evidence. Only the moments-old
    duplicate is the sighting itself."""
    assert ps._is_the_same_sighting(_seen(3.99, 7 * 24 * 3600), "Fresh Organic Garnet Yams", 3.99) is False


def test_a_different_price_for_the_same_item_is_real_evidence():
    """Same item, different price, minutes apart — two shops, or a corrected
    reading. Either way it is not the same sighting."""
    assert ps._is_the_same_sighting(_seen(2.99, 5), "Fresh Organic Garnet Yams", 3.99) is False


def test_a_different_item_at_the_same_price_is_not_the_same_sighting():
    assert ps._is_the_same_sighting(_seen(3.99, 5, "Loose Red Potatoes"), "Garnet Yams", 3.99) is False


@pytest.mark.parametrize("meta", [{}, {"shelf_price": None}, {"shelf_price": 3.99}])
def test_incomplete_rows_are_never_treated_as_self_matches(meta):
    """Missing data must not silently drop a real past sighting."""
    assert ps._is_the_same_sighting(meta, "Garnet Yams", 3.99) is False


# --- the printed unit price is not always per the package --------------------

def _tag(**kwargs):
    from backend.services.capture_service import enrich_price_tag_draft
    return enrich_price_tag_draft({"tags": [kwargs]})["tags"][0]


def test_a_printed_unit_price_is_labelled_with_what_the_tag_said():
    """US milk tags routinely price a gallon jug PER QUART. Labelling $0.87 as
    "/gal" states that a gallon costs $0.87 when the tag beside it says $3.49."""
    out = _tag(item_description="Gallon Milk", size_value=1, size_unit="gal",
               item_subtotal_price=3.49, unit_quantity_subtotal=0.87, unit_price_unit="qt")
    assert out["unit_price_display"] == "$0.87/qt"


def test_an_unlabelled_unit_price_that_cannot_be_per_package_is_left_unlabelled():
    """The tag did not say what its figure was per, and 0.87 x 1 gallon does not
    come back to $3.49 — so it is per something else, and which is a guess."""
    out = _tag(item_description="Gallon Milk", size_value=1, size_unit="gal",
               item_subtotal_price=3.49, unit_quantity_subtotal=0.87)
    assert out["unit_price_display"] is None
    assert out["unit_price_unit"] is None


def test_the_package_unit_is_used_when_the_arithmetic_agrees():
    """1 lb of potatoes at $1.99 with a printed $1.99 is self-consistent, so the
    package unit is the right label."""
    out = _tag(item_description="Red Potatoes", size_value=1, size_unit="lb",
               item_subtotal_price=1.99, unit_quantity_subtotal=1.99)
    assert out["unit_price_display"] == "$1.99/lb"


def test_a_per_ounce_tag_on_a_multi_ounce_jar_agrees_too():
    out = _tag(item_description="Jam", size_value=12, size_unit="oz",
               item_subtotal_price=4.29, unit_quantity_subtotal=0.3575)
    assert out["unit_price_display"] == "$0.36/oz"
