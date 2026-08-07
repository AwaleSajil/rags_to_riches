"""Item-name normalisation and store resolution.

normalize_item_name is the join between two very differently written
descriptions of the same product — a shelf tag and a receipt line. It is applied
to both sides, so a change here silently breaks matching for every existing
backfilled row; these tests pin the behaviour that makes the join work.
"""

import pytest

from backend.services.price_service import (
    MERCHANT_MATCH_RADIUS_M,
    haversine_m,
    nearest_merchant,
    normalize_item_name,
    observation_size,
    round_coordinate,
)
from backend.services.units import Size


# --- the join that the feature depends on ------------------------------------

def test_shelf_tag_and_receipt_line_land_on_the_same_key():
    """The whole point: OCR'd tag text and a receipt abbreviation must meet."""
    assert normalize_item_name("CHEERIOS TSTD WHL GRN 12OZ") == normalize_item_name(
        "Cheerios Toasted Whole Grain 12 oz"
    )


def test_container_words_are_dropped():
    """The jug is not the product."""
    assert normalize_item_name("Whole Milk 1 Gal Jug") == normalize_item_name("WHOLE MILK")


@pytest.mark.parametrize("variant,plain", [
    ("Organic Whole Milk", "Whole Milk"),
    ("Large Brown Eggs", "Brown Eggs"),
    ("Family Size Cereal", "Cereal"),
])
def test_product_variants_are_not_merged(variant, plain):
    """Organic milk and large eggs cost more than the conventional item. These
    look like marketing noise but naming them away would quote the user a price
    for something they did not buy — over-merging is the dangerous direction,
    since a miss just falls through to the trigram/embedding tiers."""
    assert normalize_item_name(variant) != normalize_item_name(plain)


def test_brand_abbreviations_are_left_to_the_fuzzy_tiers():
    """Documents a real limit rather than pretending it works: "GV" and "Great
    Value" do not collapse to the same exact key. Exact match is only the first
    of three tiers; trigram similarity is what catches this one."""
    assert normalize_item_name("Great Value Whole Milk") != normalize_item_name(
        "GV WHOLE MILK"
    )


def test_size_is_stripped_from_the_matching_key():
    """A 12oz and a 20oz box of the same cereal are the same *item*; the size
    lives in size_value/size_unit, not in the name, or they would never match."""
    assert normalize_item_name("Cheerios 12 oz") == normalize_item_name("Cheerios 20 oz")


def test_multipack_fragment_stripped():
    assert normalize_item_name("Sprite 6 x 12 fl oz") == normalize_item_name("Sprite")


@pytest.mark.parametrize("raw,expected", [
    ("Chicken Breast", "chicken breast"),
    ("CHKN BRD", "chicken bread"),
    ("  Eggs  ", "eggs"),
    ("2% Milk", "2% milk"),
])
def test_basic_normalisation(raw, expected):
    assert normalize_item_name(raw) == expected


def test_word_order_is_preserved():
    """"milk chocolate" and "chocolate milk" are different products, so the
    normaliser must not sort tokens."""
    assert normalize_item_name("Milk Chocolate") != normalize_item_name("Chocolate Milk")


@pytest.mark.parametrize("raw", ["", None, "   ", "!!!"])
def test_empty_inputs_give_empty_key(raw):
    assert normalize_item_name(raw) == ""


def test_noise_only_input_gives_empty_key():
    assert normalize_item_name("the of and") == ""


# --- size resolution ---------------------------------------------------------

def test_explicit_size_field_wins_over_the_name():
    assert observation_size("Cheerios 12 oz", "20 oz") == Size(20.0, "oz")


def test_falls_back_to_parsing_the_name():
    assert observation_size("Cheerios 12 oz", None) == Size(12.0, "oz")


def test_no_size_anywhere():
    assert observation_size("Cheerios", None) is None


# --- geography ---------------------------------------------------------------

def test_haversine_known_distance():
    # Two points ~1 degree of latitude apart are ~111km.
    assert haversine_m(40.0, -73.0, 41.0, -73.0) == pytest.approx(111_195, rel=0.01)


def test_haversine_zero_for_same_point():
    assert haversine_m(41.1, -73.4, 41.1, -73.4) == 0


def test_coordinates_rounded_to_about_eleven_metres():
    """Deliberately imprecise: enough to tell storefronts apart, not enough to
    identify a dwelling."""
    assert round_coordinate(41.123456789) == 41.1235
    assert round_coordinate(None) is None


STORES = [
    {"merchant_name": "Stop & Shop", "latitude": 41.1000, "longitude": -73.4000},
    {"merchant_name": "Walmart", "latitude": 41.2000, "longitude": -73.5000},
]


def test_resolves_the_store_you_are_standing_in():
    match = nearest_merchant(STORES, 41.10005, -73.40005)
    assert match["merchant_name"] == "Stop & Shop"
    assert match["distance_m"] < MERCHANT_MATCH_RADIUS_M


def test_picks_the_closer_of_two_stores():
    match = nearest_merchant(STORES, 41.1999, -73.4999)
    assert match["merchant_name"] == "Walmart"


def test_no_match_when_nothing_is_near():
    """Must return None rather than the least-far store, or every capture in a
    new town would be attributed to a shop miles away."""
    assert nearest_merchant(STORES, 42.0, -71.0) is None


def test_no_match_without_a_fix():
    assert nearest_merchant(STORES, None, None) is None


def test_no_match_with_no_known_stores():
    assert nearest_merchant([], 41.1, -73.4) is None


def test_malformed_rows_are_skipped_not_fatal():
    stores = [{"merchant_name": "Broken"}, *STORES]
    assert nearest_merchant(stores, 41.10005, -73.40005)["merchant_name"] == "Stop & Shop"


# --- hybrid search inputs ----------------------------------------------------
#
# Ranking lives in SQL (migration 020). These cover the Python side that feeds
# it, and the one decision that is easy to get backwards: what text gets
# embedded.


def test_vector_literal_is_what_pgvector_parses():
    from backend.services.price_service import to_vector_literal

    assert to_vector_literal([0.1, 0.2, 0.3]) == "[0.1,0.2,0.3]"


def test_empty_vector_is_none_not_an_empty_literal():
    """'[]' would be sent to Postgres and rejected as a malformed vector; None
    makes the SQL function skip the semantic axis instead."""
    from backend.services.price_service import to_vector_literal

    assert to_vector_literal(None) is None
    assert to_vector_literal([]) is None


def test_embedded_text_keeps_brand_and_size():
    """Brand and size are what separate two products with the same name. A
    name-only vector puts 'Great Value milk 1 gal' and 'Organic Valley milk
    12 oz' almost on top of each other."""
    from backend.services.price_service import embedding_text

    assert embedding_text("Fat Free Milk", "Great Value", "1 GAL") == (
        "Great Value Fat Free Milk 1 GAL"
    )


def test_embedded_text_is_not_the_normalized_name():
    """Normalization strips brand and packaging on purpose — that is right for
    the lexical key and wrong for the vector. Embedding the normalized form
    would give two views of one reduced string rather than two signals."""
    from backend.services.price_service import embedding_text, normalize_item_name

    raw, brand, size = "Fat Free Milk", "Great Value", "1 GAL"
    assert embedding_text(raw, brand, size) != normalize_item_name(raw)
    assert "Great Value" in embedding_text(raw, brand, size)


def test_embedding_text_tolerates_missing_parts():
    from backend.services.price_service import embedding_text

    assert embedding_text("Bananas", None, None) == "Bananas"
    assert embedding_text(None, None, None) == ""


def test_embedding_failure_does_not_raise():
    """A capture must survive a quota error: the row still saves and the lexical
    axis still works."""
    from backend.services.price_service import build_observation_embedding

    assert build_observation_embedding("Milk", "GV", "1 GAL", None) == (None, None)
    assert build_observation_embedding(None, None, None, {"llm_provider": "google"}) == (
        None,
        None,
    )


# --- recency ----------------------------------------------------------------
#
# Age is a weight, never a filter. A price from two years ago is still the honest
# answer when it is the only one — it just must not dominate the number quoted
# for today. On real data here the same item spans 0.83 to 2.99, so the summary
# has to carry a range and a confidence rather than one authoritative figure.

from datetime import date  # noqa: E402

TODAY = date(2026, 8, 1)


def test_weight_halves_every_half_life():
    from backend.services.price_service import PRICE_HALF_LIFE_DAYS, price_weight

    assert price_weight(TODAY, TODAY) == 1.0
    half_life_ago = date.fromordinal(TODAY.toordinal() - PRICE_HALF_LIFE_DAYS)
    assert round(price_weight(half_life_ago, TODAY), 3) == 0.5


def test_older_evidence_counts_far_less():
    """The 2024 banana must not out-vote the 2026 one."""
    from backend.services.price_service import price_weight

    recent = price_weight("2026-05-17", TODAY)
    old = price_weight("2024-09-01", TODAY)
    assert recent > old * 8


def test_future_dates_do_not_outweigh_reality():
    """A mistyped year is a typo, not a prophecy."""
    from backend.services.price_service import price_weight

    assert price_weight("2027-01-01", TODAY) == 1.0


def test_undated_records_are_dropped_not_guessed():
    from backend.services.price_service import price_weight

    assert price_weight(None, TODAY) == 0.0


def test_baseline_follows_the_recent_price():
    """Real bananas: 1.49 (2024) then 0.50, 0.50, 0.46. A plain mean says 0.74,
    which is not what a shopper would pay today."""
    from backend.services.price_service import weighted_baseline

    result = weighted_baseline([
        {"price": 0.46, "trans_date": "2026-05-17"},
        {"price": 0.50, "trans_date": "2025-03-20"},
        {"price": 0.50, "trans_date": "2025-03-14"},
        {"price": 1.49, "trans_date": "2024-09-01"},
    ], TODAY)
    assert result["typical"] == 0.46
    assert (result["low"], result["high"]) == (0.46, 1.49)
    assert result["count"] == 4


def test_baseline_reports_the_spread():
    """Real cilantro spans 0.83-2.99. Hiding that behind one number would be the
    confident lie this whole design exists to avoid."""
    from backend.services.price_service import weighted_baseline

    result = weighted_baseline([
        {"price": 0.83, "trans_date": "2026-07-31"},
        {"price": 2.99, "trans_date": "2026-03-06"},
    ], TODAY)
    assert result["high"] - result["low"] > 2


def test_only_ancient_evidence_yields_no_baseline():
    """Better to say 'not enough recent evidence' than to quote a 2023 price."""
    from backend.services.price_service import weighted_baseline

    assert weighted_baseline([{"price": 1.49, "trans_date": "2023-01-01"}], TODAY) is None


def test_no_records_yields_no_baseline():
    from backend.services.price_service import weighted_baseline

    assert weighted_baseline([], TODAY) is None
    assert weighted_baseline([{"price": 0, "trans_date": "2026-07-01"}], TODAY) is None


def test_calibrated_floor_is_the_measured_one():
    """Derived by scripts/calibrate_price_thresholds.py, in the direction
    matching actually runs — a shelf tag's phrasing against a stored receipt
    line. SAME pairs land 0.776-0.925, DIFF pairs 0.535-0.733; 0.75 sits in the
    gap. It has been guessed wrong three times, so it is pinned here."""
    from backend.services.price_service import MIN_SEMANTIC_SCORE

    assert MIN_SEMANTIC_SCORE == 0.75


def test_there_is_no_lexical_floor():
    """normalized_name and its trigram index were dropped in migration 023;
    matching is cosine-only. A lexical constant reappearing means someone
    reintroduced an axis the schema no longer supports."""
    import backend.services.price_service as ps

    assert not hasattr(ps, "MIN_LEXICAL_SCORE")


def test_enrichment_is_part_of_the_embedded_text():
    """The floor above only separates because enrichment is embedded. Dropping
    it from the text silently makes abbreviated lines unmatchable."""
    from backend.services.price_service import embedding_text

    text = embedding_text("GV LF 2 GAL", None, None, "A two-gallon container of Great Value low-fat milk.")
    assert "GV LF 2 GAL" in text and "low-fat milk" in text


def test_embedded_text_survives_missing_enrichment():
    """56% coverage today, so absence must degrade rather than break."""
    from backend.services.price_service import embedding_text

    assert embedding_text("BANANAS", None, None, None) == "BANANAS"
