"""The schema the chat agent is handed must describe the database that exists.

This is not pedantry about documentation. The agent writes SQL straight from
this text, so a column named here that the database dropped becomes a failed
query, and a column the database has that is missing here is data the agent
cannot reach. Both happened: the description still named
`item_unit_subtotal_price` long after migration 023 renamed it, so every agent
query about a unit price errored, and "PriceObservation" was absent entirely.

The description used to be written out twice — once for the schema resource and
once inside the query_database docstring — and the copies drifted apart from the
database and from each other. They are one string now, and these tests hold it
to the real column names.
"""

import pytest

from backend.sql_guard import ALLOWED_TABLES
from mcp_server import SCHEMA_DOC, get_schema_info, query_database


# Names that existed once and no longer do. Migrations 022-030 dropped or
# renamed every one; any reappearance means the description drifted backwards.
REMOVED_NAMES = [
    "item_unit_subtotal_price",  # renamed to unit_quantity_subtotal (023)
    "normalized_name",           # dropped, matching is semantic now (023)
    "MarketPrice",               # table dropped (027)
    "MerchantLocation",          # table dropped (030)
    "langchain_pg_embedding",    # table dropped (026)
]

# One representative column per table, chosen because each is easy to get wrong:
# the renamed one, the two added ones, and the two that must never be treated as
# money the user spent.
REQUIRED_NAMES = [
    "unit_quantity_subtotal",
    # Dropped in 024 as a bad PARSE, restored in 034 as a confirmable fact. The
    # objection to 024 stands — a guessed size is worse than none — which is why
    # these are nullable and why the description fallback is documented as a
    # guess. What changed is that a human can now correct one.
    "size_value",
    "size_unit",
    "item_quantity_unit",
    "item_savings",
    "discount_total",
    "savings_total",
    "item_qualitative_description",
    "note",
]


@pytest.mark.parametrize("name", REMOVED_NAMES)
def test_schema_never_names_something_that_was_removed(name):
    assert name not in SCHEMA_DOC


@pytest.mark.parametrize("name", REQUIRED_NAMES)
def test_schema_names_the_columns_that_exist(name):
    assert name in SCHEMA_DOC


@pytest.mark.parametrize(
    "table",
    ["Transaction", "TransactionDetail", "PriceObservation",
     "BillFile", "CSVFile", "TransactionLink"],
)
def test_every_readable_table_is_described(table):
    """A table the guard permits but the description omits is unreachable: the
    agent has no way to learn it exists."""
    assert table.lower() in ALLOWED_TABLES
    assert table in SCHEMA_DOC


def test_query_database_carries_the_same_schema():
    """The tool's description and the schema resource must not diverge again."""
    assert SCHEMA_DOC in (query_database.__doc__ or "")
    assert get_schema_info() == SCHEMA_DOC


def test_agent_is_told_price_observations_are_not_spending():
    """A shelf price is not a purchase. Summed into a spending total it invents
    money the user never spent, which is the whole reason the table is separate
    from Transaction."""
    assert "NOT a purchase" in SCHEMA_DOC
    assert "never" in SCHEMA_DOC.lower()


def test_agent_is_warned_off_selecting_embeddings():
    """3072 floats per row, useless as text, and they crowd out the answer."""
    assert "embedding" in SCHEMA_DOC
    assert "never select" in SCHEMA_DOC.lower()
