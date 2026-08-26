"""When the agent goes wrong, say something the reader can act on.

Both messages here replaced ones that cost real time. A failed query returned
Postgres's own wording, which never mentions the fix, so the agent burned two of
its twenty-five steps rediscovering it on nearly every run. And running out of
steps surfaced as "Something went wrong... try again in a moment", which sends
the user to repeat a question that will fail exactly the same way.
"""

import pytest

from backend.routers.chat import _friendly_error, _is_transient
from backend.sql_guard import ALLOWED_TABLES
from mcp_server import TABLE_NAMES, _explain_db_error


# --- unquoted table names ----------------------------------------------------

def test_the_casing_list_matches_the_allowlist():
    """sql_guard compares lowercased, so it cannot tell the agent how to spell a
    name. These are the same set; drift means suggesting a table that is not
    readable, or failing to suggest one that is."""
    assert {t.lower() for t in TABLE_NAMES} == set(ALLOWED_TABLES)


@pytest.mark.parametrize("table", TABLE_NAMES)
def test_a_missing_relation_names_the_quoted_form(table):
    """Postgres lowercases unquoted identifiers, so `FROM Transaction` reports
    `relation "transaction" does not exist` — true, and useless on its own."""
    message = _explain_db_error(
        Exception(f'relation "{table.lower()}" does not exist\nLINE 2: FROM {table}')
    )
    assert f'"{table}"' in message
    assert "double quotes" in message


def test_the_hint_says_to_re_run_the_same_query():
    """Without this the agent tends to rewrite the query from scratch, which
    spends another step and often introduces a second mistake."""
    message = _explain_db_error(Exception('relation "transaction" does not exist'))
    assert "SAME query" in message


def test_an_unrelated_database_error_is_passed_through():
    """Inventing a table-name hint for, say, a syntax error would send the agent
    chasing the wrong fix."""
    message = _explain_db_error(Exception("syntax error at or near \"SELCT\""))
    assert "SELCT" in message
    assert "double quotes" not in message


def test_a_genuinely_unknown_table_is_not_given_a_hint():
    message = _explain_db_error(Exception('relation "invoices" does not exist'))
    assert "double quotes" not in message


# --- running out of steps ----------------------------------------------------

class GraphRecursionError(Exception):
    """Same name as LangGraph's, which is half of how it is recognised."""


def test_running_out_of_steps_says_so_and_suggests_narrowing():
    message = _friendly_error(
        GraphRecursionError("Recursion limit of 25 reached without hitting a stop condition.")
    )
    assert "narrow" in message.lower()
    # The old text sent people to repeat a question that fails identically.
    assert "try again in a moment" not in message.lower()


def test_running_out_of_steps_is_recognised_by_type_alone():
    """The wording of LangGraph's message is not ours to depend on."""
    assert "narrow" in _friendly_error(GraphRecursionError("limit reached")).lower()


def test_running_out_of_steps_is_not_retried():
    """Nothing is wrong with the provider, and a retry spends the whole budget
    again to fail in the same place."""
    assert not _is_transient(
        GraphRecursionError("Recursion limit of 25 reached without hitting a stop condition.")
    )


def test_a_real_provider_outage_is_still_retried():
    assert _is_transient(Exception("503 Service Unavailable"))


# --- the deduped view --------------------------------------------------------
#
# A receipt and the bank line for the same purchase are two rows on purpose.
# The mobile list has always collapsed them before summing; the agent writes its
# own SQL and had nothing but one sentence of schema prose between it and a
# doubled total. Migration 041 makes the correct query the obvious one, which
# only works if the guard lets the agent run it and the schema doc names it.

def test_the_agent_may_query_the_deduped_view():
    """A rejected query would send the agent back to the table that double counts."""
    from backend.sql_guard import validate_select

    user_id = "11111111-1111-1111-1111-111111111111"
    validate_select(
        'SELECT category, SUM(amount) AS total FROM "TransactionDeduped" '
        f"WHERE user_id = '{user_id}' GROUP BY category",
        user_id,
    )


def test_the_schema_doc_sends_totals_to_the_view():
    """The agent only knows what this string tells it."""
    from mcp_server import get_schema_info

    doc = get_schema_info()
    assert '"TransactionDeduped"' in doc
    # Naming it is not enough — it has to say which one to reach for.
    assert "USE THIS" in doc


def test_the_view_is_not_offered_as_a_place_to_write():
    """It is a view; an INSERT against it must fail the guard like any other."""
    import pytest as _pytest

    from backend.sql_guard import validate_select

    user_id = "11111111-1111-1111-1111-111111111111"
    with _pytest.raises(Exception):
        validate_select(
            f"INSERT INTO \"TransactionDeduped\" (user_id) VALUES ('{user_id}')", user_id
        )
