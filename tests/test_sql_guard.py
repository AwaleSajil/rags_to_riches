"""The chat agent writes its own SQL and runs it on an RLS-bypassing connection,
so this guard is the boundary between a prompt injection and the whole database.
Every case here is a query the model could plausibly be talked into emitting.
"""

import pytest

from backend.sql_guard import ALLOWED_TABLES, SqlGuardError, validate_select
from conftest import OTHER_USER_ID, TEST_USER_ID

U = TEST_USER_ID
TX = '"Transaction"'


def ok(sql: str) -> None:
    validate_select(sql, U)


def rejected(sql: str) -> str:
    with pytest.raises(SqlGuardError) as excinfo:
        validate_select(sql, U)
    return str(excinfo.value)


# --- queries the agent legitimately needs -----------------------------------

@pytest.mark.parametrize("sql", [
    f"SELECT category, SUM(amount) FROM {TX} WHERE user_id = '{U}' GROUP BY category",
    f"SELECT trans_date, amount FROM {TX} WHERE user_id='{U}' ORDER BY trans_date DESC LIMIT 5",
    f"SELECT amount FROM {TX} WHERE '{U}' = user_id",
    f"WITH t AS (SELECT * FROM {TX} WHERE user_id='{U}') SELECT category FROM t",
    f"SELECT * FROM (SELECT amount FROM {TX} WHERE user_id='{U}') s WHERE amount > 10",
    f'SELECT b.s3_key FROM {TX} t JOIN "BillFile" b ON t.source_bill_file_id=b.id'
    f" WHERE t.user_id='{U}'",
    f'SELECT d.item_description FROM {TX} t JOIN "TransactionDetail" d'
    f" ON d.transaction_id=t.id AND d.user_id='{U}' WHERE t.user_id='{U}'",
    f"SELECT amount FROM {TX} WHERE user_id='{U}'"
    f" UNION ALL SELECT item_total_price FROM \"TransactionDetail\" WHERE user_id='{U}'",
])
def test_legitimate_queries_pass(sql):
    ok(sql)


# --- writes must never reach the database -----------------------------------

@pytest.mark.parametrize("sql", [
    f"DELETE FROM {TX} WHERE user_id='{U}'",
    f"UPDATE {TX} SET amount=0 WHERE user_id='{U}'",
    f"DROP TABLE {TX}",
    f"INSERT INTO {TX} (user_id) VALUES ('{U}')",
    f"TRUNCATE {TX}",
])
def test_write_statements_rejected(sql):
    rejected(sql)


def test_write_hidden_in_cte_rejected():
    """RETURNING makes a DELETE look like it produces rows; it is still a write."""
    rejected(
        f"WITH d AS (DELETE FROM {TX} WHERE user_id='{U}' RETURNING 1) SELECT * FROM d"
    )


@pytest.mark.parametrize("sql", [
    f"SELECT 1 FROM {TX} WHERE user_id='{U}';DROP TABLE {TX}",
    # Newlines instead of spaces defeated the old space-delimited keyword scan.
    f"SELECT 1 FROM {TX} WHERE user_id='{U}';\nDELETE\nFROM {TX}",
])
def test_multiple_statements_rejected(sql):
    assert "exactly one statement" in rejected(sql)


# --- tenant isolation --------------------------------------------------------

def test_missing_filter_rejected():
    rejected(f"SELECT user_id, amount FROM {TX}")


def test_negated_filter_rejected():
    """`!=` mentions the id but selects precisely everyone else's rows."""
    rejected(f"SELECT * FROM {TX} WHERE user_id != '{U}'")


def test_another_users_id_rejected():
    rejected(f"SELECT * FROM {TX} WHERE user_id = '{OTHER_USER_ID}'")


def test_id_in_comment_rejected():
    """The old guard was a substring test, which a comment satisfied."""
    rejected(f"SELECT user_id, amount FROM {TX} -- {U}")


def test_id_projected_but_not_filtered_rejected():
    """Mentioning the id in the SELECT list returns every row."""
    rejected(f"SELECT '{U}' = user_id AS mine, amount FROM {TX}")


def test_sibling_cte_cannot_shelter_behind_a_filtered_one():
    rejected(
        f"WITH a AS (SELECT 1 AS n FROM {TX} WHERE user_id='{U}'),"
        f" b AS (SELECT * FROM {TX}) SELECT * FROM b"
    )


def test_union_branch_must_be_filtered_too():
    rejected(f"SELECT amount FROM {TX} WHERE user_id='{U}' UNION ALL SELECT amount FROM {TX}")


def test_unfiltered_subquery_rejected():
    rejected(f"SELECT * FROM (SELECT amount FROM {TX}) s WHERE '{U}' <> ''")


# --- table allowlist ---------------------------------------------------------

def test_account_config_not_readable():
    """AccountConfig stores the provider API key in plaintext; reading it would
    put the key straight into the model's context."""
    assert "not readable" in rejected(f"SELECT api_key FROM \"AccountConfig\" WHERE user_id='{U}'")


def test_auth_schema_not_readable():
    rejected(f"SELECT * FROM auth.users WHERE user_id='{U}'")


def test_allowlist_covers_what_the_tools_need():
    assert {"transaction", "transactiondetail", "billfile"} <= ALLOWED_TABLES
    assert "accountconfig" not in ALLOWED_TABLES
    assert "user" not in ALLOWED_TABLES


# --- dangerous functions -----------------------------------------------------

@pytest.mark.parametrize("call", [
    "pg_read_file('/etc/passwd')",
    "pg_sleep(999)",
    "lo_import('/etc/passwd')",
])
def test_dangerous_functions_rejected(call):
    rejected(f"SELECT {call} FROM {TX} WHERE user_id='{U}'")


# --- input handling ----------------------------------------------------------

@pytest.mark.parametrize("sql", ["", "   ", "not sql at all !!!"])
def test_junk_input_rejected(sql):
    rejected(sql)


def test_empty_user_id_rejected():
    """A missing user id must fail closed, never run unscoped."""
    with pytest.raises(SqlGuardError):
        validate_select(f"SELECT * FROM {TX} WHERE user_id='{U}'", "")
