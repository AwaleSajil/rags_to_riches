"""Validation for the raw SQL the chat agent is allowed to run.

The MCP tools let the LLM write its own SQL, and the connection they use is the
`DATABASE_URL` superuser role, which bypasses RLS. Anything reaching the
database from there has to be proven safe *first*, because the model's context
contains attacker-influenceable text (OCR'd receipts, CSV merchant names), so a
prompt injection is a realistic way for hostile SQL to be generated.

The old guard was a substring test (`user_id in query`). That passes for
`... WHERE user_id != '<uuid>'` and for a uuid parked in a `--` comment, and the
keyword blocklist it sat next to was space-delimited, so `SELECT 1;\nDROP TABLE x`
slipped through. This module parses the statement instead and checks the tree.

Four layers, because none of them is individually sufficient:

1. Exactly one statement, and it must be a SELECT.
2. Only the user's own data tables may be referenced — `AccountConfig` holds
   plaintext provider API keys, and reading it would drop the key straight into
   the model's context.
3. A real `user_id = '<current user>'` equality has to appear in the tree, not
   just the literal somewhere in the text.
4. The caller runs it on a read-only connection (see `readonly_cursor`), so even
   a bypass of 1-3 cannot write.
"""

from __future__ import annotations

import logging

import sqlglot
from sqlglot import exp

logger = logging.getLogger("moneyrag.sql_guard")

# Tables the chat agent may read. Deliberately excludes "AccountConfig" (holds
# the API key — encrypted at rest, but still nothing the agent has any business
# reading), "User", and anything in auth.*.
ALLOWED_TABLES = frozenset({
    "transaction",
    "transactiondetail",
    "transactionlink",
    "billfile",
    "csvfile",
    # Shelf prices the user photographed. Readable because "is this a good
    # price?" is answered by comparing it against purchase history — but it is
    # not spending, and the schema given to the agent says so explicitly.
    "priceobservation",
})

# Node types that are never part of a read-only query. sqlglot maps anything it
# can't classify (COPY, SET, CALL, ...) to Command, so that catch-all matters.
_FORBIDDEN_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.TruncateTable, exp.Merge, exp.Command, exp.Copy, exp.Grant,
)

# Postgres functions that read files, open connections, or burn wall-clock time.
_FORBIDDEN_FUNCTIONS = frozenset({
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "pg_sleep", "pg_sleep_for", "pg_sleep_until",
    "lo_import", "lo_export", "dblink", "dblink_exec",
    "query_to_xml", "pg_terminate_backend", "pg_cancel_backend",
    "set_config", "current_setting",
})


class SqlGuardError(ValueError):
    """Raised when a query is rejected. The message goes back to the LLM."""


def _statement_tables(statement: exp.Expression) -> set[str]:
    """Every real table referenced, ignoring CTE aliases defined in the query."""
    cte_names = {
        cte.alias_or_name.lower()
        for cte in statement.find_all(exp.CTE)
        if cte.alias_or_name
    }
    tables = set()
    for table in statement.find_all(exp.Table):
        name = (table.name or "").lower()
        if name and name not in cte_names:
            tables.add(name)
    return tables


def _is_user_id_equality(node: exp.Expression, user_id: str) -> bool:
    """True for an EQ node meaning `user_id = '<user_id>'`, in either order.

    Only EQ counts, so `!=` and `<>` fail, and only a literal matching the
    authenticated id counts, so another user's id fails.
    """
    for eq in node.find_all(exp.EQ):
        for column_side, literal_side in ((eq.left, eq.right), (eq.right, eq.left)):
            if not isinstance(column_side, exp.Column):
                continue
            if (column_side.name or "").lower() != "user_id":
                continue
            if isinstance(literal_side, exp.Literal) and literal_side.this == user_id:
                return True
    return False


def _scope_is_filtered(select: exp.Select, user_id: str) -> bool:
    """True if this SELECT constrains user_id in its WHERE or a JOIN condition.

    Restricted to WHERE/JOIN-ON so that projecting the id
    (`SELECT user_id = '<id>' AS mine FROM ...`) does not count as a filter —
    that returns every row while mentioning the id.
    """
    where = select.args.get("where")
    if where is not None and _is_user_id_equality(where, user_id):
        return True
    for join in select.args.get("joins") or []:
        on_clause = join.args.get("on")
        if on_clause is not None and _is_user_id_equality(on_clause, user_id):
            return True
    return False


def _direct_tables(select: exp.Select, cte_names: frozenset[str]) -> set[str]:
    """Tables this SELECT reads in its own FROM/JOIN, ignoring nested scopes.

    A nested subquery or a CTE reference is skipped — that scope is validated
    on its own pass, so charging the outer query for it would reject the
    perfectly safe `SELECT * FROM (SELECT ... WHERE user_id='<id>') s`.
    """
    sources = []
    from_clause = select.args.get("from_") or select.args.get("from")
    if from_clause is not None:
        sources.append(from_clause.this)
    for join in select.args.get("joins") or []:
        sources.append(join.this)

    tables = set()
    for source in sources:
        if isinstance(source, exp.Table):
            name = (source.name or "").lower()
            if name and name not in cte_names:
                tables.add(name)
    return tables


def _unfiltered_scopes(statement: exp.Expression, user_id: str) -> list[str]:
    """Names of tables read by a SELECT scope that never constrains user_id.

    Checked per scope rather than once for the whole tree, so neither a sibling
    CTE nor a UNION branch can read a table unfiltered while some other part of
    the query carries the filter — e.g.
    `WITH a AS (SELECT 1 FROM t WHERE user_id='<id>'), b AS (SELECT * FROM t) SELECT * FROM b`.
    """
    cte_names = frozenset(
        cte.alias_or_name.lower()
        for cte in statement.find_all(exp.CTE)
        if cte.alias_or_name
    )
    offenders = []
    for select in statement.find_all(exp.Select):
        tables = _direct_tables(select, cte_names)
        if tables and not _scope_is_filtered(select, user_id):
            offenders.extend(sorted(tables))
    return offenders


def validate_select(sql: str, user_id: str) -> str:
    """Return `sql` unchanged if it is safe to run, else raise SqlGuardError.

    `user_id` is the authenticated user injected by the server — never anything
    the model supplied.
    """
    if not user_id:
        raise SqlGuardError("No authenticated user; refusing to run the query.")
    if not sql or not sql.strip():
        raise SqlGuardError("Empty query.")

    try:
        statements = [s for s in sqlglot.parse(sql, dialect="postgres") if s is not None]
    except Exception as e:
        raise SqlGuardError(f"Could not parse the SQL: {e}") from e

    if len(statements) != 1:
        raise SqlGuardError(
            f"Send exactly one statement (found {len(statements)}). "
            "Multiple statements separated by ';' are not allowed."
        )

    statement = statements[0]
    # UNION / INTERSECT / EXCEPT parse to SetOperation rather than Select. They
    # are read-only and useful, and every branch is a Select that the per-scope
    # filter check below still walks.
    if not isinstance(statement, (exp.Select, exp.SetOperation)):
        raise SqlGuardError(
            f"Only SELECT queries are allowed (got {type(statement).__name__.upper()})."
        )

    for node_type in _FORBIDDEN_NODES:
        found = statement.find(node_type)
        if found is not None:
            raise SqlGuardError(
                f"Query contains a forbidden {type(found).__name__.upper()} operation. "
                "Only read-only SELECTs are allowed."
            )

    for func in statement.find_all(exp.Anonymous):
        if (func.name or "").lower() in _FORBIDDEN_FUNCTIONS:
            raise SqlGuardError(f"The function {func.name}() is not allowed.")

    tables = _statement_tables(statement)
    if not tables:
        raise SqlGuardError("Query must read from at least one of your data tables.")
    disallowed = tables - ALLOWED_TABLES
    if disallowed:
        raise SqlGuardError(
            f"These tables are not readable: {', '.join(sorted(disallowed))}. "
            f"Allowed tables: {', '.join(sorted(ALLOWED_TABLES))}."
        )

    offenders = _unfiltered_scopes(statement, user_id)
    if offenders:
        raise SqlGuardError(
            f"Every SELECT that reads a table must filter it: add "
            f"WHERE user_id = '{user_id}'. Missing on: {', '.join(sorted(set(offenders)))}."
        )

    return sql


def readonly_cursor(conn):
    """Put `conn` in read-only mode with a statement timeout, return a cursor.

    This is the layer that does not depend on parsing being perfect: Postgres
    itself rejects writes on a read-only session, so a statement that somehow
    got past `validate_select` still cannot mutate anything.
    """
    conn.set_session(readonly=True, autocommit=False)
    cursor = conn.cursor()
    cursor.execute("SET LOCAL statement_timeout = '15s'")
    return cursor
