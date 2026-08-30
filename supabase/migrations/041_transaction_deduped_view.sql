-- ============================================================
-- One row per real-world purchase, for anything that adds money up.
--
-- A photographed receipt and the bank-statement line for the same purchase are
-- BOTH kept: the statement is what the bank says, the receipt is what was
-- actually bought, and neither is safe to throw away. They are reconciled by a
-- row in "TransactionLink" rather than merged.
--
-- That leaves every consumer responsible for not counting both, and only one of
-- them was doing it. The transactions list collapses each linked group before
-- summing. The CHAT AGENT writes its own SQL, and the only thing between it and
-- a doubled total was one sentence inside the schema description it is handed —
-- so "how much did I spend on groceries", the most common question asked of this
-- app, was answered wrongly by writing the most obvious possible query.
--
-- A view, so that the correct thing is also the easy thing. The agent selects
-- from here and cannot double count, instead of being asked to remember a join.
--
-- security_invoker (Postgres 15+), so the view runs with the rights of the
-- CALLER and RLS on "Transaction" still applies. Without it, a view owned by
-- postgres would hand every user's rows to anyone who selected from it.
-- The agent's own connection is a superuser that bypasses RLS regardless — it is
-- constrained separately, by sql_guard requiring a real user_id equality — but
-- the app queries as the user, and this view has to be safe on that path too.
--
-- embedding/embedding_model are deliberately absent. They are large, never
-- meaningful to report, and the schema doc already tells the agent never to
-- select them; leaving them out means a stray SELECT * cannot drag a few
-- thousand floats per row into the model's context.
-- ============================================================

CREATE OR REPLACE VIEW public."TransactionDeduped"
WITH (security_invoker = true) AS
SELECT
    t.id, t.user_id, t.trans_date, t.description, t.merchant_name, t.amount,
    t.category, t.location, t.subtotal, t.tax_total, t.tax_breakdown,
    t.discount_total, t.savings_total, t.note, t.enriched_info,
    t.source, t.source_csv_id, t.source_bill_file_id, t.content_hash,
    t.created_at
FROM public."Transaction" t
WHERE NOT EXISTS (
    -- A linked partner that OUTRANKS this row. Rank is: a receipt beats a bank
    -- or manual row, because it carries the line items and the tax breakdown and
    -- is already what the mobile list shows for a linked pair. Between two rows
    -- of equal standing the lower id wins, so the survivor is stable across
    -- queries rather than dependent on plan order.
    --
    -- Deliberately NOT a transitive closure over the link graph. Links arrive as
    -- pairs and as stars (one receipt reconciled against several statement
    -- rows), both of which this handles exactly, and a recursive CTE over every
    -- transaction would cost far more than the chain case it would protect
    -- against — a chain that one-to-one claiming in pair_same_purchase is
    -- already designed to prevent.
    SELECT 1
    FROM public."TransactionLink" l
    JOIN public."Transaction" partner
      ON partner.id = CASE
             WHEN l.transaction_id = t.id THEN l.linked_transaction_id
             ELSE l.transaction_id
         END
    WHERE l.user_id = t.user_id
      AND (l.transaction_id = t.id OR l.linked_transaction_id = t.id)
      AND (
              (COALESCE(partner.source, '') = 'bill' AND COALESCE(t.source, '') <> 'bill')
           OR (
                  (COALESCE(partner.source, '') = 'bill')
                      = (COALESCE(t.source, '') = 'bill')
                  AND partner.id < t.id
              )
          )
);

COMMENT ON VIEW public."TransactionDeduped" IS
    'Transaction with linked duplicates removed — one row per real-world purchase. Use this for any total.';

GRANT SELECT ON public."TransactionDeduped" TO authenticated;
