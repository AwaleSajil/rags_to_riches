-- Rebuild TransactionDetail: drop the derived size columns, and put
-- item_quantity_unit next to the quantity it describes.
--
-- ── Why a rebuild rather than ALTER ─────────────────────────────────────────
--
-- Postgres cannot move a column. ADD COLUMN always appends and there is no
-- ALTER TABLE ... SET ORDER, so the only way to place item_quantity_unit after
-- item_quantity is to create the table afresh and copy. Safe to do here because
-- nothing has a foreign key pointing AT this table (0 inbound FKs); its own 3
-- outbound FKs, 2 indexes and RLS policy are recreated below.
--
-- ── Why size_value / size_unit / unit_price go ──────────────────────────────
--
-- All three were derived from item_description by parse_size — no independent
-- input, exactly like normalized_name before them. "WW SPAG 16OZ" already
-- carries its size in the text.
--
-- Storing the parse turned out to be worse than not storing it. Audited against
-- the 27 populated rows, roughly a quarter were wrong, and wrong in ways that
-- look authoritative:
--
--   +RED POTA 5L US#   -> 5 LITRES of potatoes (a 5 lb bag; wrong unit family)
--   MS 13.2G STP       -> 13.2 g at $34.46  =  $2.61 per gram
--   CILANTRO 30PK      -> 30 count for $0.99 (a bunch; "30PK" is a code)
--   SS COLANDER 3QT    -> 3 qt of colander (that is its capacity, not contents)
--   GV LF 2 GAL        -> 2 gallons (almost certainly 2% fat, ONE gallon)
--
-- Caching is only a virtue when the cached value is right. Parsing at read time
-- means a parse_size fix heals every historical row; storing it freezes the
-- mistakes permanently.
--
-- It also no longer earns its place architecturally: retrieval is semantic
-- (match_purchase_history) and the comparison is the agent's job, so the agent
-- reads the size out of item_description along with everything else rather than
-- trusting a pre-parsed column sitting next to text that contradicts it.
--
-- unit_price additionally carried a factor-of-453 trap: it stored price per
-- BASE unit (per gram) while size_unit said 'lb'.

CREATE TABLE public."TransactionDetail_new" (
    id                      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id          uuid REFERENCES public."Transaction"(id) ON DELETE CASCADE,
    user_id                 uuid NOT NULL REFERENCES public."User"(id) ON DELETE CASCADE,
    bill_file_id            uuid REFERENCES public."BillFile"(id) ON DELETE CASCADE,

    item_description        text,
    item_quantity           numeric,
    item_quantity_unit      text,

    unit_quantity_subtotal  numeric,
    item_subtotal_price     numeric,
    item_savings            numeric,
    tax_rate                numeric,
    tax_amount              numeric,
    taxable                 boolean,
    item_total_price        numeric,

    enriched_info           text,
    embedding               extensions.vector,
    embedding_model         text,
    created_at              timestamptz NOT NULL DEFAULT now()
);

INSERT INTO public."TransactionDetail_new" (
    id, transaction_id, user_id, bill_file_id,
    item_description, item_quantity, item_quantity_unit,
    unit_quantity_subtotal, item_subtotal_price, item_savings,
    tax_rate, tax_amount, taxable, item_total_price,
    enriched_info, embedding, embedding_model, created_at
)
SELECT id, transaction_id, user_id, bill_file_id,
       item_description, item_quantity, item_quantity_unit,
       unit_quantity_subtotal, item_subtotal_price, item_savings,
       tax_rate, tax_amount, taxable, item_total_price,
       enriched_info, embedding, embedding_model, created_at
FROM public."TransactionDetail";

DROP TABLE public."TransactionDetail";
ALTER TABLE public."TransactionDetail_new" RENAME TO "TransactionDetail";
ALTER TABLE public."TransactionDetail" RENAME CONSTRAINT "TransactionDetail_new_pkey" TO "TransactionDetail_pkey";

CREATE INDEX idx_txdetail_user ON public."TransactionDetail" USING btree (user_id);
CREATE INDEX idx_txdetail_tx   ON public."TransactionDetail" USING btree (transaction_id);

ALTER TABLE public."TransactionDetail" ENABLE ROW LEVEL SECURITY;
CREATE POLICY txdetail_own ON public."TransactionDetail"
    FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

-- match_purchase_history selected the dropped columns, so it is rebuilt too.
-- enriched_info is now returned: the agent needs the same abbreviation decoding
-- the embedding uses ("GV LF 2 GAL" -> "a two-gallon container of Great Value
-- low-fat milk") to reason about what a row actually is.
DROP FUNCTION IF EXISTS public.match_purchase_history(text, text, integer, real);

CREATE OR REPLACE FUNCTION public.match_purchase_history(
    p_query_embedding   text,
    p_embedding_model   text    DEFAULT NULL,
    p_limit             integer DEFAULT 20,
    p_min_semantic      real    DEFAULT 0.75
)
RETURNS TABLE (
    id uuid, transaction_id uuid, item_description text,
    item_quantity numeric, item_quantity_unit text,
    unit_quantity_subtotal numeric, item_savings numeric, item_total_price numeric,
    enriched_info text, merchant_name text, trans_date date,
    discount_total numeric, score real
)
LANGUAGE sql STABLE SET search_path = public, extensions
AS $$
    SELECT d.id, d.transaction_id, d.item_description,
           d.item_quantity, d.item_quantity_unit,
           d.unit_quantity_subtotal, d.item_savings, d.item_total_price,
           d.enriched_info,
           t.merchant_name, t.trans_date, t.discount_total,
           (1 - (d.embedding <=> p_query_embedding::extensions.vector))::real AS score
    FROM public."TransactionDetail" d
    JOIN public."Transaction" t ON t.id = d.transaction_id
    WHERE d.embedding IS NOT NULL
      AND (p_embedding_model IS NULL OR d.embedding_model IS NOT DISTINCT FROM p_embedding_model)
      AND (1 - (d.embedding <=> p_query_embedding::extensions.vector)) >= p_min_semantic
    ORDER BY score DESC, t.trans_date DESC
    LIMIT p_limit;
$$;

REVOKE ALL ON FUNCTION public.match_purchase_history(text, text, integer, real) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.match_purchase_history(text, text, integer, real) TO authenticated;

-- A copy of the pre-rebuild table is left behind as TransactionDetail_backup_024.
-- Drop it once you are satisfied: DROP TABLE public."TransactionDetail_backup_024";
