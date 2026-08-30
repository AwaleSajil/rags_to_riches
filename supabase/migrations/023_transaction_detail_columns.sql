-- TransactionDetail: clearer column names, a recorded quantity unit, and
-- semantic-only matching.
--
-- ── 1. normalized_name goes ──────────────────────────────────────────────────
--
-- Matching becomes cosine-only. The trigram index goes with the column, and
-- both match functions lose their lexical axis.
--
-- Measured consequence, so it is not a surprise later: on labelled shelf-tag →
-- receipt-line pairs, the semantic axis separates cleanly ONLY because
-- enriched_info is now part of the embedded text — SAME 0.776-0.925 vs DIFF
-- 0.535-0.733, floor 0.75. Without enrichment the same pairs overlap by 0.208
-- and no floor works, missing 4 of 9 true matches. Enrichment is therefore a
-- correctness dependency, and only ~56% of rows currently have one.
DROP INDEX IF EXISTS idx_txdetail_normalized_trgm;
DROP INDEX IF EXISTS idx_txdetail_normalized;

ALTER TABLE public."TransactionDetail"
    DROP COLUMN IF EXISTS normalized_name;

-- ── 2. Name the two prices apart ─────────────────────────────────────────────
--
-- The table carried two columns that both read as "unit price" and meant
-- different things. Renaming the receipt-facing one leaves `unit_price` alone
-- but no longer adjacent to a same-sounding neighbour.
--
--   unit_quantity_subtotal — net, pre-tax, per ONE item_quantity unit. This is
--       the number printed on the receipt ($0.50/lb for bananas). Markdowns are
--       already applied; item_savings is informational and must never be
--       subtracted again.
--
--   unit_price — price per BASE unit (gram / mL / item), for comparing
--       differently-sized packages. NOT per size_unit: BB GRND TRKY1LB stores
--       size_unit='lb' and unit_price=0.006592, which is $2.99 / 453.59 g.
--       Anything reading it as per-pound is wrong by a factor of 453; see
--       units.format_unit_price for the display conversion.
ALTER TABLE public."TransactionDetail"
    RENAME COLUMN item_unit_subtotal_price TO unit_quantity_subtotal;

-- ── 3. Record what item_quantity is measured in ──────────────────────────────
--
-- item_quantity was dimensionless: "BANANAS qty 2.25" means 2.25 POUNDS, and
-- nothing said so. That is why loose produce had unit_price NULL despite the
-- receipt plainly showing $0.50/lb — the price existed, the unit was lost.
--
-- Holds 'lb', 'oz', 'ml', 'ct', 'each', ... Distinct from size_value/size_unit,
-- which describe how much is inside ONE purchase unit:
--
--   BANANAS        qty 2.25  unit 'lb'    size NULL     -> 2.25 lb loose
--   CILANTRO 30PK  qty 1.0   unit 'each'  size 30 ct    -> one pack of 30
--
-- Nullable: the vision prompt only starts extracting it now, and inferring it
-- for old rows is guesswork that would feed straight into price comparisons.
ALTER TABLE public."TransactionDetail"
    ADD COLUMN IF NOT EXISTS item_quantity_unit text;

-- ── 4. Both match functions become semantic-only ─────────────────────────────
--
-- p_query_text and p_min_lexical are gone with the trigram index. The floor of
-- 0.75 is measured (scripts/calibrate_price_thresholds.py) in the direction
-- matching actually runs: shelf-tag phrasing against a stored receipt line.
-- Row-to-row calibration gave 0.94 and would reject genuine matches here,
-- because the two sides are written by different authors for different reasons.
DROP FUNCTION IF EXISTS public.match_purchase_history(text, text, text, integer, real, real);
DROP FUNCTION IF EXISTS public.match_price_observations(text, text, text, integer, real, real);

CREATE OR REPLACE FUNCTION public.match_purchase_history(
    p_query_embedding   text,
    p_embedding_model   text    DEFAULT NULL,
    p_limit             integer DEFAULT 20,
    p_min_semantic      real    DEFAULT 0.75
)
RETURNS TABLE (
    id uuid, transaction_id uuid, item_description text,
    item_quantity numeric, item_quantity_unit text,
    unit_quantity_subtotal numeric, item_savings numeric,
    size_value numeric, size_unit text, unit_price numeric,
    merchant_name text, trans_date date, discount_total numeric,
    score real
)
LANGUAGE sql STABLE SET search_path = public, extensions
AS $$
    SELECT d.id, d.transaction_id, d.item_description,
           d.item_quantity, d.item_quantity_unit,
           d.unit_quantity_subtotal, d.item_savings,
           d.size_value, d.size_unit, d.unit_price,
           t.merchant_name, t.trans_date, t.discount_total,
           (1 - (d.embedding <=> p_query_embedding::extensions.vector))::real AS score
    FROM public."TransactionDetail" d
    JOIN public."Transaction" t ON t.id = d.transaction_id
    WHERE d.embedding IS NOT NULL
      AND (p_embedding_model IS NULL OR d.embedding_model IS NOT DISTINCT FROM p_embedding_model)
      AND (1 - (d.embedding <=> p_query_embedding::extensions.vector)) >= p_min_semantic
    -- Recency does NOT filter. An old purchase is still the honest answer when
    -- it is the only one; age is applied as a weight when the baseline is
    -- computed (price_service.price_weight), so it never hides evidence.
    ORDER BY score DESC, t.trans_date DESC
    LIMIT p_limit;
$$;

CREATE OR REPLACE FUNCTION public.match_price_observations(
    p_query_embedding   text,
    p_embedding_model   text    DEFAULT NULL,
    p_limit             integer DEFAULT 10,
    p_min_semantic      real    DEFAULT 0.75
)
RETURNS TABLE (
    id uuid, item_description text, brand text,
    size_text text, size_value numeric, size_unit text,
    observed_price numeric, unit_price numeric, currency text,
    merchant_name text, observed_at timestamptz,
    is_promotional boolean, promo_text text, promo_ends_on date,
    expires_on date, note text, score real
)
LANGUAGE sql STABLE SET search_path = public, extensions
AS $$
    SELECT o.id, o.item_description, o.brand,
           o.size_text, o.size_value, o.size_unit,
           o.observed_price, o.unit_price, o.currency,
           o.merchant_name, o.observed_at,
           o.is_promotional, o.promo_text, o.promo_ends_on,
           o.expires_on, o.note,
           (1 - (o.embedding <=> p_query_embedding::extensions.vector))::real AS score
    FROM public."PriceObservation" o
    WHERE o.embedding IS NOT NULL
      AND (p_embedding_model IS NULL OR o.embedding_model IS NOT DISTINCT FROM p_embedding_model)
      AND (1 - (o.embedding <=> p_query_embedding::extensions.vector)) >= p_min_semantic
    ORDER BY score DESC, o.observed_at DESC
    LIMIT p_limit;
$$;

REVOKE ALL ON FUNCTION public.match_purchase_history(text, text, integer, real) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.match_purchase_history(text, text, integer, real) TO authenticated;
REVOKE ALL ON FUNCTION public.match_price_observations(text, text, integer, real) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.match_price_observations(text, text, integer, real) TO authenticated;
