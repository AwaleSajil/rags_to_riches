-- Make purchase history matchable the same way shelf prices are.
--
-- "What have I paid for this?" has to compare two records that describe the same
-- product completely differently — a receipt line ("GV LF 2 GAL") and a shelf tag
-- ("Great Value Lactose Free Milk 2 Gallon"). PriceObservation already carries
-- what that comparison needs: a parsed size, a unit price, and a product-identity
-- vector. TransactionDetail carried none of it, so the two sides could not be
-- compared at all.
--
-- Why not reuse the line-item vectors already in langchain_pg_embedding: their
-- text is "Line item from Walmart: STRAWBERRIES — None", so merchant boilerplate
-- dominates the embedding. Probed with a real BANANAS vector, STRAWBERRIES scores
-- 0.937 and GREEN BEANS 0.927 — every piece of produce collapses into one
-- cluster. Those vectors do their own job (general chat retrieval) well; they
-- cannot tell two products apart. That table also has RLS enabled with no
-- policies and 49 known orphaned rows, both of which this feature would inherit.
ALTER TABLE public."TransactionDetail"
    -- Parsed from item_description where one is printed. Only about 12% of real
    -- lines carry a size — the rest are loose produce ("LIMES", "GROUND PORK") —
    -- so callers must fall back to item_quantity and say the comparison is
    -- per-item rather than per-ounce.
    ADD COLUMN IF NOT EXISTS size_value  numeric,
    ADD COLUMN IF NOT EXISTS size_unit   text,
    -- Price per size_unit, stored rather than derived so a later change to the
    -- parsing rules cannot silently restate what history cost.
    ADD COLUMN IF NOT EXISTS unit_price  numeric,
    -- Unbounded `vector`, matching PriceObservation: the embedding model is
    -- per-user config, so pinning a dimension would break any account not on
    -- Gemini. Cost is that no HNSW/IVFFlat index is possible (they need a fixed
    -- dimension) — irrelevant at a few hundred rows per user.
    ADD COLUMN IF NOT EXISTS embedding   extensions.vector,
    -- Two models of the same dimension produce vectors that are not comparable,
    -- and cosine distance returns confident nonsense rather than erroring.
    -- Mismatches are excluded at query time instead.
    ADD COLUMN IF NOT EXISTS embedding_model text,
    ADD CONSTRAINT txdetail_size_value_positive
        CHECK (size_value IS NULL OR size_value > 0) NOT VALID;

-- Mirrors match_price_observations so both sides of a comparison rank the same
-- way. SECURITY INVOKER (the default) on purpose: called with the user's JWT, so
-- the existing txdetail_own RLS policy does the tenant filtering. There is
-- deliberately no user_id argument — one that could be passed wrongly.
CREATE OR REPLACE FUNCTION public.match_purchase_history(
    p_query_text        text,
    p_query_embedding   text    DEFAULT NULL,
    p_embedding_model   text    DEFAULT NULL,
    p_limit             integer DEFAULT 20,
    -- Same measured floors as match_price_observations — see the note there and
    -- scripts/calibrate_price_thresholds.py. Both sides of a comparison must use
    -- identical floors or "seen at $4.29" and "paid $3.79" would be drawn from
    -- differently-sized candidate pools.
    p_min_lexical       real    DEFAULT 0.47,
    p_min_semantic      real    DEFAULT 0.94
)
RETURNS TABLE (
    id                       uuid,
    transaction_id           uuid,
    item_description         text,
    normalized_name          text,
    item_quantity            numeric,
    item_unit_subtotal_price numeric,
    item_savings             numeric,
    size_value               numeric,
    size_unit                text,
    unit_price               numeric,
    merchant_name            text,
    trans_date               date,
    discount_total           numeric,
    lexical_score            real,
    semantic_score           real,
    score                    real
)
LANGUAGE sql
STABLE
-- Pinned because the `<=>` cosine operator resolves through search_path even
-- though everything else here is schema-qualified.
SET search_path = public, extensions
AS $$
    WITH scored AS (
        SELECT
            d.id, d.transaction_id, d.item_description, d.normalized_name,
            d.item_quantity, d.item_unit_subtotal_price, d.item_savings,
            d.size_value, d.size_unit, d.unit_price,
            t.merchant_name, t.trans_date, t.discount_total,
            extensions.similarity(d.normalized_name, p_query_text) AS lex,
            CASE
                -- No query vector, no stored vector, or a vector from a
                -- different model: the semantic axis does not apply rather than
                -- contributing a made-up number.
                WHEN p_query_embedding IS NULL THEN NULL
                WHEN d.embedding IS NULL THEN NULL
                WHEN p_embedding_model IS NOT NULL
                     AND d.embedding_model IS DISTINCT FROM p_embedding_model THEN NULL
                ELSE 1 - (d.embedding <=> p_query_embedding::extensions.vector)
            END AS sem
        FROM public."TransactionDetail" d
        JOIN public."Transaction" t ON t.id = d.transaction_id
        WHERE d.normalized_name IS NOT NULL
    )
    SELECT
        s.id, s.transaction_id, s.item_description, s.normalized_name,
        s.item_quantity, s.item_unit_subtotal_price, s.item_savings,
        s.size_value, s.size_unit, s.unit_price,
        s.merchant_name, s.trans_date, s.discount_total,
        s.lex::real AS lexical_score,
        s.sem::real AS semantic_score,
        -- Best axis wins rather than an average: a verbatim name match and a
        -- confident semantic match are each sufficient alone, and averaging
        -- would let a row mediocre on both outrank them.
        GREATEST(COALESCE(s.lex, 0), COALESCE(s.sem, 0))::real AS score
    FROM scored s
    WHERE s.lex >= p_min_lexical
       OR s.sem >= p_min_semantic
    -- Recency deliberately does NOT filter here. An old purchase is still the
    -- honest answer when it is the only one; recency is applied as a weight when
    -- the baseline is computed, so ancient evidence barely moves the number but
    -- is never hidden.
    ORDER BY score DESC, s.trans_date DESC
    LIMIT p_limit;
$$;

REVOKE ALL ON FUNCTION public.match_purchase_history(
    text, text, text, integer, real, real
) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.match_purchase_history(
    text, text, text, integer, real, real
) TO authenticated;
