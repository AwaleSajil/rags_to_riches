-- Make PriceObservation look like TransactionDetail: 25 columns -> 18.
--
-- ── Derived columns go (same rule applied to TransactionDetail in 023/024) ───
--
--   normalized_name, size_value, size_unit, unit_price
--
-- All four were parse_size / normalize_item_name output over item_description
-- and size_text. No independent input, so storing them only froze the parser's
-- mistakes: measured on real receipt rows, ~26% of parsed sizes were wrong
-- ("5L" on a bag of potatoes read as litres, a product code read as a 30-count
-- pack). Parsing on demand means a parser fix heals history instead.
--
-- normalized_name additionally has nothing left to do: matching is semantic
-- only since 023, so there is no trigram index for it to feed.
--
-- ── Promo/expiry structure folds into one text column ───────────────────────
--
--   is_promotional, promo_text, promo_ends_on, expires_on  ->  observed_context
--
-- These asked a vision model to turn a shelf sign into booleans and ISO dates.
-- That is guesswork with an asymmetric cost: a wrong promo_ends_on silently
-- turns a limited offer into what the item normally costs, and there is no
-- error to notice. observed_context keeps the tag's own words — "2 for $5 with
-- card", "Sale ends 8/15", "best before 08/05", "CLEARANCE", "dented box" —
-- and the agent reasons from them alongside the price.
--
-- This follows the architecture the rest of the feature now uses: the database
-- retrieves candidates by vector search, and comparison is the agent's job with
-- all the information in hand. price_service.price_caveats and
-- is_baseline_quality are removed accordingly; the purchase-side helpers stay,
-- because receipt savings semantics (item_savings is already netted out) is the
-- one place an LLM reliably reaches for the wrong arithmetic.
--
-- observed_context also does for a shelf tag what enriched_info does for a
-- receipt line: it joins the embedded text, which is what makes a terse tag
-- matchable at all.
ALTER TABLE public."PriceObservation"
    DROP COLUMN IF EXISTS normalized_name,
    DROP COLUMN IF EXISTS size_value,
    DROP COLUMN IF EXISTS size_unit,
    DROP COLUMN IF EXISTS unit_price,
    DROP COLUMN IF EXISTS is_promotional,
    DROP COLUMN IF EXISTS promo_text,
    DROP COLUMN IF EXISTS promo_ends_on,
    DROP COLUMN IF EXISTS expires_on;

ALTER TABLE public."PriceObservation"
    ADD COLUMN IF NOT EXISTS observed_context text;

-- Both indexed normalized_name, which no longer exists.
DROP INDEX IF EXISTS idx_priceobs_user_item;
DROP INDEX IF EXISTS idx_priceobs_normalized_trgm;

DROP FUNCTION IF EXISTS public.match_price_observations(text, text, integer, real);

CREATE OR REPLACE FUNCTION public.match_price_observations(
    p_query_embedding   text,
    p_embedding_model   text    DEFAULT NULL,
    p_limit             integer DEFAULT 10,
    p_min_semantic      real    DEFAULT 0.75
)
RETURNS TABLE (
    id uuid, item_description text, size_text text,
    observed_price numeric, currency text, merchant_name text,
    observed_at timestamptz, category text,
    observed_context text, note text, score real
)
LANGUAGE sql STABLE SET search_path = public, extensions
AS $$
    SELECT o.id, o.item_description, o.size_text,
           o.observed_price, o.currency, o.merchant_name, o.observed_at,
           o.category, o.observed_context, o.note,
           (1 - (o.embedding <=> p_query_embedding::extensions.vector))::real AS score
    FROM public."PriceObservation" o
    WHERE o.embedding IS NOT NULL
      AND (p_embedding_model IS NULL OR o.embedding_model IS NOT DISTINCT FROM p_embedding_model)
      AND (1 - (o.embedding <=> p_query_embedding::extensions.vector)) >= p_min_semantic
    ORDER BY score DESC, o.observed_at DESC
    LIMIT p_limit;
$$;

REVOKE ALL ON FUNCTION public.match_price_observations(text, text, integer, real) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.match_price_observations(text, text, integer, real) TO authenticated;
