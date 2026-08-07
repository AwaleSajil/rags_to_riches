-- Slim the table down, then give it the search it actually needs.
--
-- Part 1 removes four columns nothing ever wrote. Part 2 adds the two that make
-- "have I seen this item before?" answerable when the words don't line up.
-- Net effect is fewer columns than before and a capability that did not exist.

-- ── Part 1: drop what was never written ─────────────────────────────────────
--
-- Verified against every insert path before dropping; the table is empty, so
-- there is nothing to preserve either way.
--
--   location    — superseded by place_label, never in any insert
--   raw_ocr     — duplicated BillFile.raw_ocr_string, reachable via
--                 source_bill_file_id; never populated
--   verified    — hardcoded true at the only writer, so it carried no signal
--   created_at  — always equal to observed_at, which nothing sets explicitly
--
-- `note` is deliberately kept: notes at capture time are a wanted feature.
ALTER TABLE public."PriceObservation"
    DROP COLUMN IF EXISTS location,
    DROP COLUMN IF EXISTS raw_ocr,
    DROP COLUMN IF EXISTS verified,
    DROP COLUMN IF EXISTS created_at;

-- ── Part 2: hybrid search ───────────────────────────────────────────────────
--
-- Trigram alone handles OCR mangling ("peanut buttr" → "peanut butter") because
-- that is a character problem. It cannot handle receipt abbreviations
-- ("gv ff gal" → "Great Value Fat Free Gallon"), which is a meaning problem —
-- measured at 0.16 similarity against real data here. Embeddings cover that
-- axis, so both are kept and neither is trusted alone.
--
-- Unbounded `vector` rather than vector(3072): embedding model is per-user
-- config, so pinning a dimension would break any account not on Gemini. The
-- cost is that no HNSW/IVFFlat index can exist — those need a fixed dimension —
-- so search is a sequential scan. Acceptable here and nowhere near the
-- transaction store's scale: one row per shelf photo, bounded by how often a
-- person photographs a price tag.
ALTER TABLE public."PriceObservation"
    ADD COLUMN IF NOT EXISTS embedding extensions.vector,
    -- Which model produced the vector above. Two models with the SAME dimension
    -- produce vectors that are not comparable, and cosine distance will happily
    -- return confident nonsense rather than an error. Mismatches are filtered
    -- out at query time instead.
    ADD COLUMN IF NOT EXISTS embedding_model text;

-- ── Ranking ─────────────────────────────────────────────────────────────────
--
-- SECURITY INVOKER (the default) on purpose: called with the user's JWT, so the
-- existing price_observation_own RLS policy does the tenant filtering. The
-- alternative — ranking in Python over an RLS-bypassing connection — would make
-- correct isolation depend on every caller remembering to filter.
--
-- p_query_embedding is text and cast here so callers can send a plain
-- '[0.1,0.2,...]' literal without depending on client-side vector encoding.
CREATE OR REPLACE FUNCTION public.match_price_observations(
    p_query_text        text,
    p_query_embedding   text    DEFAULT NULL,
    p_embedding_model   text    DEFAULT NULL,
    p_limit             integer DEFAULT 10,
    -- Floors, not a blend. A row must be a strong match on at least one axis to
    -- appear at all: without them the nearest row is always returned however
    -- unrelated, and "is this a good price?" answers from a different product.
    --
    -- MEASURED, not chosen — see scripts/calibrate_price_thresholds.py. Neither
    -- axis separates products on its own: same-product cosine spans 0.601–0.965
    -- and different-product 0.568–0.821, so they overlap by 0.22. These two
    -- floors together are the widest-margin rule that admits no wrong product
    -- across the labelled pairs. The earlier guessed values (0.3 / 0.75) let in
    -- "ORG BANANAS" for bananas, "GRND CHKN" and "WHL CARROTS" for whole
    -- chicken. Re-run the script and re-derive these if the embedding model
    -- changes — they are specific to gemini-embedding-001.
    p_min_lexical       real    DEFAULT 0.47,
    p_min_semantic      real    DEFAULT 0.94
)
RETURNS TABLE (
    id               uuid,
    item_description text,
    normalized_name  text,
    brand            text,
    size_text        text,
    size_value       numeric,
    size_unit        text,
    observed_price   numeric,
    unit_price       numeric,
    currency         text,
    merchant_name    text,
    observed_at      timestamptz,
    is_promotional   boolean,
    promo_text       text,
    promo_ends_on    date,
    expires_on       date,
    note             text,
    lexical_score    real,
    semantic_score   real,
    score            real
)
LANGUAGE sql
STABLE
-- Pinned because the `<=>` cosine operator is resolved through search_path even
-- though the tables and functions here are schema-qualified. Without this, a
-- schema earlier in a caller's path could supply a different operator and
-- quietly change what "nearest" means.
SET search_path = public, extensions
AS $$
    WITH scored AS (
        SELECT
            o.*,
            extensions.similarity(o.normalized_name, p_query_text) AS lex,
            CASE
                -- No query vector, no stored vector, or a vector from a
                -- different model: the semantic axis simply does not apply
                -- rather than contributing a made-up number.
                WHEN p_query_embedding IS NULL THEN NULL
                WHEN o.embedding IS NULL THEN NULL
                WHEN p_embedding_model IS NOT NULL
                     AND o.embedding_model IS DISTINCT FROM p_embedding_model THEN NULL
                ELSE 1 - (o.embedding <=> p_query_embedding::extensions.vector)
            END AS sem
        FROM public."PriceObservation" o
        WHERE o.normalized_name IS NOT NULL
    )
    SELECT
        s.id, s.item_description, s.normalized_name, s.brand,
        s.size_text, s.size_value, s.size_unit,
        s.observed_price, s.unit_price, s.currency,
        s.merchant_name, s.observed_at,
        s.is_promotional, s.promo_text, s.promo_ends_on, s.expires_on, s.note,
        s.lex::real  AS lexical_score,
        s.sem::real  AS semantic_score,
        -- Best axis wins rather than an average: a verbatim name match and a
        -- confident semantic match are each sufficient on their own, and
        -- averaging would let a row that is mediocre on both outrank them.
        GREATEST(COALESCE(s.lex, 0), COALESCE(s.sem, 0))::real AS score
    FROM scored s
    WHERE s.lex >= p_min_lexical
       OR s.sem >= p_min_semantic
    ORDER BY score DESC, s.observed_at DESC
    LIMIT p_limit;
$$;

-- Callable by signed-in users only; RLS inside still scopes rows to the caller.
REVOKE ALL ON FUNCTION public.match_price_observations(
    text, text, text, integer, real, real
) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.match_price_observations(
    text, text, text, integer, real, real
) TO authenticated;
