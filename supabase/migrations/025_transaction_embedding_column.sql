-- Move transaction vectors onto the Transaction row, and add the search
-- function that replaces langchain's similarity_search.
--
-- Why on the row: deleting a transaction now takes its vector with it, so the
-- orphaned-vector class of bug becomes structurally impossible. The langchain
-- store accumulated 49 stale line-item vectors that still matched searches and
-- let the agent quote line items deleted during a receipt re-review. The
-- table's own RLS policy also applies, where the langchain tables have RLS
-- enabled with NO policies and tenancy rested on a cmetadata filter over an
-- RLS-bypassing connection.
ALTER TABLE public."Transaction"
    ADD COLUMN IF NOT EXISTS embedding       extensions.vector,
    ADD COLUMN IF NOT EXISTS embedding_model text;

-- Copy, not re-embed. These vectors came from the same text and the same model,
-- so regenerating them would spend 392 API calls to arrive at identical numbers.
-- Verified afterwards: 392/392 copied, zero differing from their source.
UPDATE public."Transaction" t
SET embedding = e.embedding,
    embedding_model = COALESCE(c.embedding_model, 'gemini-embedding-001')
FROM public.langchain_pg_embedding e
LEFT JOIN public."AccountConfig" c ON c.user_id::text = e.cmetadata->>'user_id'
WHERE e.id = t.id::text
  AND e.cmetadata->>'vector_type' = 'transaction'
  AND t.embedding IS NULL;

-- Chat retrieval. Deliberately defaults to NO floor (0.0), preserving the
-- behaviour callers already had: langchain's similarity_search returned k rows
-- regardless of distance. That is a real weakness — asking "shampoo" returns
-- the nearest three rows even when none is a shampoo — but changing it here
-- would silently alter chat behaviour, so it stays a separate decision.
-- Product matching does NOT share this: match_purchase_history enforces the
-- measured 0.75 floor.
CREATE OR REPLACE FUNCTION public.match_transactions(
    p_query_embedding   text,
    p_embedding_model   text    DEFAULT NULL,
    p_limit             integer DEFAULT 10,
    p_min_semantic      real    DEFAULT 0.0
)
RETURNS TABLE (
    id uuid, trans_date date, merchant_name text, description text,
    category text, amount numeric, note text, enriched_info text, score real
)
LANGUAGE sql STABLE SET search_path = public, extensions
AS $$
    SELECT t.id, t.trans_date, t.merchant_name, t.description,
           t.category, t.amount, t.note, t.enriched_info,
           (1 - (t.embedding <=> p_query_embedding::extensions.vector))::real AS score
    FROM public."Transaction" t
    WHERE t.embedding IS NOT NULL
      AND (p_embedding_model IS NULL OR t.embedding_model IS NOT DISTINCT FROM p_embedding_model)
      AND (1 - (t.embedding <=> p_query_embedding::extensions.vector)) >= p_min_semantic
    ORDER BY score DESC
    LIMIT p_limit;
$$;

REVOKE ALL ON FUNCTION public.match_transactions(text, text, integer, real) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.match_transactions(text, text, integer, real) TO authenticated;

-- langchain_pg_embedding / langchain_pg_collection are intentionally NOT dropped
-- here. Everything in them now lives on the rows, so they are redundant — but
-- leaving them in place keeps a fallback until the new path has been exercised.
