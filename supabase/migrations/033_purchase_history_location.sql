-- Where a past purchase happened, returned alongside its price.
--
-- The column has always been on "Transaction"; the match function simply never
-- selected it, so every price comparison was blind to geography. The cost is
-- concrete: a gallon of milk bought in Huntsville, Alabama in March 2025 was
-- being held up as the going rate for a shelf in Norwalk, Connecticut in August
-- 2026 — a different state, a different cost of living, and a year and a half
-- apart. The numbers were right and the conclusion was not.
--
-- Recency was already weighted. Distance was not even visible.

DROP FUNCTION IF EXISTS public.match_purchase_history(text, text, integer, real);

CREATE OR REPLACE FUNCTION public.match_purchase_history(
    p_query_embedding text,
    p_embedding_model text DEFAULT NULL::text,
    p_limit integer DEFAULT 20,
    p_min_semantic real DEFAULT 0.75
)
RETURNS TABLE(
    id uuid, transaction_id uuid, item_description text,
    item_quantity numeric, item_quantity_unit text,
    unit_quantity_subtotal numeric, item_savings numeric, item_total_price numeric,
    enriched_info text, merchant_name text, location text, trans_date date,
    discount_total numeric, score real
)
LANGUAGE sql
STABLE
SET search_path TO 'public', 'extensions'
AS $function$
    SELECT d.id, d.transaction_id, d.item_description,
           d.item_quantity, d.item_quantity_unit,
           d.unit_quantity_subtotal, d.item_savings, d.item_total_price,
           d.enriched_info,
           t.merchant_name, t.location, t.trans_date, t.discount_total,
           (1 - (d.embedding <=> p_query_embedding::extensions.vector))::real AS score
    FROM public."TransactionDetail" d
    JOIN public."Transaction" t ON t.id = d.transaction_id
    WHERE d.embedding IS NOT NULL
      AND (p_embedding_model IS NULL OR d.embedding_model IS NOT DISTINCT FROM p_embedding_model)
      AND (1 - (d.embedding <=> p_query_embedding::extensions.vector)) >= p_min_semantic
    ORDER BY score DESC, t.trans_date DESC
    LIMIT p_limit;
$function$;
