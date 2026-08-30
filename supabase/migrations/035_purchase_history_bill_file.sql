-- Return WHICH receipt a matched purchase came from.
--
-- Asked to show proof of the prices it had just quoted, the agent had no id for
-- them, so it wrote SQL to find the receipts again — searching item text for
-- "milk". The only line literally containing MILK is Costco's "3 WHOLE MILK";
-- the Walmart rows it had quoted are "GV LF 2 GAL" and "GV RF 2 GAL". It
-- attached two Costco receipts as evidence for two Walmart prices.
--
-- The prices were right and the proof was somebody else's. Carrying the
-- bill_file_id through means the receipt is fetched by identity rather than
-- re-derived from a text search that cannot match an abbreviation.
DROP FUNCTION IF EXISTS public.match_purchase_history(text, text, integer, real);

CREATE FUNCTION public.match_purchase_history(
    p_query_embedding text, p_embedding_model text DEFAULT NULL::text,
    p_limit integer DEFAULT 20, p_min_semantic real DEFAULT 0.75
)
RETURNS TABLE(
    id uuid, transaction_id uuid, bill_file_id uuid, item_description text,
    item_quantity numeric, item_quantity_unit text,
    size_value numeric, size_unit text,
    unit_quantity_subtotal numeric, item_savings numeric, item_total_price numeric,
    enriched_info text, merchant_name text, location text, trans_date date,
    discount_total numeric, score real
)
LANGUAGE sql STABLE SET search_path TO 'public', 'extensions'
AS $function$
    SELECT d.id, d.transaction_id, d.bill_file_id, d.item_description,
           d.item_quantity, d.item_quantity_unit,
           d.size_value, d.size_unit,
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
