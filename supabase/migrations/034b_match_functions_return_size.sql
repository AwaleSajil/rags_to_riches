-- Teach the price-observation search the names migration 034 gave the columns.
--
-- RECONSTRUCTED, the same way 036 was. This migration is recorded as applied
-- ("034b_match_functions_return_size") and the live function matches what is
-- below, but the file was never committed — so the repository could not rebuild
-- the database it describes. What follows is the live definition, verbatim.
--
-- Why it has to exist at all: 034 RENAMED "PriceObservation".item_quantity to
-- size_value and item_quantity_unit to size_unit. A LANGUAGE sql function whose
-- body is a quoted string is re-parsed on every call and is NOT rewritten by a
-- column rename, so the definition left behind by 029 went on selecting
-- o.item_quantity and would raise "column o.item_quantity does not exist" on
-- every call. Nothing surfaces that: compare_price catches a failed search and
-- reports "no history to compare against", so a shelf price the user had
-- already photographed would simply never appear as evidence — indistinguishable
-- from never having seen it.
--
-- match_purchase_history needed the same treatment and got it here too; the
-- committed 035 has since redefined that one with bill_file_id, so only the
-- price-observation half is restored below.
DROP FUNCTION IF EXISTS public.match_price_observations(text, text, integer, real);

CREATE FUNCTION public.match_price_observations(
    p_query_embedding text,
    p_embedding_model text    DEFAULT NULL::text,
    p_limit           integer DEFAULT 20,
    p_min_semantic    real    DEFAULT 0.75
)
RETURNS TABLE(
    id uuid, bill_file_id uuid, merchant_name text, location text,
    item_description text, size_value numeric, size_unit text,
    unit_quantity_subtotal numeric, item_subtotal_price numeric,
    item_qualitative_description text, brand_name text, enriched_info text,
    note text, created_at timestamp with time zone, score real
)
LANGUAGE sql STABLE SET search_path TO 'public', 'extensions'
AS $function$
    SELECT p.id, p.bill_file_id, p.merchant_name, p.location,
           p.item_description, p.size_value, p.size_unit,
           p.unit_quantity_subtotal, p.item_subtotal_price,
           p.item_qualitative_description, p.brand_name, p.enriched_info,
           p.note, p.created_at,
           (1 - (p.embedding <=> p_query_embedding::extensions.vector))::real AS score
    FROM public."PriceObservation" p
    WHERE p.embedding IS NOT NULL
      AND (p_embedding_model IS NULL OR p.embedding_model IS NOT DISTINCT FROM p_embedding_model)
      AND (1 - (p.embedding <=> p_query_embedding::extensions.vector)) >= p_min_semantic
    ORDER BY score DESC, p.created_at DESC
    LIMIT p_limit;
$function$;

REVOKE ALL ON FUNCTION public.match_price_observations(text, text, integer, real) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.match_price_observations(text, text, integer, real) TO authenticated;
