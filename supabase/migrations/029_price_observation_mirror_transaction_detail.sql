-- Rebuild PriceObservation to mirror TransactionDetail. 18 columns -> 17.
--
-- The two tables get compared against each other on every price check, so
-- holding the same facts in the same shape removes a translation layer that had
-- no reason to exist. A tag reading "$4.29 / 12 OZ" now lands exactly as a
-- receipt line would:
--
--   item_quantity 12, item_quantity_unit 'oz',
--   unit_quantity_subtotal 0.3575, item_subtotal_price 4.29
--
-- Dropped and recreated rather than altered: the table was empty (0 rows), so
-- there is nothing to migrate and column order can be right from the start.
--
-- ── Renames ─────────────────────────────────────────────────────────────────
--   source_bill_file_id -> bill_file_id            (matches TransactionDetail)
--   brand               -> brand_name
--   observed_price      -> item_subtotal_price
--   observed_at         -> created_at
--   observed_context    -> item_qualitative_description
--
-- ── Gone ────────────────────────────────────────────────────────────────────
--   size_text        superseded by item_quantity + item_quantity_unit
--   currency         USD-only; free to re-add while the table is empty
--   category         the agent infers it from the description
--   latitude / longitude / place_label  -> one `location` text column
--
-- ── Why location is text, not coordinates ───────────────────────────────────
--
-- The store is resolved ON THE DEVICE at capture time — a GPS fix plus reverse
-- geocoding gives "Main St, Norwalk", and proximity to a known MerchantLocation
-- gives the merchant name. Only those answers are stored. Raw coordinates never
-- reach this table, which is strictly more private than the previous design
-- (which rounded to 4dp precisely because storing an exact fix would pinpoint a
-- home). MerchantLocation still holds lat/long: that is the learned-store list,
-- and matching against it needs them.
--
-- ── Three text fields, three sources ────────────────────────────────────────
--   item_qualitative_description  what the PHOTO showed that is not a number,
--                                 in the tag's own words
--   enriched_info                 generated later, same role it plays on
--                                 TransactionDetail
--   note                          what the USER said in chat — "for the party",
--                                 "cheaper than last week"
--
-- The first two join the embedded text: measured on labelled pairs, a bare tag
-- is close to unmatchable and the surrounding words are what make it findable
-- at the 0.75 floor.
--
-- `note` deliberately does NOT. The vector represents product identity, and an
-- occasion or an opinion is not identity — embedding "for the party" would pull
-- an unrelated item toward every other party purchase and weaken exactly the
-- match this table exists to make.
DROP TABLE IF EXISTS public."PriceObservation";

CREATE TABLE public."PriceObservation" (
    id                           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                      uuid NOT NULL REFERENCES public."User"(id) ON DELETE CASCADE,
    bill_file_id                 uuid REFERENCES public."BillFile"(id) ON DELETE CASCADE,

    merchant_name                text,
    location                     text,

    item_description             text,
    item_quantity                numeric,
    item_quantity_unit           text,
    unit_quantity_subtotal       numeric,
    item_subtotal_price          numeric,

    item_qualitative_description text,
    brand_name                   text,
    enriched_info                text,
    note                         text,

    embedding                    extensions.vector,
    embedding_model              text,
    created_at                   timestamptz NOT NULL DEFAULT now(),

    CHECK (item_quantity IS NULL OR item_quantity > 0),
    CHECK (unit_quantity_subtotal IS NULL OR unit_quantity_subtotal >= 0),
    CHECK (item_subtotal_price IS NULL OR item_subtotal_price >= 0)
);

CREATE INDEX idx_priceobs_user      ON public."PriceObservation"(user_id, created_at DESC);
CREATE INDEX idx_priceobs_bill_file ON public."PriceObservation"(bill_file_id);

ALTER TABLE public."PriceObservation" ENABLE ROW LEVEL SECURITY;
CREATE POLICY price_observation_own ON public."PriceObservation"
    FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

DROP FUNCTION IF EXISTS public.match_price_observations(text, text, integer, real);

CREATE OR REPLACE FUNCTION public.match_price_observations(
    p_query_embedding   text,
    p_embedding_model   text    DEFAULT NULL,
    p_limit             integer DEFAULT 10,
    p_min_semantic      real    DEFAULT 0.75
)
RETURNS TABLE (
    id uuid, bill_file_id uuid, merchant_name text, location text,
    item_description text, item_quantity numeric, item_quantity_unit text,
    unit_quantity_subtotal numeric, item_subtotal_price numeric,
    item_qualitative_description text, brand_name text, enriched_info text,
    note text, created_at timestamptz, score real
)
LANGUAGE sql STABLE SET search_path = public, extensions
AS $$
    SELECT o.id, o.bill_file_id, o.merchant_name, o.location,
           o.item_description, o.item_quantity, o.item_quantity_unit,
           o.unit_quantity_subtotal, o.item_subtotal_price,
           o.item_qualitative_description, o.brand_name, o.enriched_info,
           o.note, o.created_at,
           (1 - (o.embedding <=> p_query_embedding::extensions.vector))::real AS score
    FROM public."PriceObservation" o
    WHERE o.embedding IS NOT NULL
      AND (p_embedding_model IS NULL OR o.embedding_model IS NOT DISTINCT FROM p_embedding_model)
      AND (1 - (o.embedding <=> p_query_embedding::extensions.vector)) >= p_min_semantic
    ORDER BY score DESC, o.created_at DESC
    LIMIT p_limit;
$$;

REVOKE ALL ON FUNCTION public.match_price_observations(text, text, integer, real) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.match_price_observations(text, text, integer, real) TO authenticated;
