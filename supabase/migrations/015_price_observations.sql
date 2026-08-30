-- A price tag records what something COSTS, not that it was bought. These
-- deliberately do not live in "Transaction": writing them there would inflate
-- every spending total and chart, and would break the agent's "spending is
-- positive" premise — you would "spend" money by walking through a store.
--
-- Note there is no content_hash / dedup constraint, unlike "Transaction".
-- Repeat observations of the same item are the whole point: they are the price
-- history that later comparisons are drawn from.
CREATE TABLE IF NOT EXISTS public."PriceObservation" (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             uuid NOT NULL REFERENCES public."User"(id) ON DELETE CASCADE,
    source_bill_file_id uuid REFERENCES public."BillFile"(id) ON DELETE CASCADE,

    -- What was seen. normalized_name is the join key against
    -- "TransactionDetail".normalized_name (see 017) — lowercased, de-branded,
    -- singularised — so "Cheerios 12oz" and "cheerios" can be matched.
    item_description    text NOT NULL,          -- verbatim from the tag
    normalized_name     text,
    brand               text,

    -- Size is not optional detail: $4.99 is meaningless without "for 12 oz".
    -- Comparing a 12oz jar against a 20oz jar without normalising to unit_price
    -- produces confidently wrong answers, so size is stored parsed AND verbatim.
    size_value          numeric,                -- 12
    size_unit           text,                   -- 'oz' (normalised)
    size_text           text,                   -- "12 OZ" as printed

    observed_price      numeric NOT NULL,
    unit_price          numeric,                -- observed_price / size_value
    currency            text NOT NULL DEFAULT 'USD',

    -- "2 for $5" is only $2.50/unit if you buy two, so the offer is kept
    -- verbatim and the review screen asks rather than silently dividing.
    is_promotional      boolean NOT NULL DEFAULT false,
    promo_text          text,
    -- When the offer stops ("Sale ends 8/15"). Without this a limited-time
    -- price silently becomes the baseline that later prices are judged against,
    -- making every normal price look like a rip-off.
    promo_ends_on       date,
    -- Use-by / best-before printed on the product, when visible. This is the
    -- one that prevents a genuinely misleading recommendation: stores discount
    -- meat and dairy *because* they expire tomorrow, so "40% below your usual"
    -- is only a good deal if you are cooking it tonight. Comparisons must be
    -- able to say why the price is low rather than just that it is.
    expires_on          date,

    -- A shelf tag almost never names the store, so where the photo was taken is
    -- usually the only signal for which merchant this price belongs to.
    -- Coordinates are rounded to 4dp (~11m) before insert: enough to match a
    -- storefront, not enough to pinpoint a home. Capture is opt-in and the
    -- whole feature works with these null (the user types the store instead).
    merchant_name       text,
    location            text,
    latitude            numeric,
    longitude           numeric,
    place_label         text,           -- reverse-geocoded, e.g. "Main St, Norwalk"
    observed_at         timestamptz NOT NULL DEFAULT now(),

    category            text,
    raw_ocr             jsonb,                  -- full vision output, for re-review
    verified            boolean NOT NULL DEFAULT false,
    note                text,
    created_at          timestamptz NOT NULL DEFAULT now(),

    CHECK (observed_price >= 0),
    CHECK (size_value IS NULL OR size_value > 0)
);

-- The dominant read: "what have I seen this item priced at, most recent first".
CREATE INDEX IF NOT EXISTS idx_priceobs_user_item
    ON public."PriceObservation"(user_id, normalized_name, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_priceobs_user_observed
    ON public."PriceObservation"(user_id, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_priceobs_file
    ON public."PriceObservation"(source_bill_file_id);

ALTER TABLE public."PriceObservation" ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS price_observation_own ON public."PriceObservation";
CREATE POLICY price_observation_own ON public."PriceObservation"
    FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
