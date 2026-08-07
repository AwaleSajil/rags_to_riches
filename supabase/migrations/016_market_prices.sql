-- Cached retailer prices used to answer "is this a good price?" against
-- somewhere other than the user's own history.
--
-- Deliberately NOT user-scoped. This is reference data, not personal data: what
-- a user looks up is private, but the resulting shelf price at Walmart is not.
-- A shared cache means one user's lookup spares everyone else the API call,
-- which matters because the lookup costs an LLM round trip and the Gemini free
-- tier allows very few per day.
CREATE TABLE IF NOT EXISTS public."MarketPrice" (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    normalized_name text NOT NULL,          -- join key, see price_service.normalize_item_name
    brand           text,
    size_value      numeric,
    size_unit       text,
    merchant_name   text NOT NULL,          -- 'Walmart', 'Stop & Shop', ...
    price           numeric NOT NULL,
    unit_price      numeric,                -- price per size_unit; the comparable number
    currency        text NOT NULL DEFAULT 'USD',
    region          text,                   -- grocery prices vary a lot by market
    source_url      text,
    -- How much to trust this row. Search + LLM extraction is noisy, so the UI
    -- shows provenance and age rather than presenting every hit as fact.
    confidence      numeric,
    retrieved_at    timestamptz NOT NULL DEFAULT now(),
    CHECK (price >= 0),
    CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    UNIQUE (normalized_name, brand, size_value, size_unit, merchant_name, region)
);

-- Lookup path: "what do my reference merchants charge for this item".
CREATE INDEX IF NOT EXISTS idx_marketprice_lookup
    ON public."MarketPrice"(normalized_name, merchant_name, retrieved_at DESC);

ALTER TABLE public."MarketPrice" ENABLE ROW LEVEL SECURITY;

-- Readable by any signed-in user; writes go through the backend's service-role
-- connection only, so a client cannot poison the shared cache.
DROP POLICY IF EXISTS market_price_read ON public."MarketPrice";
CREATE POLICY market_price_read ON public."MarketPrice"
    FOR SELECT TO authenticated USING (true);

-- Which retailers a given user wants compared, and where they shop. Region
-- keeps a Connecticut user from being shown California prices.
ALTER TABLE public."AccountConfig"
    ADD COLUMN IF NOT EXISTS reference_merchants text[] NOT NULL DEFAULT ARRAY['Walmart'],
    ADD COLUMN IF NOT EXISTS home_region text,
    -- Location capture is opt-in and off by default; see 015 for why it exists.
    ADD COLUMN IF NOT EXISTS location_capture_enabled boolean NOT NULL DEFAULT false;
