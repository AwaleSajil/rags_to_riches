-- Stores the user has confirmed being at, so the app stops asking.
--
-- The first price check at a new store asks "which shop is this?"; the answer
-- is written here with the coordinates, and every later capture within ~150m
-- resolves the merchant automatically. This is what makes store identification
-- work without a billed Places API — it just takes one visit to learn.
--
-- Proximity is computed in Python (haversine over the user's small list) rather
-- than in SQL, so no PostGIS or earthdistance extension is required.
CREATE TABLE IF NOT EXISTS public."MerchantLocation" (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       uuid NOT NULL REFERENCES public."User"(id) ON DELETE CASCADE,
    merchant_name text NOT NULL,
    -- Rounded to 4dp (~11m) by the caller, matching PriceObservation.
    latitude      numeric NOT NULL,
    longitude     numeric NOT NULL,
    address       text,
    city          text,
    region        text,
    visit_count   integer NOT NULL DEFAULT 1,
    last_seen_at  timestamptz NOT NULL DEFAULT now(),
    created_at    timestamptz NOT NULL DEFAULT now(),
    CHECK (latitude BETWEEN -90 AND 90),
    CHECK (longitude BETWEEN -180 AND 180),
    UNIQUE (user_id, merchant_name, latitude, longitude)
);

-- The whole per-user set is loaded to run the distance match, so index on owner.
CREATE INDEX IF NOT EXISTS idx_merchantlocation_user
    ON public."MerchantLocation"(user_id, last_seen_at DESC);

ALTER TABLE public."MerchantLocation" ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS merchant_location_own ON public."MerchantLocation";
CREATE POLICY merchant_location_own ON public."MerchantLocation"
    FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
