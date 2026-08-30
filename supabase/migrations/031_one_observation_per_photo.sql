-- One photo is one price sighting.
--
-- Confirming a price tag inserted a fresh row every time, so re-opening a photo
-- and pressing Compare again produced a second observation of the same sighting
-- (measured: two rows for one photo, 92 seconds apart). That is not harmless
-- duplication — the whole point of this table is to be evidence, and one photo
-- counted twice reads as two independent sightings agreeing with each other.
--
-- Re-photographing the same item next week is a NEW BillFile, so it still gets
-- its own row. Only re-confirming the SAME photo collapses.
--
-- Partial, because bill_file_id is null for prices the user simply mentions in
-- chat ("milk is $3.49 today"). Each of those genuinely is a separate sighting
-- and must stay insertable.

-- Collapse existing duplicates first, keeping the most recent — it reflects any
-- corrections the user made on the card before pressing Compare again.
DELETE FROM "PriceObservation" a
USING "PriceObservation" b
WHERE a.bill_file_id IS NOT NULL
  AND a.bill_file_id = b.bill_file_id
  AND a.user_id = b.user_id
  AND (a.created_at, a.id) < (b.created_at, b.id);

CREATE UNIQUE INDEX IF NOT EXISTS price_observation_one_per_photo
    ON "PriceObservation" (user_id, bill_file_id)
    WHERE bill_file_id IS NOT NULL;
