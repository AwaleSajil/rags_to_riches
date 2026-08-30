-- One photo can show several price tags.
--
-- Migration 031 keyed an observation to its photo, which stopped a re-confirm
-- from duplicating a sighting. That was right about re-confirms and wrong about
-- shelves: a photo of a dairy case holds a tag per product, and the previous key
-- made storing more than one of them structurally impossible.
--
-- The key gains the tag's POSITION in the photo. Deliberately not the item
-- description: that is editable on the confirm card, so keying on it would move
-- the key out from under the row the moment a user fixed a misread name, and the
-- correction would insert a second observation instead of amending the first.

ALTER TABLE "PriceObservation"
    ADD COLUMN IF NOT EXISTS tag_index integer NOT NULL DEFAULT 0;

COMMENT ON COLUMN "PriceObservation".tag_index IS
    'Which tag in the source photo this came from, 0-based. Always 0 for prices '
    'mentioned in chat rather than photographed.';

DROP INDEX IF EXISTS price_observation_one_per_photo;

CREATE UNIQUE INDEX IF NOT EXISTS price_observation_one_per_tag
    ON "PriceObservation" (user_id, bill_file_id, tag_index)
    WHERE bill_file_id IS NOT NULL;
