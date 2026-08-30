-- The unit a shelf tag's PRINTED per-unit price is quoted in.
--
-- Stores very often quote a unit that is not the package unit: a one-gallon
-- milk jug is commonly tagged "UNIT PRICE PER QUART 0.87". Without recording
-- which unit the printed figure refers to, $0.87 gets filed as the price of a
-- whole gallon that actually costs $3.49, and every later comparison against
-- it is wrong by a factor of four.
--
-- Null when the tag does not say — the vision prompt is explicit that this must
-- be copied from the label, never computed.
--
-- RECONSTRUCTED. This migration was applied to the database before its file was
-- committed, so this describes the column as it exists rather than the original
-- statement. Written to match the live schema: text, nullable, no default.
ALTER TABLE "PriceObservation"
  ADD COLUMN IF NOT EXISTS unit_price_unit text;
