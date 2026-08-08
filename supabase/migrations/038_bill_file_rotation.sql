-- How far a stored photo must be turned to be read the right way up.
--
-- Phones record orientation inconsistently and a receipt is often shot
-- sideways on purpose, because it is long. The viewer could already rotate
-- one, but only for as long as it was open — the next person to look started
-- from the same sideways picture.
--
-- Stored as an ANGLE rather than by rewriting the image. The photo is the
-- evidence behind a financial record: re-encoding it loses a little quality
-- every time, an interrupted re-upload can leave a receipt half-replaced with
-- no other copy, and a stored angle is undone by setting it back to zero.
--
-- Degrees clockwise, and only the four quarter turns — anything else is a
-- crop problem, not a rotation.
ALTER TABLE "BillFile"
  ADD COLUMN IF NOT EXISTS rotation smallint NOT NULL DEFAULT 0;

ALTER TABLE "BillFile"
  DROP CONSTRAINT IF EXISTS billfile_rotation_quarter_turns;

ALTER TABLE "BillFile"
  ADD CONSTRAINT billfile_rotation_quarter_turns
  CHECK (rotation IN (0, 90, 180, 270));
