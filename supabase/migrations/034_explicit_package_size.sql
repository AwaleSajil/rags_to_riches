-- One meaning per column, on both sides of a price comparison.
--
-- These two tables were shaped to mirror each other so the comparison would
-- need no translation layer. They did not: the same two column names held
-- different facts.
--
--   PriceObservation.item_quantity  was the PACKAGE SIZE ("$4.29 / 12 OZ" -> 12
--                                   'oz'). Nothing is bought from a shelf tag,
--                                   so there was never a count to store.
--   TransactionDetail.item_quantity is HOW MANY YOU BOUGHT — 1 bag, 2.25 lb —
--                                   and the size of one of them had nowhere to
--                                   live at all.
--
-- So the comparison grew two different functions to read the same column names
-- two different ways, and the package size on the receipt side had to be
-- re-parsed out of the description every time. That parse is not always
-- decidable: "+RED POTA 5L US#" is a five POUND bag, read as five litres, which
-- made a $4.99 bag look like $1.00/l and dropped the user's most relevant
-- purchase out of every potato comparison as "different units".
--
-- PriceObservation is a rename only — the values were already sizes.
-- TransactionDetail gains the columns it never had.
--
-- This partially reverses migration 024, which dropped size_value/size_unit
-- because a PARSED size was wrong ~26% of the time. The objection stands and is
-- the reason these stay nullable: a size is only as good as what put it there.
-- What changed is that they are now confirmable — the vision pass reads them off
-- the label and the review screens let a human fix them, rather than a regex
-- guessing from an abbreviation.

ALTER TABLE "PriceObservation" RENAME COLUMN item_quantity TO size_value;
ALTER TABLE "PriceObservation" RENAME COLUMN item_quantity_unit TO size_unit;

COMMENT ON COLUMN "PriceObservation".size_value IS
    'How much is in the package the tag prices, e.g. 12 for "12 OZ".';
COMMENT ON COLUMN "PriceObservation".size_unit IS
    'Unit of size_value as PRINTED — oz, lb, g, ml, l, gal, ct. Never converted.';

ALTER TABLE "TransactionDetail" ADD COLUMN IF NOT EXISTS size_value numeric;
ALTER TABLE "TransactionDetail" ADD COLUMN IF NOT EXISTS size_unit text;

COMMENT ON COLUMN "TransactionDetail".size_value IS
    'How much is in ONE purchase unit — 5 for a 5 lb bag. NULL when unknown; '
    'callers fall back to parsing item_description and must treat that as a guess.';
COMMENT ON COLUMN "TransactionDetail".size_unit IS
    'Unit of size_value. Distinct from item_quantity_unit, which is what '
    'item_quantity counts: a 5 lb bag is item_quantity 1 "each", size 5 "lb".';
