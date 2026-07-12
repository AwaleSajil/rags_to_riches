-- Optional location for a transaction: store address/city from a receipt image,
-- or a location column found in a CSV. Nullable — not every source has it.
ALTER TABLE public."Transaction" ADD COLUMN IF NOT EXISTS location text;
