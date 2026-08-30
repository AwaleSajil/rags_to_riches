-- Clarify that line-item prices are PRE-TAX, and add a true post-tax line total.
--   item_unit_price  -> item_unit_subtotal_price   (pre-tax unit price)
--   item_total_price -> item_subtotal_price         (pre-tax line total = qty x unit)
--   tax_amount        = item_subtotal_price * tax_rate/100   (was always 0)
--   item_total_price  = item_subtotal_price + tax_amount     (NEW, post-tax line total)
-- Guarded so it is safe to re-run.
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns
             WHERE table_schema = 'public' AND table_name = 'TransactionDetail'
               AND column_name = 'item_unit_price') THEN
    ALTER TABLE public."TransactionDetail"
      RENAME COLUMN item_unit_price TO item_unit_subtotal_price;
  END IF;

  IF EXISTS (SELECT 1 FROM information_schema.columns
             WHERE table_schema = 'public' AND table_name = 'TransactionDetail'
               AND column_name = 'item_total_price')
     AND NOT EXISTS (SELECT 1 FROM information_schema.columns
             WHERE table_schema = 'public' AND table_name = 'TransactionDetail'
               AND column_name = 'item_subtotal_price') THEN
    ALTER TABLE public."TransactionDetail"
      RENAME COLUMN item_total_price TO item_subtotal_price;
  END IF;
END $$;

-- New post-tax line total column.
ALTER TABLE public."TransactionDetail"
  ADD COLUMN IF NOT EXISTS item_total_price numeric;  -- post-tax line total

-- Backfill: derive per-item tax from the rate (0 for exempt), then the post-tax total.
UPDATE public."TransactionDetail"
   SET tax_amount = round(COALESCE(item_subtotal_price, 0) * COALESCE(tax_rate, 0) / 100.0, 2);
UPDATE public."TransactionDetail"
   SET item_total_price = COALESCE(item_subtotal_price, 0) + COALESCE(tax_amount, 0);
