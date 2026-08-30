-- Discounts / savings on receipts.
--
-- Two distinct things get printed on a receipt and must not be confused:
--
--   1. Item-level markdowns ("SAVINGS x.xx-" under a line, with a "PRICE YOU PAY"
--      that is already the net). These are ALREADY reflected in the item price we
--      store, so the line totals already sum to the balance. We only record how
--      much was marked down, for display ("you saved $X"). This is item_savings.
--
--   2. Order-level coupons that are subtracted from the whole basket (e.g.
--      "$5 off $50"). These reduce the balance and are NOT baked into any single
--      line item, so they are subtracted from the header amount. This is
--      discount_total.
--
-- A receipt's "Savings Summary / Card Savings / Total Savings" block is only a
-- recap of (1) and must never be subtracted again — doing so double-counts.
--
--   savings_total = sum(item_savings) + discount_total   (display only)
ALTER TABLE public."TransactionDetail" ADD COLUMN IF NOT EXISTS item_savings   numeric;  -- per-line markdown (regular - net paid), informational
ALTER TABLE public."Transaction"       ADD COLUMN IF NOT EXISTS discount_total numeric;  -- order-level coupons subtracted from the basket
ALTER TABLE public."Transaction"       ADD COLUMN IF NOT EXISTS savings_total  numeric;  -- total the shopper saved (markdowns + coupons), display only
