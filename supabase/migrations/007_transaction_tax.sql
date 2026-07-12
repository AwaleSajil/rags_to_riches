-- Bill-level tax with support for multiple rates within one receipt.
--   subtotal      = pre-tax sum
--   tax_total     = sum of all tax lines
--   tax_breakdown = [{label, rate, amount}, ...]  (one entry per distinct rate)
-- Per line item, taxable marks whether that item was taxed.
ALTER TABLE public."Transaction"       ADD COLUMN IF NOT EXISTS subtotal      numeric;
ALTER TABLE public."Transaction"       ADD COLUMN IF NOT EXISTS tax_total     numeric;
ALTER TABLE public."Transaction"       ADD COLUMN IF NOT EXISTS tax_breakdown jsonb;
ALTER TABLE public."TransactionDetail" ADD COLUMN IF NOT EXISTS taxable       boolean;
ALTER TABLE public."TransactionDetail" ADD COLUMN IF NOT EXISTS tax_rate      numeric;  -- rate applied to THIS item (0 = exempt)
