-- One camera button now captures two different things: a receipt (many items,
-- totals, tax) or a shelf price tag (one item, one price). The file row is
-- created at upload time, before the vision model has classified the image, so
-- the kind lives on the file and starts out unresolved.
--
-- 'unknown' is a real state, not a failure: a receipt photographed on a shelf is
-- genuinely ambiguous, and the app asks the user rather than guessing — a wrong
-- guess that gets verified writes junk into the ledger silently.
--
-- The DEFAULT exists to backfill rows that predate this column: every one of
-- them is a receipt, so no data migration is needed.
--
-- New captures do NOT rely on it. capture_service inserts kind='unknown'
-- explicitly and only writes the real kind once the vision model has answered,
-- because the row is created before classification runs. If that step fails,
-- inheriting 'receipt' would leave an unexamined photo claiming to be a receipt
-- — which a user could then confirm, inventing spending that never happened.
ALTER TABLE public."BillFile"
    ADD COLUMN IF NOT EXISTS kind text NOT NULL DEFAULT 'receipt';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'billfile_kind_check'
    ) THEN
        ALTER TABLE public."BillFile"
            ADD CONSTRAINT billfile_kind_check
            CHECK (kind IN ('receipt', 'price_tag', 'unknown'));
    END IF;
END $$;

-- The capture screen polls for images still awaiting classification/review.
CREATE INDEX IF NOT EXISTS idx_billfile_user_kind
    ON public."BillFile"(user_id, kind);
