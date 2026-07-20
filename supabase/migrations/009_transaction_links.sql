-- Link matching source records without deleting either transaction.
-- Examples: the same bank transaction present in two CSV exports, or a bank
-- CSV entry matched to its receipt image.
CREATE TABLE IF NOT EXISTS public."TransactionLink" (
    id                    uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id               uuid NOT NULL REFERENCES public."User"(id) ON DELETE CASCADE,
    transaction_id        uuid NOT NULL REFERENCES public."Transaction"(id) ON DELETE CASCADE,
    linked_transaction_id uuid NOT NULL REFERENCES public."Transaction"(id) ON DELETE CASCADE,
    match_type            text NOT NULL, -- csv_csv | csv_receipt
    confidence            numeric NOT NULL DEFAULT 1,
    created_at            timestamptz NOT NULL DEFAULT now(),
    CHECK (transaction_id <> linked_transaction_id),
    UNIQUE (user_id, transaction_id, linked_transaction_id)
);

CREATE INDEX IF NOT EXISTS idx_transaction_link_left
    ON public."TransactionLink"(user_id, transaction_id);
CREATE INDEX IF NOT EXISTS idx_transaction_link_right
    ON public."TransactionLink"(user_id, linked_transaction_id);

ALTER TABLE public."TransactionLink" ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS transaction_link_own ON public."TransactionLink";
CREATE POLICY transaction_link_own ON public."TransactionLink"
    FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
