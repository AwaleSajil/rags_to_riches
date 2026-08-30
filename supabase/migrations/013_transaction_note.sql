-- Free-text note a user can attach to any transaction while viewing it.
-- The note is folded into the transaction's embedding text (see
-- vector_db_client.sync_transactions), so notes are semantically searchable.
ALTER TABLE public."Transaction" ADD COLUMN IF NOT EXISTS note text;
