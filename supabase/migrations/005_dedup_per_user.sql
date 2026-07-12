-- ============================================================
-- Scope transaction de-duplication to the user.
--
-- Before: UNIQUE (content_hash) was global — two different users could not
-- hold the same (date, amount, merchant) transaction, and a cross-user hash
-- collision could make the ON CONFLICT upsert misbehave under RLS.
-- After: UNIQUE (user_id, content_hash) — dedup is per-user.
--
-- Paired with money_rag.py using on_conflict="user_id,content_hash".
-- ============================================================

ALTER TABLE public."Transaction" DROP CONSTRAINT IF EXISTS "Transaction_content_hash_key";

ALTER TABLE public."Transaction"
    ADD CONSTRAINT "Transaction_user_content_hash_key" UNIQUE (user_id, content_hash);
