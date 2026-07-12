-- ============================================================
-- BASE SCHEMA — run this FIRST (before 001/002/003)
-- Reconstructed from the application code (db_client.py, money_rag.py,
-- routers/*, schemas/*) because the original tables were created in the
-- old Supabase project and were never committed to this repo.
--
-- Run in: Supabase Dashboard > SQL Editor.
-- Tables are quoted PascalCase to match the code (.table("User"), etc.).
-- ============================================================

-- ---------- pgvector (semantic search, replaces Qdrant) ----------
-- Pre-installed on Supabase in the `extensions` schema; this is idempotent.
-- The app's vector tables (langchain_pg_collection / langchain_pg_embedding)
-- are created automatically by langchain_postgres on first ingest.
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

-- ---------- User (mirrors auth.users) ----------
CREATE TABLE IF NOT EXISTS public."User" (
    id              uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email           text,
    hashed_password text,
    created_at      timestamptz NOT NULL DEFAULT now()
);

-- ---------- AccountConfig (per-user LLM settings) ----------
CREATE TABLE IF NOT EXISTS public."AccountConfig" (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         uuid NOT NULL REFERENCES public."User"(id) ON DELETE CASCADE,
    llm_provider    text,
    api_key         text,
    decode_model    text,
    embedding_model text,
    deep_enrichment boolean NOT NULL DEFAULT false,
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (user_id)
);

-- ---------- CSVFile ----------
CREATE TABLE IF NOT EXISTS public."CSVFile" (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     uuid NOT NULL REFERENCES public."User"(id) ON DELETE CASCADE,
    filename    text,
    s3_key      text,
    upload_date timestamptz NOT NULL DEFAULT now()
);

-- ---------- BillFile (receipt/bill images) ----------
CREATE TABLE IF NOT EXISTS public."BillFile" (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id        uuid NOT NULL REFERENCES public."User"(id) ON DELETE CASCADE,
    filename       text,
    s3_key         text,
    upload_date    timestamptz NOT NULL DEFAULT now(),
    -- Raw JSON the vision LLM extracted from the receipt (money_rag._ingest_bill).
    raw_ocr_string text
);

-- ---------- Transaction ----------
CREATE TABLE IF NOT EXISTS public."Transaction" (
    id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id              uuid NOT NULL REFERENCES public."User"(id) ON DELETE CASCADE,
    trans_date           date,
    description          text,
    amount               numeric,
    category             text DEFAULT 'Uncategorized',
    merchant_name        text,
    enriched_info        text,
    content_hash         text,
    source               text,               -- 'csv' | 'bill'
    source_csv_id        uuid REFERENCES public."CSVFile"(id) ON DELETE CASCADE,
    source_bill_file_id  uuid REFERENCES public."BillFile"(id) ON DELETE CASCADE,
    created_at           timestamptz NOT NULL DEFAULT now(),
    -- Required for the code's upsert(on_conflict="content_hash").
    -- NOTE: hash = date+amount+first-merchant-word (NOT user-scoped), so this
    -- unique constraint is global. If you want two users to be able to hold the
    -- same transaction, change to UNIQUE (user_id, content_hash) AND update the
    -- on_conflict key in money_rag.py accordingly.
    UNIQUE (content_hash)
);
CREATE INDEX IF NOT EXISTS idx_transaction_user      ON public."Transaction"(user_id);
CREATE INDEX IF NOT EXISTS idx_transaction_csv       ON public."Transaction"(source_csv_id);
CREATE INDEX IF NOT EXISTS idx_transaction_bill      ON public."Transaction"(source_bill_file_id);

-- ---------- TransactionDetail (line items from bills) ----------
CREATE TABLE IF NOT EXISTS public."TransactionDetail" (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id   uuid REFERENCES public."Transaction"(id) ON DELETE CASCADE,
    user_id          uuid NOT NULL REFERENCES public."User"(id) ON DELETE CASCADE,
    bill_file_id     uuid REFERENCES public."BillFile"(id) ON DELETE CASCADE,
    item_description text,
    item_quantity    numeric,
    item_unit_price  numeric,
    tax_amount       numeric,
    item_total_price numeric,
    enriched_info    text,
    created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_txdetail_user ON public."TransactionDetail"(user_id);
CREATE INDEX IF NOT EXISTS idx_txdetail_tx   ON public."TransactionDetail"(transaction_id);

-- ============================================================
-- Row-Level Security
-- The backend calls Supabase with the end-user's JWT (see dependencies.get_supabase),
-- so auth.uid() is the logged-in user. Each user may only touch their own rows.
-- ============================================================
ALTER TABLE public."User"              ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."AccountConfig"     ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."CSVFile"           ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."BillFile"          ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."Transaction"       ENABLE ROW LEVEL SECURITY;
ALTER TABLE public."TransactionDetail" ENABLE ROW LEVEL SECURITY;

-- User: a row is "yours" when its id == auth.uid()
DROP POLICY IF EXISTS user_self ON public."User";
CREATE POLICY user_self ON public."User"
    FOR ALL USING (auth.uid() = id) WITH CHECK (auth.uid() = id);

-- Everything else keys off user_id
DROP POLICY IF EXISTS acctcfg_own ON public."AccountConfig";
CREATE POLICY acctcfg_own ON public."AccountConfig"
    FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS csvfile_own ON public."CSVFile";
CREATE POLICY csvfile_own ON public."CSVFile"
    FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS billfile_own ON public."BillFile";
CREATE POLICY billfile_own ON public."BillFile"
    FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS transaction_own ON public."Transaction";
CREATE POLICY transaction_own ON public."Transaction"
    FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS txdetail_own ON public."TransactionDetail";
CREATE POLICY txdetail_own ON public."TransactionDetail"
    FOR ALL USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

-- ============================================================
-- Storage bucket for uploaded CSVs / bill images.
-- Code uses: client.storage.from_("money-rag-files"), keys = "{user_id}/{folder}/{file}"
-- ============================================================
INSERT INTO storage.buckets (id, name, public)
VALUES ('money-rag-files', 'money-rag-files', false)
ON CONFLICT (id) DO NOTHING;

-- Each user may only touch objects under their own "{uid}/..." prefix.
DROP POLICY IF EXISTS moneyrag_files_own ON storage.objects;
CREATE POLICY moneyrag_files_own ON storage.objects
    FOR ALL
    USING (bucket_id = 'money-rag-files' AND (storage.foldername(name))[1] = auth.uid()::text)
    WITH CHECK (bucket_id = 'money-rag-files' AND (storage.foldername(name))[1] = auth.uid()::text);
