-- ============================================================
-- Backfill: Insert any existing auth users into public."User"
-- Run this AFTER 001_auto_sync_user_trigger.sql
-- ============================================================

INSERT INTO public."User" (id, email, hashed_password)
SELECT id, email, 'managed_by_supabase_auth'
FROM auth.users
ON CONFLICT (id) DO UPDATE SET
  email = EXCLUDED.email;
