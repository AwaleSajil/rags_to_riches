-- ============================================================
-- Trigger: Auto-populate public."User" when a new auth user signs up
-- Run this in the Supabase SQL Editor (Dashboard > SQL Editor)
-- ============================================================

-- 1. Create the trigger function
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER          -- runs as the DB owner, bypasses RLS
SET search_path = public  -- prevents search-path hijacking
AS $$
BEGIN
  INSERT INTO public."User" (id, email, hashed_password)
  VALUES (
    NEW.id,
    NEW.email,
    'managed_by_supabase_auth'
  )
  ON CONFLICT (id) DO UPDATE SET
    email = EXCLUDED.email;
  RETURN NEW;
END;
$$;

-- 2. Drop existing trigger if it exists (idempotent)
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;

-- 3. Create the trigger
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW
  EXECUTE FUNCTION public.handle_new_user();
