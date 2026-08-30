-- One account per email address.
--
-- auth.users already enforces this for password signups (users_email_partial_key),
-- but public."User" is only a mirror of it and asserted nothing. Both writers —
-- handle_new_user() and the login/register upserts in routers/auth.py — conflict
-- on `id`, so two auth rows carrying the same address would land here as two
-- rows without complaint. This makes the invariant the schema's job instead of
-- an assumption about what the auth service happens to do.
--
-- Case-insensitive: addresses are treated that way in practice, and the auth
-- service lowercases on signup. A row differing only by case therefore means
-- something wrote this table directly, which is exactly the drift worth failing
-- on rather than quietly storing.
--
-- Deliberately NOT normalising further. Stripping dots or +suffixes would merge
-- addresses that are genuinely distinct at most providers — that is Gmail's
-- local convention, not a property of email, and applying it here would lock a
-- real person out of their own signup.
--
-- NULL email stays permitted: a unique index allows many NULLs, the column has
-- always been nullable, and tightening that is not this migration's job.
CREATE UNIQUE INDEX IF NOT EXISTS user_email_lower_key
    ON public."User" (lower(email));

-- Keep the writer consistent with the index above. Normalising on the way in
-- means a mixed-case row can never be created in the first place, so the
-- constraint stays a backstop rather than becoming the thing users trip over.
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $function$
BEGIN
  INSERT INTO public."User" (id, email, hashed_password)
  VALUES (
    NEW.id,
    lower(NEW.email),
    'managed_by_supabase_auth'
  )
  ON CONFLICT (id) DO UPDATE SET
    email = EXCLUDED.email;
  RETURN NEW;
END;
$function$;
