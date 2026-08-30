/**
 * The minimum password length the UI enforces.
 *
 * This must match `password_min_length` on the Supabase project. A client-only
 * check is decorative — the API accepts whatever the server allows, so anyone
 * calling it directly, or running an older build, gets the server's floor
 * instead of this one. Changing it here means changing it there too.
 *
 * Shared rather than redeclared so the signup screen and the password-reset
 * screen cannot drift apart and reject each other's passwords.
 */
export const MIN_PASSWORD_LENGTH = 10;
