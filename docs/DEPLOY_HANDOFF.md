# Deploy handoff — OCI, rags2riches.duckdns.org

Written 30 Aug 2026 from the laptop side, for whoever picks this up on the VM
(likely a Claude session over SSH remote dev). Everything about the *code* below
was verified; everything about *how this box deploys* was not, because that
process is not written down anywhere in this repo. Fixing that is task 5.

`origin/main` is at `3535ea2` — PR #5, 64 commits, CI green.

---

## 1. Why this redeploy is not optional

Two things are broken in production **right now**, and neither is fixed without
a rebuild.

**Signup is broken.** Email confirmation was switched on at the Supabase project
level this session (`mailer_autoconfirm: false`). Every new signup now gets a
verification email whose link returns to `/auth/callback`. The deployed bundle
predates that route entirely, so the link lands on a page that cannot consume
the token: the account is confirmed server-side, the user sees a login screen,
and nothing explains why. This was reproduced repeatedly before the cause was
found.

**Account deletion answers 503.** `SUPABASE_SERVICE_KEY` is almost certainly
unset on this box — it is listed as outstanding in `HANDOFF.md` §4. Deleting an
auth record is the one thing that key is used for, and the anon key cannot do
it. Both stores require a working in-app delete, so this is a submission
blocker, not a nicety.

The first is a code problem and the rebuild fixes it. The second is an
environment problem and the rebuild will *not* fix it on its own.

---

## 2. Environment — what must be set

From `.env.example`. The two marked **NEW** are the ones this session added
pressure to; the rest should already exist.

| Variable | Notes |
|---|---|
| `SUPABASE_URL` | |
| `SUPABASE_KEY` | The **anon** key. `/api/v1/public-config` serves this to unauthenticated callers. The API refuses to boot if it detects a service key here. |
| `DATABASE_URL` | Structured queries and pgvector both. |
| `APP_ENCRYPTION_KEY` | **Must be byte-identical to the current value.** It is the Fernet key encrypting every user's stored LLM API key. A new one does not error — it silently makes existing keys undecryptable. Copy it off the running container before you change anything. |
| `SUPABASE_SERVICE_KEY` | **NEW** — service_role key. Without it deletion 503s. Never reaches a client; there is a test holding that line. |
| `SUPPORT_EMAIL` | `r2rapp.support@gmail.com`. Renders on `/privacy`, `/terms`, `/account-deletion`. |
| `PUBLISHER_NAME` | `Sajil Awale`. Blank renders a visible `[not configured]` marker. |
| `ALLOWED_ORIGINS` | Optional. The web build is served same-origin by this process, so usually empty. |

---

## 3. The rebuild

The root `Dockerfile` is a two-stage build and both halves matter here:

- **Stage 1** runs `npx expo export --platform web` and produces `dist`.
- **Stage 2** copies that into `./static/`, which `backend/main.py` serves with
  an SPA fallback.

So **the frontend fix ships automatically with a backend image rebuild** — there
is no separate frontend deploy step. Do not skip stage 1's cache.

Two deliberate details in stage 1, worth not "fixing": it deletes any `.env*`
and forces `EXPO_PUBLIC_API_URL=""`, so the bundle talks to a *relative*
`/api/v1` and is therefore correct wherever it is served. A baked absolute URL
is the bug that produces a working page making requests to the wrong host.

The container listens on **7860** and runs uvicorn with `--proxy-headers
--forwarded-allow-ips *`, i.e. it expects to sit behind the TLS terminator that
fronts `rags2riches.duckdns.org`.

Sequence, whatever the mechanism turns out to be:

1. Capture the current env (especially `APP_ENCRYPTION_KEY`) before touching
   anything.
2. `git pull` to `3535ea2` or later on `main`.
3. Add `SUPABASE_SERVICE_KEY` wherever this box keeps its env.
4. Rebuild the image — no cached stage 1.
5. Restart, keeping the old image tagged so a rollback is one command.

---

## 4. Verify, in this order

Cheapest first; each one fails distinctly.

1. **`/privacy`, `/terms`, `/account-deletion` render with no `[not configured]`
   markers.** Proves `PUBLISHER_NAME` and `SUPPORT_EMAIL` reached the process.
2. **Sign up with a fresh address.** Expect: "Check your email", then a link
   that returns to the site and shows a green ✓ "Email verified" screen, then
   the chat tab. If it dumps you on a login screen, the old bundle is still
   being served — check that stage 1 actually re-ran.
3. **Sign up again with that same address.** Expect "An account already exists
   for this email. Try signing in instead." Supabase answers duplicates with a
   success response on purpose, so the wrong behaviour here is a cheerful
   "check your email" for mail that will never arrive.
4. **Forgot password** on that account. Expect a "Choose a new password" screen,
   *not* a silent sign-in. A silent sign-in means the `type=recovery` branch is
   not in the running bundle.
5. **Delete the account from Settings.** Expect success, not 503. This is the
   `SUPABASE_SERVICE_KEY` check and the one both stores care about.
6. **The six fixes from `HANDOFF.md` §4**, still unproven against reality:
   upload a CSV; photograph and verify a receipt; then edit one line item and
   confirm the unit and size survive. That last one is bug #2 and is the one
   worth watching.

Redirect URLs are already allow-listed on the Supabase project, including
`https://rags2riches.duckdns.org/**`, and `site_url` points there. Nothing to
change on that side.

---

## 5. Write down how this box deploys

This is the actual deliverable of this handoff.

Nothing in the repo describes it. `README.md` §Deployment still documents two
Hugging Face Spaces, which this project has moved off; the only trace of the
current host anywhere is a URL in `frontend/eas.json`. That is how a redeploy
became a blocking question that had to be asked instead of read.

Once the steps are known, put them in `README.md` under Deployment — replacing
the HF Spaces section, not appended beside it — covering: where the checkout
lives, how the image is built and restarted, where env vars are stored, what
terminates TLS, and how to roll back. `HANDOFF.md` §4 also needs its "HF Spaces
→ repository secrets" line corrected to whatever is true.

Automating this in CI is explicitly **not** the next step. `.github/workflows/ci.yml`
runs tests and typecheck only, and that is the right scope until the manual
process has been performed once and written down. Encoding an undocumented
process into a workflow just moves the guesswork somewhere harder to read.

---

## 6. Not covered here

`r2r://auth/callback` — the custom scheme the shipped Android app uses — has
only ever been exercised through its web and `exp://` equivalents. It needs a
real installed build, so it gets tested with the Play build and not before.

The Brevo sender is a `gmail.com` address, which cannot align DKIM. Verification
mail therefore has a real chance of being filtered. That is survivable for one
person testing and becomes a launch blocker the moment twelve testers each need
to receive a link — a ~$10 domain verified in Brevo fixes it.
