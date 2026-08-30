# R2R — session handoff

Written 29 Aug 2026 at the end of a long session. Everything below was verified
in that session unless it says otherwise.

---

## 0. State of the branch — the work is committed

Re-verified 29 Aug 2026. Working tree clean; §4 is where to start.

```
branch: feat/pgvector-and-mobile
740 tests passing, frontend typecheck clean
```

The session's work landed in four commits, each standing alone:

| Commit | Contents |
|---|---|
| `f2b44d8` | **security** — path traversal in the SPA fallback (`backend/main.py`) + `tests/test_static_fallback.py` |
| `15e1487` | **schemas** — line-item unit/size round trip (`backend/schemas/transactions.py`, both editing screens, `receiptMath.ts`) + `tests/test_detail_round_trip.py` |
| `f845476` | **fixes** — storage keys, `check_price` size keys, observation metadata, capture temp dir, `splitSize`, migration `034b` |
| `a0dba52` | **account deletion + policies** — `account_service.py`, the route, Settings UI, `/account-deletion`, `/privacy`, `/terms`, page rendering, app links |

Nothing here is pushed to a remote yet. Verify with `.venv/bin/python -m pytest -q`
and `cd frontend && npx tsc --noEmit`.

---

## 1. What this project is

Personal finance app. Upload bank CSVs and photograph receipts; an LLM reads
them; chat with your spending. FastAPI + LangGraph backend, Expo/React Native
frontend, Supabase (auth, Postgres, pgvector, storage), deployed as a single
container on OCI at `rags2riches.duckdns.org` (HF Spaces is retired).

**Users bring their own LLM API key, and that is the shipping model** — not a
stopgap. See §2. It means the app has no marginal cost per user, which is what
makes a $25 launch possible at all.

---

## 2. Decisions already made — do not relitigate

| Decision | Detail |
|---|---|
| **Android first** | Play before App Store. `Claude.md` says Android; that is current. iOS config exists and is kept working, but Play is the target. |
| **Free app, BYO key** *(revised 29 Aug 2026 — supersedes freemium Option B)* | Users bring their own AI provider key; the app is free and takes no money. Rationale: it already works, it costs nothing per user, and **monetising is the single trigger** that would force company formation, publish a home address, and unlock all of §5. Not monetising keeps every one of those costs at zero. |
| **No company, for now** | Only needed once you monetise. A **free** app on a personal Play account publishes **name + country only** — the full home address appears only for merchant accounts (paid apps or IAP). Apple separately permits individuals to display a P.O. Box. Revisit this the day a paid tier is on the table, not before. |
| **Personal Play account is fine** | Costs $25 once, no company, no D-U-N-S. The price is the closed-testing rule: 12 testers opted in continuously for 14 days, which applies to personal accounts created after 13 Nov 2023. Organisation accounts appear to skip it — confirm with Play support if that two weeks ever becomes the blocker. |
| **No Plaid — CSV import stays** | Plaid bills Transactions as a *recurring monthly fee per bank login*, reported at $0.30–$3.00 per user per month, for as long as the connection lives. On a free app every user would be a permanent bill. CSV import is the moat, not the limitation: it also works in every country Plaid does not. |

---

## 3. What the previous session did

**Six bugs found and fixed**, all verified, all with regression tests (none had
one before — that is why they survived):

1. **Unauthenticated arbitrary file read** in the SPA fallback. `request.url.path`
   arrives percent-decoded, so `/%2e%2e/%2e%2e/proc/self/environ` escaped the
   static root and leaked `DATABASE_URL` and `APP_ENCRYPTION_KEY`. Fixed with
   `static_file_within()`.
2. **Line-item units silently wiped.** `item_quantity_unit`/`size_value`/`size_unit`
   were undeclared in the pydantic schemas, so they were stripped inbound *and*
   filtered out of responses. `TransactionDetail.size_value` was NULL on every
   receipt ever verified.
3. **Storage key collisions.** Two different files with the same name shared one
   object under `upsert`, overwriting each other. Now hash-prefixed.
4. **`check_price` stored no size** — used pre-migration-034 key names.
5. **Self-sighting guard was dead code** — `_is_the_same_sighting` read a
   `description` key the metadata never emitted.
6. **`splitSize` silently dropped multi-word units** — correcting a tag to
   `fl oz` looked accepted and changed nothing.

**Account deletion built and verified end-to-end against live data.** Deleted a
real test account: `auth.users`, `User`, `AccountConfig`, 1 CSV, 33 transactions,
33 embeddings and 1 storage object all went to zero; 3 other users, 461 other
transactions and 27 other storage objects untouched.

**Privacy policy and terms written and served** at `/privacy` and `/terms`, plus
`/account-deletion`. Linked from Settings and the login screen.

**Missing migration recovered**: `034b_match_functions_return_size.sql` was
applied to the database but never committed.

---

## 4. Outstanding — compliance track (finish this first)

Small, unblocked, and it gets a build in front of testers. With §5 deferred,
**this is the whole remaining path to launch.** Total cash cost: $25.

- [x] ~~**Commit** (§0).~~ Done — four commits, see §0.
- [x] ~~**Set three values.**~~ Done: `PUBLISHER_NAME=Sajil Awale` and
      `SUPPORT_EMAIL=r2rapp.support@gmail.com` in `.env` (gitignored — set these
      as HF Space secrets too), and governing law is the State of Connecticut,
      USA, in `backend/pages/terms.html`. All three pages verified rendering
      with no placeholders. Original note follows:
      ~~**Set three values.**~~ `PUBLISHER_NAME` and `SUPPORT_EMAIL` in `.env`;
      governing-law jurisdiction in `backend/pages/terms.html` (a loud red marker,
      deliberately not defaulted). Pages render `[not configured]` until then.
      Publishing as an individual, `PUBLISHER_NAME` is your own legal name — the
      policy templates interpolate a name and an email and **no physical
      address**, so this exposes nothing the store listing wouldn't.
- [x] ~~**`SUPABASE_SERVICE_KEY` on the deployed backend.**~~ Set in
      `~/rags_to_riches/.env` on the OCI box (`rags2riches.duckdns.org`) and
      confirmed present in the running container's environment. Production
      is no longer an HF Spaces repository secret setup — env vars live in
      that one `.env` file, loaded via `docker run --env-file`. The deploy
      mechanism itself (checkout location, build/restart commands, TLS,
      rollback) is now written down in `README.md` under Deployment; see
      `docs/DEPLOY_HANDOFF.md` for the redeploy this was part of.
- [ ] **Smoke-test the six fixes in the real app.** Only deletion was proven
      against reality. Upload a CSV, photograph and verify a receipt, then edit
      one line item and confirm the unit and size survive — that is bug #2, and
      it is the one worth seeing.
- [~] **`eas.json`** — written (`frontend/eas.json`, three profiles, documented
      in the README). Remaining: `eas init` (needs your Expo login, writes
      `extra.eas.projectId`), then
      `eas build --platform android --profile production` → Play internal
      testing.
- [ ] **Create the personal Play account** ($25, one-time) and **start
      recruiting the 12 testers immediately.** The closed test needs them opted
      in *continuously for 14 days*, so it is the longest pole in this list and
      it runs in parallel with everything above it. Start it first, finish it
      last.
- [ ] Data Safety form at submission. The disclosure that matters: transaction
      descriptions and receipt images go to a third-party AI provider **chosen
      and paid for by the user**, and DuckDuckGo receives merchant/item names
      when Deep Enrichment is on.

Housekeeping: the stale **Aug 25 build** that squatted on port 8000 has been
killed (it predated every fix, so smoke-testing against it would have proved
nothing). Port 8000 is still held by an unrelated `auto_etl` uvicorn running
since 22 Aug — leave it or kill it, but run this backend on 8011 either way.

---

## 5. Deferred — the Option B build

**Not being built.** §2 chose a free BYO-key app, which parks this entire section.
Kept because the research is sound and the seam analysis stays true.

**The trigger that revives it:** deciding to charge. On that day this section
comes back *together with* company formation, a new organisation Play account
(a personal account that has already published as a merchant has already
exposed the address), and the store-listing consequences in §2.

### The good news: the seam already exists

Every consumer of the AI credential reads `config["api_key"]`, and every one of
them gets `config` from **one function** — `config_service.get_config()`.
Verified call sites:

```
backend/services/ingestion_worker.py:43     backend/services/price_service.py:759
backend/services/rag_manager.py:80          backend/services/price_service.py:1188
backend/services/transaction_service.py:645 backend/services/transaction_service.py:729
money_rag.py:222 -> mcp_server.py:349 (via CURRENT_LLM_API_KEY env)
```

So injecting a platform key in `get_config()` reaches all of them with **no
downstream edits**. That is the whole reason this is tractable.

### Phase 1 — managed key

- Add a tier to `AccountConfig` (`tier: 'byo' | 'free' | 'pro'`) or a separate
  entitlement table. Migration + RLS.
- New settings: `PLATFORM_GOOGLE_API_KEY` / `PLATFORM_OPENAI_API_KEY`.
- In `get_config()`, when tier is managed, substitute the platform key.
- **`config_router._public_config` must never present a platform key as the
  user's** — `api_key_set`/`api_key_hint` currently describe whatever is in the
  config dict. Add a test.
- The Settings screen needs to stop demanding a key on managed tiers. Today three
  places hard-gate on it: `chat.py:108`, `file_service.py:200`,
  `capture_service.py:356`.

### Phase 2 — usage metering

Meter what actually costs money. It is uneven:

| Action | Cost shape |
|---|---|
| Receipt scan | one vision call — predictable |
| CSV upload | one LLM call per ~10 unique merchants + embeddings — scales with file size |
| **Chat turn** | **most variable** — 40-step recursion limit, spawns an MCP subprocess per turn |

Chat is where free-tier money leaks. Enforce **before** the work starts, at the
three gates listed above. Surface remaining quota in the UI.

### Phase 3 — rate limiting

There is **none** today. `MAX_CONCURRENT_CHATS_PER_USER = 1` caps one user; there
is no global cap. Once inference is on your card, an abusive user costs you money
directly.

### Phase 4 — IAP

Play Billing + StoreKit, or RevenueCat to avoid writing both. **Validate receipts
server-side** — never trust a client entitlement claim. Sync entitlement to the
tier from Phase 1.

### Phase 5 — before real traffic

`rag_manager._instances`, `file_service.ingestion_status` and
`capture_service._pending` are module-level dicts, and pending captures keep
bytes on the local disk. `rag_manager`'s own docstring says the design requires a
single-instance deploy. Paying users on a sleeping single-instance HF Space is
not viable. Move this state to Postgres.

---

## 6. Facts worth having

- Supabase project ref `hhydsgaqzkfrxwanvtgt`. Migrations 038–041 exist in the
  repo but are absent from the CLI ledger — they were applied out-of-band. The
  objects all exist; only the ledger is incomplete.
- `.env` needs: `SUPABASE_URL`, `SUPABASE_KEY` (anon), `DATABASE_URL`,
  `APP_ENCRYPTION_KEY`, `SUPABASE_SERVICE_KEY` (service_role), `SUPPORT_EMAIL`,
  `PUBLISHER_NAME`.
- The service key is reachable **only** through `dependencies.admin_client()` —
  `grep admin_client` finds every place RLS is bypassed. Keep it that way.
- `tests/test_policy_pages.py` holds the privacy policy to the code: it fails if
  `deep_enrichment` stops defaulting to off, if an analytics SDK appears in
  `package.json`, or if the capture route grows a lat/lon parameter.
- Frontend has **no test runner**. CI only typechecks. Two of the six bugs were
  frontend-side.
- Run the backend on a free port (8011 worked) — 8000 is contested.
