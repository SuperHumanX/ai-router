# GlobalApps.md — Ports & Services Registry

**Purpose:** single source of truth for every port in use across `~/Projects` (and `~/Downloads/Projects`), so a new service never silently collides with an existing one the way Bifrost collided with fastERP-v2's dev server on port 8090 (discovered 2026-08-09 — see "Known issues" below).

**Copy of this file lives in:** `~/Projects/GlobalApps.md` (canonical) and duplicated into `physician/`, `physician-web/`, `portfolio_tracker/`, `CatalogValidator/` (Downloads copy), `medresearch/`, `health-repo/`, `fastERP-v2/`, `ai-router/`, `bifrost-main/`. **Update the canonical copy first, then re-copy** — there's no symlink/sync automation, so these can drift; if you find a stale copy, that's expected until this gets automated.

**Ground truth used to build this**: `lsof -iTCP -sTCP:LISTEN`, `launchctl list` + each plist's `ProgramArguments`, `docker ps`, and each project's README/Operations/Architecture/Deployment docs — cross-checked against each other on 2026-08-09. Re-verify with the same commands before trusting this for a new service; ports drift.

---

## Port allocation map

| Port | App | Process | Managed by | Status |
|---|---|---|---|---|
| 3000 | **cardiomdpal** (physician-web) — Next.js web app | `npm run start -- -p 3000` | launchd `com.cardiomdpal.web` | ✅ active |
| 5432 | Postgres 16 (native, Homebrew) — `physicianweb` DB | `postgres` | launchd `homebrew.mxcl.postgresql@16` | ✅ active, shared |
| 5433 | Postgres 16 (Docker) — fastERP-v2's own DB | `fasterp-v2-postgres-1` container | Docker Compose (fastERP-v2) | ✅ active |
| 6333 | Qdrant (local Docker container) | `qdrant` container | Docker (manual/unknown compose) | ⚠️ **check before trusting** — health-repo/physician now use **Qdrant Cloud** (`QDRANT_URL` in `.env`), not this local instance. Likely a pre-cloud-migration leftover. See "Known issues". |
| 6379 | Redis | native `redis-server` (127.0.0.1 only) **and** `fasterp-v2-redis-1` (Docker, `*:6379`) | Homebrew `homebrew.mxcl.redis` **and** Docker Compose (fastERP-v2) | ⚠️ **two processes claiming the same port** — see "Known issues" |
| 6767 | Headroom (token-compression MCP tool) | `headroom-*` | Headroom's own launchd/user agent | ✅ active, not a project app |
| 8002 | **cardiomdpal MCP server** (Medical Knowledge Core) | `uvicorn mcp_server.main:app --port 8002` | launchd `com.cardiomdpal.mcpserver` | ✅ active |
| 8010 | Research Bridge (`research_api.py`), medresearchpro's **local** copy | `python3 scripts/research_api.py` (cwd `health-repo/`) | launchd `info.medresearchpro.bridge` | ⚠️ **confirmed vestigial** (see `physician_web_services` memory / this doc's "Known issues") — the real public medresearchpro.info runs on Azure. This local copy serves nothing in production. |
| 8011 | **cardiomdpal's dedicated Research Bridge** (`research_api.py`) | same script, separate process | launchd `com.cardiomdpal.researchbridge` | ✅ active — this is the one `physician-web` and `physician/mcp_server` actually talk to |
| 8020 | medresearchpro **local** API copy (`api/main.py`) | `bin/launchd_api.sh` → uvicorn | launchd `info.medresearchpro.api` | ⚠️ **running locally but purpose unclear** — production `/v1/*` API is on Azure (`api.medresearchpro.info`). Investigate whether this local copy is dev/test-only or another leftover before assuming it's needed. |
| 8080 | **portfolio_tracker** (radixboard.com) — Streamlit dashboard | `venv_new/bin/python3 web_dashboard.py` | launchd `com.portfoliotracker.web` | ✅ active — public via its own Cloudflare tunnel (`radixboard.com` → `localhost:8080`) |
| 8090 | ~~Bifrost gateway (intended)~~ → **actually fastERP-v2's dev server** | `tsx src/server.ts` (fastERP-v2) | fastERP-v2, manual/dev process | ⚠️ **collision** — Bifrost's `start.sh` blindly trusts "port already bound = Bifrost already running" and silently no-ops. Bifrost has been moved to **8095** (see below) to avoid touching fastERP-v2. Fix `start.sh`'s guard before reusing 8090 for anything. |
| 8095 | **Bifrost AI gateway** (local LLM router/cache) | `bifrost-http-0 -port 8095` | manual (`bifrost-main/start.sh --port 8095`), not launchd | ⚠️ active but **its own provider-key routing is broken** independent of the port move — see "Known issues" |
| 8123, 9000 | ClickHouse (Docker) — fastERP-v2 | `fasterp-v2-clickhouse-1` container | Docker Compose (fastERP-v2) | ✅ active |
| 8501–8504 | **CatalogValidator** — Streamlit dashboards (admin/customer/supplier + one more) | `streamlit run frontend/...` | manual/dev processes (not launchd) | ✅ active, dev-run |
| 9092, 9093 | Kafka (native, Homebrew) | `java ... kafka` | launchd `homebrew.mxcl.kafka` | ✅ active, shared (fastERP uses this) |
| 11434 | Ollama (default) | — | — | referenced by `ai-router`/CatalogValidator configs as a fallback local LLM; not seen bound right now |
| 11435 | **LocalIntelligence** — local SLM server (MLX/Ollama-compat) | `mlx_lm.server` or similar | manual, per LocalIntelligence README | referenced by `AI_ROUTER_LOCAL_BASE_URL` / `LOCAL_INTEL_URL` across physician, CatalogValidator, portfolio_tracker |
| 20241, 20242 | Cloudflare Tunnels | `cloudflared` | launchd `info.medresearchpro.tunnel` (cardiomdpal.com → :3000) and `com.portfoliotracker.cloudflaredtunnel` (radixboard.com → :8080) | ✅ active |

**Azure (not local — listed for completeness, see `medresearch/DEPLOYMENT.md`):** medresearchpro.info runs on `instance-health-1` — nginx :80/:443 → Next.js :3001, FastAPI :8020, bridge :8010. These port *numbers* coincidentally overlap with local ports above (8010, 8020) but are a **completely separate machine** — no actual collision, just a naming coincidence worth knowing about so you don't confuse local vs. Azure logs.

**Other launchd jobs with no listening port** (cron-style, one-shot, or client-only — not services): `com.cardiomdpal.encyclopedia-refresh`, `com.mac.cardiology.refresh`, `com.cardioapp.pipeline` (broken/dormant per prior investigation), `info.medresearchpro.harvest`, `info.medresearchpro.sync`, `com.portfoliotracker.{dailysnapshot,hourlysnapshot,newsflash,nightlyrestart,secfilings,weeklysummary}`, `com.portfoliotracker.bot` / `com.catalogvalidator.telegrambot` (Telegram bots, long-poll, no inbound port), `com.superhumanx.hybrid-daemon` (CatalogValidator).

---

## Known issues (found while building this doc, 2026-08-09)

1. **Bifrost / fastERP-v2 port collision (8090).** fastERP-v2's `tsx src/server.ts` dev server has been squatting on 8090 since at least Aug 7. Bifrost's `start.sh` checks `lsof -ti :$PORT` and treats *anything* listening as "Bifrost already running," so it silently never started. Every Gemini call routed through Bifrost has been failing for as long as this collision existed. **Fix applied:** Bifrost moved to port 8095 for now. **Still needed:** harden `start.sh`'s guard to verify it's actually talking to a Bifrost instance (e.g. hit `/health` and check the response shape) before treating a bound port as "already running."

2. **Bifrost's provider-key routing is broken independent of the port issue.** Even freshly started with a clean port, keys configured via `data/config.json` (google, anthropic, **and** a newly-added openrouter) all fail with `"no keys found that support model: ..."` at request time, despite the same keys showing `status: success` in Bifrost's own `config.db`. This affects *every* provider, not just the new one — reproducible via direct curl to `/v1/chat/completions`. Not yet root-caused; suspect a mismatch between the running binary version (`v1.5.0-prerelease3`, cached under `~/Library/Caches/bifrost/`) and how `config.json` → `config.db` sync wires keys into the live request router. **Current workaround:** `physician/gemini_helper.py`, `mcp_server/tools/live_harvest.py`, and physician-web's `/api/research/chat` all now call **OpenRouter directly** (bypassing Bifrost entirely), which works and was verified live. Bifrost's Claude/Ollama automatic-fallback and semantic-caching features are unused until this is root-caused.

3. **Redis port collision (6379).** Native Homebrew redis (`127.0.0.1:6379`) and fastERP-v2's Dockerized redis (`*:6379`) are both configured to listen there. They may be coexisting only because one is IPv4-loopback-only and the other is a wildcard bind across both families — this is fragile and depends on which one a given client's DNS/connect resolution picks. Worth deciding on one and removing the other.

4. **Local Qdrant Docker container (6333) may be dead weight.** health-repo and physician both point at **Qdrant Cloud** now (`QDRANT_URL` in `.env`, per the 2026-07-12 outage investigation where a stale `localhost:6333` reference caused a real production bug). If nothing intentionally still uses the local container, it's pure resource waste and a second source of "why isn't my search working" confusion like the one already documented in `project_physician_web_services` memory.

5. **`info.medresearchpro.api` and `info.medresearchpro.bridge` running locally, purpose unclear.** Per `medresearch/DEPLOYMENT.md` and prior investigation, the *public* medresearchpro.info runs entirely on Azure — "no production HTTP traffic touches the Mac." The bridge (8010) was already confirmed vestigial. The local API copy (8020, `info.medresearchpro.api`) hasn't been investigated the same way — worth the same treatment (confirm nothing references it, then retire) rather than leaving it running indefinitely "just in case."

6. ~~Two CatalogValidator directories.~~ **RESOLVED (confirmed 2026-09-24, not a new fix — the symlink already existed):** `~/Downloads/Projects` is a symlink to `~/Projects` (`Projects -> /Users/mac/Projects`). `~/Downloads/Projects/CatalogValidator` and `~/Projects/CatalogValidator` are the same physical directory — there was only ever one CatalogValidator. The original Aug 9 check (`ls -la` on the CatalogValidator dir itself) missed this because the symlink is one level up, at `~/Downloads/Projects`, not on `CatalogValidator` itself.

---

## Consolidation opportunities (not yet actioned — needs a decision, not a unilateral change)

These aren't fixed in this pass — flagging for a decision before touching anything, since several affect other live products or need confirmation about what's actually safe to retire:

- **Bifrost vs. direct-OpenRouter calls**: now that `gemini_helper.py` and friends call OpenRouter directly (bypassing Bifrost), is Bifrost worth keeping running at all until its routing bug is fixed? It's currently providing zero value.
- ~~Two "AI Router" implementations~~ **PARTIALLY RESOLVED (2026-09-23):** the three Python clients (`gateway.py`'s copies in physician, portfolio_tracker, CatalogValidator) had drifted into two different architectures — worse, one had features the other didn't and vice versa (see `ai-router/ARCHITECTURE.md`'s "Drift history" for the full story). Merged into one canonical `gateway.py` and re-synced to all three, plus fixed physician's stale `.env` overrides. The Python vs. TypeScript split (`gateway.py` vs. `router.ts`, used only by physician-web) is **still two implementations** — both independently converged on OpenRouter-as-shared-BYOK-fallback, but remain separate codebases with no shared source. Unifying those two is still open.
- **Redis**: pick native Homebrew or fastERP-v2's Dockerized instance, not both.
- **Local Qdrant container**: confirm unused, then retire.
- **`info.medresearchpro.{api,bridge}` local copies**: confirm unused (same treatment as the earlier bridge investigation), then retire both.

---

## Adding a new service — checklist

1. `lsof -iTCP -sTCP:LISTEN -P -n` — confirm the port you want is actually free (don't trust docs alone, per the Bifrost incident above).
2. Add a row to the table above **before** starting the service.
3. Re-copy this file to every location listed at the top.
4. If it's a persistent service, prefer launchd (`KeepAlive`) over a manually-backgrounded process — manual processes are exactly what caused the Bifrost collision (nothing supervised fastERP-v2's dev server or warned Bifrost's `start.sh` that the port was taken by something else).
