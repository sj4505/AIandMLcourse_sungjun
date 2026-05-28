# VERIFY.md — budget-app Polar Webhook 401 Error
Session: 2026-04-29
State: ENV_CHECK_IN_PROGRESS

---

## Project Anchors

| Field | Value |
|-------|-------|
| Vercel project name | `ai_and_ml` |
| GitHub repo | `https://github.com/sj4505/AIandMLcourse_sungjun` (myrepo) |
| Current branch | `portfolio` |
| Latest commit | `0bbd58f` — feat: add vibe-coder-context-checkpoint skill to portfolio |
| Failure symptom | 401 from Polar webhook — `validateEvent` throws |

---

## Baseline
Working: unknown (new debug session)
Initial failure: 401 from `/api/webhook/polar` — signature validation fails

---

## Polar Webhook Checklist

### Check 1 — SDK Installed (Mode A) ✅ PASS
- Expected: `@polar-sh/sdk` in package.json
- Actual: `"@polar-sh/sdk": "^0.47.0"`
- Result: match
- If wrong: manual HMAC verification would be needed — confirmed failure pattern from prior session

### Check 2 — SDK Used in Handler (Mode A) ✅ PASS
- Expected: `validateEvent` from `@polar-sh/sdk/webhooks`
- Actual: `const { validateEvent } = require('@polar-sh/sdk/webhooks')` — used on line 32
- bodyParser disabled: `handler.config = { api: { bodyParser: false } }` ✅
- Raw body via Buffer: ✅
- Result: match
- If wrong: SDK call would fail or body would be pre-consumed

### Check 3 — Registered Webhook URL (Mode B) ⬜ PENDING
- Action: Polar dashboard → project → Webhooks
- Provide: exact registered webhook URL
- Expected: `https://[deployment-url]/api/webhook/polar`

### Check 4 — POLAR_WEBHOOK_SECRET Prefix (Mode B) ⬜ PENDING
- Action: Vercel dashboard → ai_and_ml → Settings → Environment Variables
- Provide: masked prefix of `POLAR_WEBHOOK_SECRET` (first 12 chars only)
- Expected prefix: `polar_whs_`

---

## Vercel Deployment Checklist (추가 확인 필요)

### Check — Production Branch (Mode B) ⬜ PENDING
- Action: vercel.com → ai_and_ml → Settings → Git → Production Branch
- Provide: current value of "Production Branch" field
- Expected: `portfolio` (current working branch)
- WHY: 이전 삽질에서 main 브랜치 불일치로 코드가 아예 반영 안 된 사례 있음

---

## Mismatches Found
(none yet — pending Mode B checks)
