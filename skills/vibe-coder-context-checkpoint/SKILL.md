# SKILL: Vibe Coder External Service Checkpoint
version: 1.3
target: Claude Code
audience: Vibe coders (non-expert developers using AI-assisted coding)
scope: External service integration failures involving Vercel, Google OAuth, Polar, Supabase, and GitHub

---

## Purpose

Prevent context drift during external-service debugging by verifying the environment layer before modifying code.
Use this skill to establish project identity, verify exact configuration state, record all checks in `VERIFY.md`, and stop repeated blind code edits.

This skill also serves a secondary purpose: teaching the user what to check and why,
so they become less dependent on AI assistance over time.

---

## Execution Model

This skill uses two verification modes. Apply the correct mode to each check.

### Mode A — Claude Executes Directly
Claude runs terminal commands using its bash tool and reads the output directly.
No user action required. Claude records results in VERIFY.md immediately.

```
Applicable commands:
  git branch --show-current
  git log --oneline -1
  git remote -v
  git status --short
  vercel ls --prod [project-name]
  vercel env ls
  cat package.json | grep [package]
  ls, cat, grep on local project files
```

When using Mode A:
- Run the command silently and record the result.
- Show the result to the user with a one-line explanation of what it means.
- State the comparison result explicitly.
- Do not ask the user to run commands that Claude can run directly.

### Mode B — Screenshot via VS Code Claude Code
Claude cannot access external dashboards directly via terminal.
For Mode B checks, instruct the user to switch to VS Code Claude Code and attach a screenshot.
VS Code Claude Code reads the screenshot, extracts values, and writes results to VERIFY.md directly.
Apply Mode B only when the required information is unavailable via terminal.

```
Applicable checks:
  Vercel Production Branch setting
  Google OAuth authorized redirect URIs
  Polar registered webhook URL
  Environment variable presence (masked values visible in dashboard)
  GitHub OAuth app settings
```

When using Mode B:
- Instruct the user to switch to VS Code Claude Code window.
- Provide the exact URL path to the correct dashboard setting.
- Ask the user to take a screenshot (Windows: Win + Shift + S) and attach it in VS Code Claude Code.
- VS Code Claude Code reads the screenshot directly and writes verified values to VERIFY.md.
- Never ask the user to manually type or copy-paste dashboard values — screenshot is more reliable.
- After screenshot is analyzed, VERIFY.md is updated and result is stated explicitly.

---

## Activation Logic

Activate this skill only when **external service involvement is present** AND at least one trigger condition is true.

### External Service Involvement Required
At least one of the following must be true:

- The issue involves Vercel, Google OAuth, Polar, Supabase, or GitHub
- The issue involves deployment, auth, webhook, database connection, environment variables, or repository linkage
- The user mentions an external dashboard, CLI sync issue, production mismatch, or callback/redirect problem

### Trigger Conditions
Activate immediately if any of the following is true:

- HTTP error appears: `401`, `403`, `404`, `500`
- User reports: "not reflecting", "deployed but nothing changed", "why isn't it working", "I definitely did this"
- Same error or symptom appears 2+ times
- Immediately before or after executing any Vercel CLI command

### Do Not Activate If
Do not activate for clearly local-only problems:

- syntax errors, type errors, formatting/lint issues
- isolated UI bugs with no external dependency
- obvious non-integration logic bugs

If uncertain, ask one short classification question before activating.

---

## Operating States

- `UNVERIFIED` — environment not yet checked
- `ANCHORS_SET` — project identity fixed
- `BASELINE_RECORDED` — current working baseline captured
- `ENV_CHECK_IN_PROGRESS` — service checklist running
- `ENV_VERIFIED` — all required checks passed
- `ENV_MISMATCH_FOUND` — one or more checks failed
- `CODE_CHANGES_ALLOWED` — code modification now permitted
- `REVERIFY_REQUIRED` — prior verification invalidated by CLI execution or state drift
- `ROLLBACK_RECOMMENDED` — three consecutive failures reached

Do not modify code unless state is `CODE_CHANGES_ALLOWED`.

---

## Hard Rules

```
RULE 1: Do not modify integration-related code before required environment verification is complete.
        Exception: trivial non-integration fixes that do not affect auth, deployment,
        env vars, webhooks, routing, or external service behavior.

RULE 2: Do not use yes/no questions for verification.
        Require exact values for non-secret fields.
        For secrets, use safe verification format only.

RULE 3: Always explain WHY before requesting a verification step. Limit WHY to one line.

RULE 4: Once a value is verified and recorded in VERIFY.md, do not request it again
        unless: Vercel CLI ran / branch changed / deployment target changed /
        new session started / explicit evidence of drift.

RULE 5: After 2 identical failures, stop code modification and re-run the relevant checklist.

RULE 6: After 3 consecutive failed attempts, initiate rollback protocol.

RULE 7: Use the project's active integrated terminal by default.
        On Windows, prefer VS Code integrated terminal over standalone shells
        if CLI/path issues are suspected.

RULE 8: Never proceed on assumed matches.
        Show compared values explicitly and state: match / mismatch.

RULE 9: Record all verification results in VERIFY.md before moving to the next stage.

RULE 10: Never request or store full secret values.
         Use masked, presence-based, prefix-based, or format-based verification only.

RULE 11: Run all terminal-executable checks directly using the bash tool.
         Do not ask the user to run commands that Claude can execute itself.
```

---

## Safe Secret Handling

### Never Request
- full API keys, client secrets, webhook secrets, service role keys, auth secrets, tokens

### Allowed Verification Formats
- presence / absence
- masked prefix (first 8–12 characters only)
- expected prefix pattern (e.g. `polar_whs_`)
- key length
- correct environment scope
- correct server/client usage location

### Examples
- `NEXTAUTH_SECRET` → confirm presence only
- `SUPABASE_SERVICE_ROLE_KEY` → confirm presence + server-only usage
- `POLAR_WEBHOOK_SECRET` → verify prefix pattern only (`polar_whs_`)
- `GOOGLE_CLIENT_SECRET` → confirm presence only

---

## User Communication Rules

### Format for Mode A (Claude executes)

```
[Claude runs command silently]
Result: [output]
WHY this matters: [one line explanation]
Compared:
- Expected: [value]
- Actual: [value]
Result: match / mismatch
If this had been wrong: [specific consequence in one line]
```

### Format for Mode B (VS Code Claude Code screenshot)

Terminal Claude Code instructs:
```
WHY: [one line — what breaks if this is wrong]
ACTION: Switch to VS Code Claude Code →
        Open [exact URL path] →
        Screenshot (Win + Shift + S) → attach in VS Code Claude Code
```

VS Code Claude Code receives screenshot, then:
```
[Reads screenshot directly]
Extracted value: [value read from screenshot]
WHY this matters: [one line explanation]
Compared:
- Expected: [value]
- Actual: [value read from image]
Result: match / mismatch
If this had been wrong: [specific consequence in one line]
[Writes result to VERIFY.md immediately]
```

### General Communication Rules

```
- For Mode A: run first, then explain. Never ask the user to do what Claude can do.
- For Mode B: always use VS Code Claude Code screenshot method. Never ask user to type dashboard values manually.
- Never ask "is it X?" or "did you check?"
- State all results explicitly: match / mismatch / present / missing / correct / incorrect.
- Announce completion: "✅ [item] confirmed. Moving on."
- Do not re-request verified items unless re-verification is triggered.
```

### Learning Reinforcement Rules

These rules exist to reduce the user's dependence on AI over time.
Apply them consistently, but keep each explanation to one short line.

```
- After each verified item, add one line explaining what would have broken if it was wrong.
  Format: "If this had been wrong: [specific consequence]."

- When a mismatch is found, explain the root cause in plain language before fixing it.
  Do not fix silently. The user must understand what happened.

- When the same check appears in multiple sessions, note it:
  "This is worth remembering — [item] should always be checked first when [symptom] appears."

- When rollback is triggered, name the failure pattern:
  "This is a [pattern name] failure. Next time, check [item] first."

- Never perform verification silently. Every check must be visible to the user
  with its result and WHY stated. This is how they learn what to look for independently.
```

---

## Session Start Protocol

### Step 1 — Fix Project Anchors
Collect and record in `VERIFY.md` header. All later checks use these values only.

```
Ask the user:
- Vercel project name (shown in Vercel dashboard top-left)
- Deployment URL (e.g. xxx.vercel.app)
- GitHub repo name
- Current working branch
```

### Step 2 — Record Baseline

```
WHY: This becomes the rollback checkpoint if things break.
MODE A: Claude executes → git log --oneline -1
Record: commit hash + currently working features + initial failure symptom + session start time
```

### Step 3 — Determine Relevant Checklist
Run only the checklist(s) relevant to the current issue.
Do not run unrelated checklists unless the same symptom persists or multiple services are clearly involved.

---

## VERIFY.md

Use `VERIFY_TEMPLATE.md` as the base. Copy it to the project root as `VERIFY.md` at session start.

Key rules:
- `verified` = exact value was obtained from a concrete source and compared explicitly
- `reported` = user described it without exact value or without source-backed comparison
- Never store full secret values
- Every mismatch must include an impact statement

---

## Core Workflow

```
Entry condition: integration-related error or symptom detected

Step 1. Declare: "Before touching code, let's verify the environment first."
Step 2. Create VERIFY.md from VERIFY_TEMPLATE.md.
Step 3. Set project anchors → state: ANCHORS_SET
Step 4. Record baseline (Mode A) → state: BASELINE_RECORDED
Step 5. Run only the relevant service checklist → state: ENV_CHECK_IN_PROGRESS
        Use Mode A for all terminal-executable checks.
        Use Mode B only for dashboard-only checks.
Step 6. Record all results in VERIFY.md.
Step 7. Evaluate:
        - all required checks pass → state: ENV_VERIFIED → CODE_CHANGES_ALLOWED
        - any required check fails → state: ENV_MISMATCH_FOUND

Exit condition: all required checklist items verified and recorded.
Only then: integration-related code changes may begin.
```

---

## Re-Verification Triggers

Set state to `REVERIFY_REQUIRED` and re-run relevant checks if any of the following occurs:

```
- vercel / vercel link / vercel deploy / vercel env pull executed
- branch changed
- deployment target changed
- environment variables changed
- new session begins
- observed state contradicts recorded state
```

When re-verifying: use Mode A first, then Mode B if needed.
Compare against previous VERIFY.md values. Record what changed.

---

## Repeated Failure Protocol

### Trigger
Same error, symptom, or mismatch appears a second time.

### Action

```
Declare:
"Same error again. Continuing to modify code will pollute the context.
Let's stop and re-verify the environment layer."

Then:
1. Stop code modification.
2. Re-run the relevant checklist (Mode A first, then Mode B).
3. Compare new results against VERIFY.md.
4. Record what changed or remained unchanged.
5. Explain to the user why this check matters for this specific symptom.
```

If nothing changed between checks: narrow to code logic, or expand to next dependent service.

---

## Rollback Protocol

### Trigger
3 consecutive failed attempts on the same task.

### Action

```
Step 1 — Summarize attempts from VERIFY.md
  "Attempt 1: [action] → failed ([reason])"
  "Attempt 2: [action] → failed ([reason])"
  "Attempt 3: [action] → failed ([reason])"

Step 2 — Identify and name the failure pattern
  "This is a [pattern] failure. The root cause is [plain language explanation]."
  "Next time [symptom] appears, check [item] first."

Step 3 — Propose rollback
  "3 consecutive failures reached.
   Failed attempts are accumulating and causing context drift.
   Rolling back to the last known good checkpoint is faster than continuing.
   Should we /clear and restart from the checkpoint?"

Step 4 — Record before reset in VERIFY.md rollback section:
  - last known good commit
  - what was working
  - failed attempts and reasons
  - what must not be retried unchanged

Step 5 — New session instruction:
  Begin with: "Read VERIFY.md first."
  Do not repeat listed failed attempts without new evidence.
  Re-run relevant environment verification before modifying code.
```

---

## Service Checklists

Each item specifies execution mode explicitly.
Mode A = Claude executes directly. Mode B = user confirms from dashboard.

---

### Vercel Deployment Errors

**Entry condition:** deployment not reflecting, `404`, old production code still serving

#### Check 1 — CRITICAL — Mode A
- **Check:** Current branch
- **Why:** Pushing from the wrong branch means changes never deploy.
- **Execute:** `git branch --show-current`
- **Pass:** branch name recorded

#### Check 2 — CRITICAL — Mode B
- **Check:** Vercel Production Branch
- **Why:** Vercel CLI may silently reset this to `main`.
- **Action:** vercel.com → [project] → Settings → Git → Production Branch
- **Provide:** exact current value of "Production Branch" field
- **Pass:** matches current working branch from Check 1

#### Check 3 — CRITICAL — Mode A
- **Check:** Production commit vs local commit
- **Why:** "Ready Stale" means this commit is not currently live.
- **Execute:**
  - `vercel ls --prod [project-name]` → production commit hash
  - `git log --oneline -1` → local latest commit hash
- **Pass:** first 7 characters match

#### Check 4 — REQUIRED — Mode A
- **Check:** Required environment variables exist in correct scope
- **Why:** Missing variables break auth, routing, or payments even after successful deployment.
- **Execute:** `vercel env ls`
- **Pass:** all required variable names present in correct environment scope

#### Check 5 — CONDITIONAL — Mode A
- **Check:** Windows hostname compatibility
- **Why:** Non-ASCII computer names can break Vercel CLI HTTP requests.
- **Execute:** check system hostname if CLI shows HTTP header errors
- **Pass:** hostname contains only ASCII characters, or issue not hostname-related

---

### Google OAuth Errors

**Entry condition:** OAuth redirect error, login failure, callback `400` / `401`

#### Check 1 — CRITICAL — Mode B
- **Check:** Authorized redirect URIs
- **Why:** Google rejects auth if the callback URI doesn't match exactly — even a trailing slash matters.
- **Action:** Google Cloud Console → OAuth 2.0 Client → Authorized redirect URIs
- **Provide:** copy the full list of currently registered redirect URIs
- **Pass:** exact callback URI for current environment is present

#### Check 2 — CRITICAL — Mode B
- **Check:** Deployment URL callback coverage
- **Why:** Local and production callback URIs must be registered separately.
- **Action:** from the list provided in Check 1
- **Provide:** copy the exact matching entry for `https://[deployment-url]/api/auth/callback/google`
- **Pass:** exact expected callback URI is present in the list

#### Check 3 — REQUIRED — Mode A
- **Check:** Required environment variables
- **Why:** Any single missing variable breaks the entire OAuth flow.
- **Execute:** `vercel env ls`
- **Pass:** all present in correct scope:
  `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `NEXTAUTH_SECRET`, `NEXTAUTH_URL`

#### Check 4 — REQUIRED — Mode A
- **Check:** `NEXTAUTH_URL` value alignment
- **Why:** Auth callbacks fail if the configured app URL doesn't match the deployment environment.
- **Execute:** `vercel env ls` + inspect configured value via safe path
- **Pass:** matches intended deployment URL anchor

---

### Polar Payment / Webhook Errors

**Entry condition:** webhook signature failure, payment events missing

#### Check 1 — CRITICAL — Mode A
- **Check:** Official SDK installed
- **Why:** Manual HMAC verification is error-prone and has caused confirmed failures in real cases.
- **Execute:** `cat package.json | grep polar`
- **Pass:** `@polar-sh/sdk` present
- **If missing:** install SDK before any code review

#### Check 2 — REQUIRED — Mode A
- **Check:** SDK actually used in webhook handler + no manual HMAC implementation
- **Why:** Package installed does not mean it is being used. Manual HMAC implementation diverges from Polar internal logic even when SDK is installed. (confirmed: 7 failed attempts in real case before switching to SDK)
- **Execute:**
  - grep -r validateEvent . → confirm SDK usage
  - grep -r createHmac . → detect manual implementation
- **Pass:** validateEvent found AND createHmac NOT found in webhook handler
- **If createHmac detected:** Declare: Manual HMAC implementation detected. This is the most common cause of Polar webhook 401 errors. Replace with official SDK before any other fix.
  Remove: crypto.createHmac
  Replace with: const { validateEvent } = require(@polar-sh/sdk/webhooks) then event = validateEvent(rawBody, req.headers, process.env.POLAR_WEBHOOK_SECRET)

#### Check 3 — CRITICAL — Mode B
- **Check:** Registered webhook URL
- **Why:** Wrong destination means events never arrive.
- **Action:** Polar dashboard → project → Webhooks
- **Provide:** exact registered webhook URL
- **Pass:** matches intended deployment webhook endpoint

#### Check 4 — REQUIRED — Mode B
- **Check:** Webhook secret prefix
- **Why:** Wrong prefix format causes verification failure. Confirmed failure case.
- **Action:** Vercel dashboard → Environment Variables → `POLAR_WEBHOOK_SECRET`
- **Provide:** masked prefix only
- **Pass:** prefix matches `polar_whs_`

---

### Supabase Connection Errors

**Entry condition:** DB connection failure, unauthorized access, RLS failure

#### Check 1 — CRITICAL — Mode A
- **Check:** Required environment variables
- **Why:** Without correct URL and keys, no connection can be established.
- **Execute:** `vercel env ls`
- **Pass:** both present in correct scope:
  `NEXT_PUBLIC_SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`

#### Check 2 — CRITICAL — Mode A
- **Check:** Key type usage location
- **Why:** Service-role key in client-exposed code is a security incident.
- **Execute:** search codebase for Supabase client initialization and env var usage
- **Pass:** service-role key is server-only; client code uses only public-safe values

#### Check 3 — REQUIRED — Mode A
- **Check:** Supabase URL alignment
- **Why:** Wrong project URL sends requests to the wrong backend.
- **Execute:** read `NEXT_PUBLIC_SUPABASE_URL` from env config
- **Pass:** matches intended Supabase project host

---

### GitHub Connection Errors

**Entry condition:** push failure, wrong repo, changes not reflecting after push

#### Check 1 — CRITICAL — Mode A
- **Check:** Remote URL
- **Why:** Changes may be pushed to the wrong repository.
- **Execute:** `git remote -v`
- **Pass:** matches project anchor repo

#### Check 2 — CRITICAL — Mode A
- **Check:** Current branch
- **Why:** Changes may be committed or pushed on the wrong branch.
- **Execute:** `git branch --show-current`
- **Pass:** matches intended working branch

#### Check 3 — REQUIRED — Mode A
- **Check:** Working tree state
- **Why:** Uncommitted changes are not included in pushes.
- **Execute:** `git status --short`
- **Pass:** working tree state understood and recorded

---

## Comparison Rule

Always use this exact format when comparing two values:

```
Compared:
- Expected: [value]
- Actual: [value]
Result: match / mismatch
If this had been wrong: [specific consequence in one line]
```

Do not imply matches indirectly. State them explicitly.
Always include the consequence line — this is how the user learns what each check protects against.

---

## Skill Exit States

- `CODE_CHANGES_ALLOWED` — all required checks passed
- `ENV_MISMATCH_FOUND` — at least one required check failed
- `REVERIFY_REQUIRED` — environment drift invalidated prior verification
- `ROLLBACK_RECOMMENDED` — repeated attempts exceeded threshold

If exit state is not `CODE_CHANGES_ALLOWED`, do not resume integration-related code changes.

---

## Never Do

```
NEVER modify integration-related code before environment verification completes
NEVER ask yes/no verification questions
NEVER request or store full secret values
NEVER request the same verified item twice unless re-verification is triggered
NEVER ask the user to run a command that Claude can execute directly (Mode A)
NEVER continue blind code edits after the same failure repeats twice
NEVER continue after 3 consecutive failures without rollback protocol
NEVER assume deployment, branch, env, or callback values match — always verify explicitly
NEVER skip Production Branch re-check after Vercel CLI execution
NEVER implement provider-secret verification manually when official SDK is available
NEVER perform verification silently — every check must be visible with its result and WHY
NEVER fix a mismatch without explaining the root cause to the user first
```

---

## Integration Boundary

This skill runs **before** code-layer debugging workflows.

- **This skill** — determines whether the environment/configuration state is valid
- **Code debugging** — determines how application logic should change

Do not merge the two stages. Verify the environment first, then debug code if still necessary.
