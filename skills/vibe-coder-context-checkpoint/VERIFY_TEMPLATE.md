# VERIFY.md
session_start:
issue_summary:
state: UNVERIFIED

---

## Anchors
(fixed for entire session — all checks use these values only)

- vercel_project:
- deployment_url:
- github_repo:
- current_branch:
- rollback_commit:

---

## Baseline
(what was working before this session started)

- working_features:
  -
  -
- initial_failure_symptom:

---

## Verification Log

- `verified` = exact value obtained from a concrete source and compared explicitly
- `reported` = user described it without exact value or source-backed comparison

| item | expected | actual | result | mode | status |
|------|----------|--------|--------|------|--------|
| | | | match / mismatch | A (claude) / B (user) | verified / reported |

---

## Attempt Log
(do not repeat failed attempts without new evidence)

| # | what was tried | error or result | outcome |
|---|----------------|-----------------|---------|
| 1 | | | success / failure |
| 2 | | | success / failure |
| 3 | | | success / failure |

---

## Rollback
(fill this out BEFORE running /clear)

- rollback_target_commit:
- working_at_that_point:
  -
  -
- do_not_retry:
  -
  -
