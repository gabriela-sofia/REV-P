# REV-P v1ul — Versioning Policy Classifier

## Purpose

Reads the v1uk artifact size inventory and assigns a final versioning policy
to each file. Produces a registry with staging eligibility and a summary.

## Policies

| Policy | Meaning |
|--------|---------|
| PUBLIC_VERSIONED | Small, clean file — OK to version |
| PUBLIC_SUMMARY_ONLY | Large CSV — summary allowed, full file local-only |
| LOCAL_ONLY_LARGE_DERIVED | local_runs / local_only — never version |
| RAW_EXTERNAL_NEVER_VERSION | data/external_raw — never version |
| CACHE_NEVER_VERSION | data/external_cache — never version |
| BLOCKED_GT_100MB | > 100MB — hard block |
| BLOCKED_GT_50MB | > 50MB — blocked for GitHub |
| REVIEW_REQUIRED | Requires manual review |

## Guardrails

Review-only. No operational labels, targets, ground truth or formal negatives
created. This script only reads the v1uk inventory and applies rule-based
classification.
