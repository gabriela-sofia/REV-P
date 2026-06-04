# REV-P v1um — Large CSV Public Summary Generator

## Purpose

For each CSV blocked from versioning (>50MB or classified as local-only), generates
a small public summary that can be versioned safely. The full CSV is never copied.

## What summaries contain

- Row count (by streaming — file never fully loaded into memory)
- Column list
- Status-column value counts (from sample)
- Up to 50 sample rows
- SHA-256 of the first 64 KB (provenance anchor)
- Guardrail block

## Known large files processed

- `datasets/formal_negative_evidence_intake_registry.csv` (≈161 MB)
- `datasets/formal_negative_candidate_decision_audit.csv` (≈77 MB)

Both are git-ignored and remain local-only. Their summaries are versioned.

## Guardrails

Review-only. No operational labels, targets, ground truth or formal negatives
created or summarised into the public outputs.
