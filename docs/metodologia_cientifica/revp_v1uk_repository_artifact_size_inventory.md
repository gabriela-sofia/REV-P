# REV-P v1uk — Repository Artifact Size Inventory

## Purpose

Scans all files (tracked and untracked) in the repository and records their
sizes, extensions, directories, and preliminary size classifications. Files
larger than 50MB and 100MB are explicitly flagged.

## Outputs

- `datasets/repository_artifact_size_inventory_v1uk.csv` — one row per file
- `datasets/repository_artifact_size_summary_v1uk.csv` — aggregate metrics

## Size Classes

| Class | Threshold |
|-------|-----------|
| BLOCKED_GT_100MB | > 100 MB |
| BLOCKED_GT_50MB | > 50 MB |
| WARNING_GT_10MB | > 10 MB |
| OK | ≤ 10 MB |

## Guardrails

Review-only. No operational labels, targets, ground truth or formal negatives
are created by this script. It reads file metadata only; large files are never
read in full.

## Notes

- `.git/` and `__pycache__/` are excluded.
- Hashes are computed only over the first 64 KB to avoid loading large files.
