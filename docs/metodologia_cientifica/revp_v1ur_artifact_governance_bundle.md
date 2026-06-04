# REV-P v1ur — Artifact Governance Bundle

## Final Status

**ARTIFACT_GOVERNANCE_READY_WITH_LOCAL_ONLY_FILES**

## Summary Metrics

| Metric | Value |
|--------|-------|
| Files scanned | 3900 |
| Tracked files | 1820 |
| Untracked files | 2080 |
| Files > 10MB | 48 |
| Files > 50MB | 9 |
| Files > 100MB | 2 |
| Blocked large files | 1322 |
| Public summaries created | 10 |
| Safe-to-stage files | 2578 |
| Guardrail violations | 294 |
| Raw/cache files blocked | 3 |

## Final Status Criteria

| Status | Condition |
|--------|-----------|
| `ARTIFACT_GOVERNANCE_READY` | Zero violations, no large local files |
| `ARTIFACT_GOVERNANCE_READY_WITH_LOCAL_ONLY_FILES` | Large files present but blocked and have public summaries |
| `ARTIFACT_GOVERNANCE_FAIL_CLOSED` | Guardrail violations exist or large files have no summaries |

## Stages

- **v1uk** — Repository artifact size inventory
- **v1ul** — Versioning policy classifier
- **v1um** — Large CSV public summary generator
- **v1un** — Public staging candidate manifest
- **v1uo** — Public repository guardrail scanner
- **v1up** — PowerShell precommit gate generator
- **v1uq** — Gitignore and repository policy updater
- **v1ur** — This bundle (orchestration + summary)

## Guardrails

Review-only. No operational labels, targets, ground truth, formal negatives,
DINO validation claims, or training flags are created by any stage in this block.
External observational evidence remains required for any operational claim.
