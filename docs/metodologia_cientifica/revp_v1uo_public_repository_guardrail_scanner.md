# REV-P v1uo — Public Repository Guardrail Scanner

## Purpose

Scans files in the public staging candidate list for content violations that
would compromise the public repository's integrity.

## Checks

| Check | Description |
|-------|-------------|
| abs_path | Absolute Windows/Unix paths found in file content |
| local_runs | References to local_runs/ directory |
| raw_cache | File is in raw/cache directory |
| forbidden_term | Forbidden operational terminology |
| forbidden_flag | Guardrail flags set to true (e.g. can_train_model,true) |
| large_file | File exceeds 50MB |
| no_header | CSV without header row |

## Claudete false-positive protection

The scanner uses word-boundary regex to detect "Claude" without matching
"Claudete" or "Claude Code".

## Expected result

For a clean repository, `guardrail_violations_in_safe_to_stage = 0`.

## Guardrails

Review-only. No operational labels, targets, ground truth or formal negatives
are created.
