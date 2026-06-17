# REV-P v2cp - external evidence manifest

This milestone creates a provenance, license and hash manifest for registered
or local external evidence. Local files receive SHA256 hashes when present.
Missing local files, unknown license and missing hash remain blocked.

The public manifest suppresses local raw-file paths when redistribution is not
allowed. Manifest status never means operational truth, labels, training data or
automatic detection.

Primary outputs:

- `datasets/external_evidence/external_evidence_manifest_v2cp.csv`
- `outputs_public/tables/revp_external_evidence_manifest_public_v2cp.csv`
- `outputs_public/logs_summary/revp_external_evidence_license_hash_rollup_v2cp.csv`

Execution:

```powershell
python scripts\multimodal\revp_v2cp_external_evidence_manifest.py --force
```
