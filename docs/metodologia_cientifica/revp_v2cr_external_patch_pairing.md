# REV-P v2cr - external patch pairing

This milestone pairs REV-P patches with external evidence only after QA marks an
external geometry as a validated candidate. Pairing blocks when patch boundary,
validated external geometry or CRS compatibility is missing.

Blocked rows leave intersection area and ratio fields empty. Executed pairing,
if later enabled with complete inputs, remains candidate-only and cannot create
labels, negatives, training data or operational validation.

Execution:

```powershell
python scripts\multimodal\revp_v2cr_external_patch_pairing.py --force
```
