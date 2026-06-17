# REV-P v2cn - external evidence gap matrix

This milestone creates a regional matrix for Recife, Petropolis and Curitiba
that records missing observed geometry, CRS, provenance, hash, license, patch
boundary and replay readiness.

The matrix is an audit and planning artifact. It does not promote evidence, does
not create labels, and does not assert observed intersection. Critical
geospatial gaps are assigned when observed geometry, CRS or patch boundary is
missing. Provenance, hash and license gaps remain explicit blockers.

Primary output:

- `outputs_public/tables/revp_external_evidence_gap_matrix_v2cn.csv`

Execution:

```powershell
python scripts\multimodal\revp_v2cn_external_evidence_gap_matrix.py --force
```
