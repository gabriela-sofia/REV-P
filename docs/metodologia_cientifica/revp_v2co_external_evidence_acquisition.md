# REV-P v2co - controlled external evidence acquisition

This milestone registers external evidence sources and supports controlled
downloads only when `--allow-downloads` is provided. The default mode is
offline and never accesses the network.

`datasets/external_evidence/sources_registry_v2co.csv` is the controlled source
registry and download allowlist. A row can be downloaded only when the source
family is permitted, the URL is present, the license is known,
`download_allowed=true`, and public redistribution is allowed. Raw files, when
allowed, are saved only under `datasets/external_evidence/raw/`.

No raw external file is written to `outputs_public`.

Execution:

```powershell
python scripts\multimodal\revp_v2co_external_evidence_acquisition.py --offline --force
python scripts\multimodal\revp_v2co_external_evidence_acquisition.py --allow-downloads --force
```
