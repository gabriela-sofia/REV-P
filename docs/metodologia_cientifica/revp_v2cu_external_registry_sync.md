# REV-P v2cu - external registry sync

This milestone synchronizes v2cs/v2ct metadata into a new controlled
acquisition registry at `datasets/external_evidence/sources_registry_v2cu.csv`.

It does not overwrite `sources_registry_v2co.csv`. If license or redistribution
is not confirmed, `download_allowed=false` and `public_repo_allowed=false`.
Source-family rows do not become direct download URLs.
