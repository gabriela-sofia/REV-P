#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2aa_common import parse_args, run_sentinel_sidecar_metadata_resolver
except ModuleNotFoundError:
    from revp_v2aa_common import parse_args, run_sentinel_sidecar_metadata_resolver


if __name__ == "__main__":
    run_sentinel_sidecar_metadata_resolver(parse_args())
