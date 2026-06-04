#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1ux_curitiba_common import parse_args, run_geodata_metadata_audit
except ModuleNotFoundError:
    from revp_v1ux_curitiba_common import parse_args, run_geodata_metadata_audit


if __name__ == "__main__":
    run_geodata_metadata_audit(parse_args())
