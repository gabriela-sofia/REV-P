#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uy_curitiba_common import parse_args, run_geodata_endpoint_probe
except ModuleNotFoundError:
    from revp_v1uy_curitiba_common import parse_args, run_geodata_endpoint_probe


if __name__ == "__main__":
    run_geodata_endpoint_probe(parse_args())
