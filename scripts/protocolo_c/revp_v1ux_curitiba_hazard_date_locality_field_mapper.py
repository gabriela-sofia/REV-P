#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1ux_curitiba_common import parse_args, run_hazard_date_locality_field_mapper
except ModuleNotFoundError:
    from revp_v1ux_curitiba_common import parse_args, run_hazard_date_locality_field_mapper


if __name__ == "__main__":
    run_hazard_date_locality_field_mapper(parse_args())
