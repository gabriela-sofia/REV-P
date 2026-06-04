#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1ux_curitiba_common import parse_args, run_schema_audit
except ModuleNotFoundError:
    from revp_v1ux_curitiba_common import parse_args, run_schema_audit


if __name__ == "__main__":
    run_schema_audit(parse_args())
