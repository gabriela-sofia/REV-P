#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uv_curitiba_common import parse_args, run_defesa_civil_pr_resolver
except ModuleNotFoundError:
    from revp_v1uv_curitiba_common import parse_args, run_defesa_civil_pr_resolver


if __name__ == "__main__":
    run_defesa_civil_pr_resolver(parse_args())
