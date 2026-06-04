#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ab_common import parse_args, run_package_completeness_scorer
except ModuleNotFoundError:
    from revp_v2ab_common import parse_args, run_package_completeness_scorer


if __name__ == "__main__":
    run_package_completeness_scorer(parse_args())
