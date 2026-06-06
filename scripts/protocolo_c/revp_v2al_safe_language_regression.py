#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2al_common import parse_args, run_safe_language_regression
except ModuleNotFoundError:
    from revp_v2al_common import parse_args, run_safe_language_regression


if __name__ == "__main__":
    run_safe_language_regression(parse_args())
