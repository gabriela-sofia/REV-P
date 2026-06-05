#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ai_common import parse_args, run_safe_promotion_blockers
except ModuleNotFoundError:
    from revp_v2ai_common import parse_args, run_safe_promotion_blockers


if __name__ == "__main__":
    run_safe_promotion_blockers(parse_args())
