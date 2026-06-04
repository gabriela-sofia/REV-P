#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ag_common import parse_args, run_unlinkable_date_guard_updater
except ModuleNotFoundError:
    from revp_v2ag_common import parse_args, run_unlinkable_date_guard_updater


if __name__ == "__main__":
    run_unlinkable_date_guard_updater(parse_args())
