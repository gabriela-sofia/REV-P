#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ac_common import parse_args, run_migration_diff_auditor
except ModuleNotFoundError:
    from revp_v2ac_common import parse_args, run_migration_diff_auditor


if __name__ == "__main__":
    run_migration_diff_auditor(parse_args())
