#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2ab_common import parse_args, run_schema_migration_plan_builder
except ModuleNotFoundError:
    from revp_v2ab_common import parse_args, run_schema_migration_plan_builder


if __name__ == "__main__":
    run_schema_migration_plan_builder(parse_args())
