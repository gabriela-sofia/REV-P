#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uw_curitiba_common import parse_args, run_event_date_hazard_audit
except ModuleNotFoundError:
    from revp_v1uw_curitiba_common import parse_args, run_event_date_hazard_audit


if __name__ == "__main__":
    run_event_date_hazard_audit(parse_args())
