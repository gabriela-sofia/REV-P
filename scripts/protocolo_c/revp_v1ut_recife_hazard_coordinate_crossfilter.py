#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1ut_recife_common import run_hazard_coordinate_crossfilter
except ModuleNotFoundError:
    from revp_v1ut_recife_common import run_hazard_coordinate_crossfilter


if __name__ == "__main__":
    run_hazard_coordinate_crossfilter()
