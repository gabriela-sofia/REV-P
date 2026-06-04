#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uu_recife_common import run_administrative_geometry_consolidator
except ModuleNotFoundError:
    from revp_v1uu_recife_common import run_administrative_geometry_consolidator


if __name__ == "__main__":
    run_administrative_geometry_consolidator()
