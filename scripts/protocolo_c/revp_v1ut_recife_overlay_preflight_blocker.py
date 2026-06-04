#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1ut_recife_common import run_overlay_preflight_blocker
except ModuleNotFoundError:
    from revp_v1ut_recife_common import run_overlay_preflight_blocker


if __name__ == "__main__":
    run_overlay_preflight_blocker()
