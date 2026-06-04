#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1ut_recife_common import run_event_window_coordinate_filter
except ModuleNotFoundError:
    from revp_v1ut_recife_common import run_event_window_coordinate_filter


if __name__ == "__main__":
    run_event_window_coordinate_filter()
