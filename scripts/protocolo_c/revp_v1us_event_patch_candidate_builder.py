#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1us_common import main_for
except ModuleNotFoundError:
    from revp_v1us_common import main_for

if __name__ == "__main__":
    main_for("event_patch_candidate_builder")
