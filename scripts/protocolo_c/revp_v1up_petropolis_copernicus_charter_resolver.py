#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1up_petropolis_common import main_for
except ModuleNotFoundError:
    from revp_v1up_petropolis_common import main_for

if __name__ == "__main__":
    main_for("copernicus_charter_resolver")
