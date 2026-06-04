#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uq_petropolis_common import main_for
except ModuleNotFoundError:
    from revp_v1uq_petropolis_common import main_for

if __name__ == "__main__":
    main_for("phenomenon_separation_decision_matrix")
