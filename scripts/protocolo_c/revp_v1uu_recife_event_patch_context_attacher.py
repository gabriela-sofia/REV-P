#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uu_recife_common import run_event_patch_context_attacher
except ModuleNotFoundError:
    from revp_v1uu_recife_common import run_event_patch_context_attacher


if __name__ == "__main__":
    run_event_patch_context_attacher()
