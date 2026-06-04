#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2af_common import parse_args, run_artifact_freshness_auditor
except ModuleNotFoundError:
    from revp_v2af_common import parse_args, run_artifact_freshness_auditor


if __name__ == "__main__":
    run_artifact_freshness_auditor(parse_args())
