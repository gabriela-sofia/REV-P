#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1ut_recife_common import run_geojson_context_classifier
except ModuleNotFoundError:
    from revp_v1ut_recife_common import run_geojson_context_classifier


if __name__ == "__main__":
    run_geojson_context_classifier()
