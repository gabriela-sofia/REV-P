from __future__ import annotations

import argparse

import susc_17c18_chirps_stac_common as s17c18


def main() -> int:
    parser = argparse.ArgumentParser(description="SUSC-17C18 estatistica zonal CHIRPS e consulta STAC Sentinel-2 por AOI")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("plan")
    sub.add_parser("query-chirps")
    sub.add_parser("compute-chirps-zonal")
    sub.add_parser("query-stac-sentinel2")
    sub.add_parser("build-offline")
    sub.add_parser("audit")
    args = parser.parse_args()

    if args.mode == "plan":
        print(s17c18.plan_text())
        return 0
    if args.mode == "query-chirps":
        code, text = s17c18.query_chirps_text()
        print(text)
        return code
    if args.mode == "compute-chirps-zonal":
        code, text = s17c18.compute_chirps_zonal_text()
        print(text)
        return code
    if args.mode == "query-stac-sentinel2":
        code, text = s17c18.query_stac_sentinel2_text()
        print(text)
        return code
    if args.mode == "build-offline":
        s17c18.build_all()
        print("build-offline: outputs 17C18 reconstruidos a partir de snapshots locais.")
        return 0
    if args.mode == "audit":
        return s17c18.validate()
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
