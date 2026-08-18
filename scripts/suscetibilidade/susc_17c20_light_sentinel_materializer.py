from __future__ import annotations

import argparse

import susc_17c20_light_sentinel_common as s17c20


def main() -> int:
    parser = argparse.ArgumentParser(description="SUSC-17C20 materializacao real de recorte leve Sentinel-2 por AOI")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("plan")
    sub.add_parser("resolve-assets")
    sub.add_parser("materialize-light")
    sub.add_parser("compute-indices")
    sub.add_parser("build-offline")
    sub.add_parser("audit")
    args = parser.parse_args()

    if args.mode == "plan":
        print(s17c20.plan_text())
        return 0
    if args.mode == "resolve-assets":
        code, text = s17c20.resolve_assets_text()
        print(text)
        return code
    if args.mode == "materialize-light":
        code, text = s17c20.materialize_light_text()
        print(text)
        return code
    if args.mode == "compute-indices":
        print(s17c20.compute_indices_text())
        return 0
    if args.mode == "build-offline":
        s17c20.build_all()
        print("build-offline: outputs 17C20 reconstruidos a partir dos artefatos leves locais.")
        return 0
    if args.mode == "audit":
        return s17c20.validate()
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
