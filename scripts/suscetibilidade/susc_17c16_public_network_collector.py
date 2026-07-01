from __future__ import annotations

import argparse

import susc_17c16_public_network_common as s17c16


def main() -> int:
    parser = argparse.ArgumentParser(description="SUSC-17C16 coleta publica com opt-in de rede")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("capabilities")
    sub.add_parser("probe-public")
    sub.add_parser("collect-lightweight")
    sub.add_parser("probe-sensor-catalog")
    sub.add_parser("intake-collected")
    sub.add_parser("status")
    sub.add_parser("audit")
    args = parser.parse_args()

    if args.mode == "capabilities":
        print(s17c16.capabilities_text())
        return 0
    if args.mode == "probe-public":
        code, text = s17c16.probe_public_text()
        print(text)
        return code
    if args.mode == "collect-lightweight":
        code, text = s17c16.collect_lightweight_text()
        print(text)
        return code
    if args.mode == "probe-sensor-catalog":
        code, text = s17c16.probe_sensor_catalog_text()
        print(text)
        return code
    if args.mode == "intake-collected":
        print(s17c16.intake_collected_text())
        return 0
    if args.mode == "status":
        print(s17c16.status_text())
        return 0
    if args.mode == "audit":
        return s17c16.validate()
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
