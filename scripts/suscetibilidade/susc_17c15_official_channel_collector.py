from __future__ import annotations

import argparse
from pathlib import Path

import susc_17c15_channel_collector_common as s17c15


def main() -> int:
    parser = argparse.ArgumentParser(description="SUSC-17C15 coletores oficiais controlados")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("capabilities")
    sub.add_parser("collect-public")
    sub.add_parser("probe")
    sub.add_parser("intake-collected")
    validate = sub.add_parser("validate-artifact")
    validate.add_argument("--artifact-path", required=True)
    sub.add_parser("status")
    sub.add_parser("audit")
    sub.add_parser("blocked-external")
    args = parser.parse_args()

    if args.mode == "capabilities":
        print(s17c15.capabilities_text())
        return 0
    if args.mode == "collect-public":
        code, text = s17c15.collect_public_text()
        print(text)
        return code
    if args.mode == "probe":
        code, text = s17c15.probe_text()
        print(text)
        return code
    if args.mode == "intake-collected":
        print(s17c15.intake_collected_text())
        return 0
    if args.mode == "validate-artifact":
        print(s17c15.validate_artifact(Path(args.artifact_path)))
        return 0
    if args.mode == "status":
        print(s17c15.capabilities_text())
        return 0
    if args.mode == "audit":
        return s17c15.validate()
    if args.mode == "blocked-external":
        for row in s17c15.build_blocked_external_action_registry():
            print(f"{row['formal_request_id']} {row['why_blocked']}")
        return 0
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
