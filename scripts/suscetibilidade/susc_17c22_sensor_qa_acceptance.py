from __future__ import annotations

import argparse

import susc_17c22_sensor_qa_acceptance_common as s17c22


def main() -> int:
    parser = argparse.ArgumentParser(description="SUSC-17C22 aceite formal automatizado do QA sensorial e feature matrix candidata")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("evaluate-qa")
    sub.add_parser("build-feature-matrix")
    sub.add_parser("trace-provenance")
    sub.add_parser("audit-blockers")
    sub.add_parser("build-offline")
    sub.add_parser("audit")
    args = parser.parse_args()

    if args.mode == "evaluate-qa":
        print(s17c22.evaluate_qa_text())
        return 0
    if args.mode == "build-feature-matrix":
        print(s17c22.build_feature_matrix_text())
        return 0
    if args.mode == "trace-provenance":
        print(s17c22.trace_provenance_text())
        return 0
    if args.mode == "audit-blockers":
        print(s17c22.audit_blockers_text())
        return 0
    if args.mode == "build-offline":
        s17c22.build_all()
        print("build-offline: outputs 17C22 reconstruidos a partir dos artefatos locais.")
        return 0
    if args.mode == "audit":
        return s17c22.validate()
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
