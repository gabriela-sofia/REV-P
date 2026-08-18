from __future__ import annotations

import argparse

import susc_17c23_fallback_policy_common as s17c23


def main() -> int:
    parser = argparse.ArgumentParser(description="SUSC-17C23 politica de equivalencia Earth Search e matriz cientifica review-only")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("evaluate-earth-search")
    sub.add_parser("evaluate-candidate-patch-policy")
    sub.add_parser("promote-review-only-science")
    sub.add_parser("build-candidate-matrix")
    sub.add_parser("build-offline")
    sub.add_parser("audit")
    args = parser.parse_args()

    if args.mode == "evaluate-earth-search":
        print(s17c23.evaluate_earth_search_text())
        return 0
    if args.mode == "evaluate-candidate-patch-policy":
        print(s17c23.evaluate_candidate_patch_policy_text())
        return 0
    if args.mode == "promote-review-only-science":
        print(s17c23.promote_review_only_science_text())
        return 0
    if args.mode == "build-candidate-matrix":
        print(s17c23.build_candidate_matrix_text())
        return 0
    if args.mode == "build-offline":
        s17c23.build_all()
        print("build-offline: outputs 17C23 reconstruidos a partir dos artefatos locais.")
        return 0
    if args.mode == "audit":
        return s17c23.validate()
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
