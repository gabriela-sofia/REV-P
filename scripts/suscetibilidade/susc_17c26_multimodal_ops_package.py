from __future__ import annotations

import argparse

import susc_17c26_multimodal_ops_common as s17c26


def main() -> int:
    parser = argparse.ArgumentParser(description="SUSC-17C26 pacote multimodal operacional, evidence graph e fila G4/G5")
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in [
        "build-dossiers", "build-evidence-packets", "build-evidence-graph", "rank-review-targets",
        "build-ground-reference-queue", "build-source-query-packages", "build-review-checklists",
        "build-master-index", "build-reproducibility-manifest", "build-offline", "audit",
    ]:
        sub.add_parser(mode)
    args = parser.parse_args()

    if args.mode == "build-dossiers":
        print(s17c26.build_dossiers_text())
        return 0
    if args.mode == "build-evidence-packets":
        print(s17c26.build_evidence_packets_text())
        return 0
    if args.mode == "build-evidence-graph":
        print(s17c26.build_evidence_graph_text())
        return 0
    if args.mode == "rank-review-targets":
        print(s17c26.rank_review_targets_text())
        return 0
    if args.mode == "build-ground-reference-queue":
        print(s17c26.build_ground_reference_queue_text())
        return 0
    if args.mode == "build-source-query-packages":
        print(s17c26.build_source_query_packages_text())
        return 0
    if args.mode == "build-review-checklists":
        print(s17c26.build_review_checklists_text())
        return 0
    if args.mode == "build-master-index":
        print(s17c26.build_master_index_text())
        return 0
    if args.mode == "build-reproducibility-manifest":
        print(s17c26.build_reproducibility_manifest_text())
        return 0
    if args.mode == "build-offline":
        s17c26.build_all()
        print("build-offline: outputs 17C26 reconstruidos a partir dos artefatos locais.")
        return 0
    if args.mode == "audit":
        return s17c26.validate()
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
