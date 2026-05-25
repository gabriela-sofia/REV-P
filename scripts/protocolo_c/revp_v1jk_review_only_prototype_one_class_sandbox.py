"""
REV-P v1jk - REVIEW_ONLY_PROTOTYPE_AND_ONE_CLASS_SANDBOX.

Runs a local review-only sandbox over the v1ji/v1jj official-anchor batch.
The stage builds a feature table, exploratory prototype summaries, optional
one-class scores, and a final boundary. It does not create labels, formal
negatives, model weights, scientific claims, or DINO unfreeze.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REVP_ROOT = SCRIPT_PATH.parents[2]
LOCAL_RUN_DIR = REVP_ROOT / "local_runs" / "protocolo_c" / "v1jk"
V1JI_PATCH_ROOT = REVP_ROOT / "local_runs" / "protocolo_c" / "v1ji" / "patches"
DATASETS_DIR = REVP_ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"

ANCHORS_PATH = DATASETS_DIR / "official_multi_anchor_registry.csv"
PATCH_PATH = DATASETS_DIR / "multi_anchor_multimodal_patch_registry.csv"
DINO_PATH = DATASETS_DIR / "multi_anchor_dino_review_embedding_registry.csv"
CONTROLS_PATH = DATASETS_DIR / "control_candidate_expansion_registry.csv"
BOUNDARY_PATH = DATASETS_DIR / "sandbox_training_boundary_registry.csv"

FEATURE_FIELDS = [
    "anchor_id",
    "date",
    "locality_or_unit",
    "phenomenon_group",
    "dino_cosine_similarity",
    "dino_euclidean_distance",
    "ndwi_pre_mean",
    "ndwi_post_mean",
    "ndwi_delta",
    "ndbi_pre_mean",
    "ndbi_post_mean",
    "ndbi_delta",
    "b02_delta",
    "b03_delta",
    "b04_delta",
    "b08_delta",
    "b11_delta",
    "b12_delta",
    "dem_mean",
    "slope_mean",
    "aspect_mean",
    "s1_availability",
    "s2_qa_status",
    "control_type",
    "feature_status",
]

REGISTRY_FIELDS = [
    "sandbox_id",
    "feature_count",
    "anchor_count",
    "control_candidate_count",
    "dino_available_count",
    "spectral_available_count",
    "dem_available_count",
    "one_class_sandbox_status",
    "prototype_analysis_status",
    "scientific_claim_status",
    "supervised_training_status",
    "can_create_training_label",
    "can_train_model",
    "can_unfreeze_dino_for_scientific_claim",
    "primary_blocker",
    "minimum_evidence_needed",
    "notes",
]

PRIVATE_FRAGMENTS = ["C:\\Users\\gabriela", "Documents\\REV-P", "Documents/REV-P"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def write_schema(path: Path, fields: list[str], prefix: str) -> None:
    write_csv(path, [{"field": field, "description": f"{prefix}: {field}."} for field in fields], ["field", "description"])


def prepare(force: bool) -> None:
    if force and LOCAL_RUN_DIR.exists():
        resolved = LOCAL_RUN_DIR.resolve()
        expected = (REVP_ROOT / "local_runs" / "protocolo_c" / "v1jk").resolve()
        if resolved != expected:
            raise RuntimeError(f"Refusing to clear unexpected path: {resolved}")
        shutil.rmtree(resolved)
    LOCAL_RUN_DIR.mkdir(parents=True, exist_ok=True)


def safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")[:120]


def float_or_blank(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except Exception:
        return None


def fmt(value: float | None, digits: int = 8) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def by_anchor(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["anchor_id"]: row for row in rows if row.get("anchor_id")}


def read_s2_stats(anchor_id: str, relation: str) -> dict[str, float] | None:
    path = V1JI_PATCH_ROOT / safe_slug(anchor_id) / "s2" / f"{anchor_id}_{relation}_s2.local_geotiff"
    if not path.exists():
        return None
    try:
        import numpy as np
        import rasterio

        with rasterio.open(path) as src:
            data = src.read([1, 2, 3, 4, 5, 6]).astype("float64")
        means = np.nanmean(data, axis=(1, 2))
        b02, b03, b04, b08, b11, b12 = [float(v) for v in means]
        ndwi = float(np.nanmean((data[1] - data[3]) / np.maximum(data[1] + data[3], 1.0e-9)))
        ndbi = float(np.nanmean((data[4] - data[3]) / np.maximum(data[4] + data[3], 1.0e-9)))
        return {"b02": b02, "b03": b03, "b04": b04, "b08": b08, "b11": b11, "b12": b12, "ndwi": ndwi, "ndbi": ndbi}
    except Exception:
        return None


def read_dem_stats(anchor_id: str) -> dict[str, float] | None:
    path = V1JI_PATCH_ROOT / safe_slug(anchor_id) / "dem" / f"{anchor_id}_dem_terrain.local_geotiff"
    if not path.exists():
        return None
    try:
        import numpy as np
        import rasterio

        with rasterio.open(path) as src:
            data = src.read(masked=True).astype("float64")
        means = np.ma.mean(data, axis=(1, 2)).filled(float("nan"))
        return {
            "dem_mean": float(means[0]) if len(means) > 0 else float("nan"),
            "slope_mean": float(means[1]) if len(means) > 1 else float("nan"),
            "aspect_mean": float(means[2]) if len(means) > 2 else float("nan"),
        }
    except Exception:
        return None


def build_feature_table(anchors: list[dict[str, str]], patch_rows: list[dict[str, str]], dino_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    patch_map = by_anchor(patch_rows)
    dino_map = by_anchor(dino_rows)
    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        anchor_id = anchor["anchor_id"]
        patch = patch_map.get(anchor_id, {})
        dino = dino_map.get(anchor_id, {})
        pre = read_s2_stats(anchor_id, "pre")
        post = read_s2_stats(anchor_id, "post")
        dem = read_dem_stats(anchor_id) or {}
        s1_pair = patch.get("s1_pre_status") == "QA_PASS" and patch.get("s1_post_status") == "QA_PASS"
        row = {
            "anchor_id": anchor_id,
            "date": anchor.get("date", ""),
            "locality_or_unit": anchor.get("documented_event_unit_id", ""),
            "phenomenon_group": anchor.get("phenomenon_group", ""),
            "dino_cosine_similarity": dino.get("cosine_similarity", ""),
            "dino_euclidean_distance": dino.get("euclidean_distance", ""),
            "ndwi_pre_mean": fmt(pre["ndwi"]) if pre else "",
            "ndwi_post_mean": fmt(post["ndwi"]) if post else "",
            "ndwi_delta": fmt(post["ndwi"] - pre["ndwi"]) if pre and post else "",
            "ndbi_pre_mean": fmt(pre["ndbi"]) if pre else "",
            "ndbi_post_mean": fmt(post["ndbi"]) if post else "",
            "ndbi_delta": fmt(post["ndbi"] - pre["ndbi"]) if pre and post else "",
            "dem_mean": fmt(dem.get("dem_mean")),
            "slope_mean": fmt(dem.get("slope_mean")),
            "aspect_mean": fmt(dem.get("aspect_mean")),
            "s1_availability": "S1_PRE_POST_QA_PASS" if s1_pair else "S1_PAIR_INCOMPLETE_OR_UNAVAILABLE",
            "s2_qa_status": "S2_PRE_POST_QA_PASS" if patch.get("s2_pre_status") == "QA_PASS" and patch.get("s2_post_status") == "QA_PASS" else "S2_QA_INCOMPLETE",
            "control_type": "OFFICIAL_ANCHOR_REVIEW_CANDIDATE",
            "feature_status": "FEATURES_READY" if dino.get("dino_status") == "DINO_QA_PASS" else "DINO_FEATURE_UNAVAILABLE",
        }
        for band in ["b02", "b03", "b04", "b08", "b11", "b12"]:
            row[f"{band}_delta"] = fmt(post[band] - pre[band]) if pre and post else ""
        rows.append(row)
    return rows


def numeric_vector(row: dict[str, Any]) -> list[float]:
    keys = [
        "dino_cosine_similarity",
        "dino_euclidean_distance",
        "ndwi_delta",
        "ndbi_delta",
        "b02_delta",
        "b03_delta",
        "b04_delta",
        "b08_delta",
        "b11_delta",
        "b12_delta",
        "dem_mean",
        "slope_mean",
        "aspect_mean",
    ]
    return [float_or_blank(row.get(key)) or 0.0 for key in keys]


def standardize(matrix: list[list[float]]) -> list[list[float]]:
    if not matrix:
        return []
    cols = list(zip(*matrix))
    mus = [mean(col) for col in cols]
    sigmas = [pstdev(col) or 1.0 for col in cols]
    return [[(value - mus[i]) / sigmas[i] for i, value in enumerate(row)] for row in matrix]


def distance_distribution(feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(feature_rows):
        lc = float_or_blank(left.get("dino_cosine_similarity"))
        le = float_or_blank(left.get("dino_euclidean_distance"))
        if lc is None or le is None:
            continue
        for right in feature_rows[i + 1 :]:
            rc = float_or_blank(right.get("dino_cosine_similarity"))
            re = float_or_blank(right.get("dino_euclidean_distance"))
            if rc is None or re is None:
                continue
            dist = math.sqrt((lc - rc) ** 2 + (le - re) ** 2)
            rows.append(
                {
                    "pair_id": f"{left['anchor_id']}__{right['anchor_id']}",
                    "left_anchor_id": left["anchor_id"],
                    "right_anchor_id": right["anchor_id"],
                    "dino_pair_metric_distance": f"{dist:.8f}",
                    "distance_basis": "DINO_PRE_POST_DIAGNOSTIC_METRICS_ONLY",
                    "notes": "Distance uses stored pair diagnostics, not raw DINO vectors.",
                }
            )
    return rows


def change_ranking(feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = [float_or_blank(row.get("dino_euclidean_distance")) or 0.0 for row in feature_rows]
    mu = mean(values) if values else 0.0
    sigma = pstdev(values) or 1.0
    ranked = sorted(feature_rows, key=lambda row: float_or_blank(row.get("dino_euclidean_distance")) or -1.0, reverse=True)
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked, start=1):
        value = float_or_blank(row.get("dino_euclidean_distance")) or 0.0
        z = (value - mu) / sigma
        rows.append(
            {
                "rank": rank,
                "anchor_id": row["anchor_id"],
                "dino_cosine_similarity": row.get("dino_cosine_similarity", ""),
                "dino_euclidean_distance": row.get("dino_euclidean_distance", ""),
                "ndwi_delta": row.get("ndwi_delta", ""),
                "ndbi_delta": row.get("ndbi_delta", ""),
                "review_outlier_status": "EXPLORATORY_HIGH_CHANGE_REVIEW" if z >= 1.0 else "WITHIN_REVIEW_RANGE",
                "notes": "Ranking is exploratory review-only and does not create a class or label.",
            }
        )
    return rows


def pca_projection(feature_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    matrix = standardize([numeric_vector(row) for row in feature_rows])
    if not matrix:
        return [], "PCA_NOT_AVAILABLE_NO_FEATURES"
    try:
        import numpy as np

        x = np.array(matrix, dtype="float64")
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        projected = x @ vt[:2].T
        rows = []
        for source, coords in zip(feature_rows, projected):
            rows.append(
                {
                    "anchor_id": source["anchor_id"],
                    "pca_1": f"{float(coords[0]):.8f}",
                    "pca_2": f"{float(coords[1]) if projected.shape[1] > 1 else 0.0:.8f}",
                    "projection_method": "NUMPY_SVD_PCA",
                    "exploratory_group": "",
                    "notes": "Projection is exploratory and does not define classes.",
                }
            )
        try:
            from sklearn.cluster import KMeans

            k = min(3, len(rows))
            labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit(projected).labels_ if k > 1 else [0] * len(rows)
            for row, label in zip(rows, labels):
                row["exploratory_group"] = f"REVIEW_GROUP_{int(label) + 1}"
            return rows, "PCA_AND_EXPLORATORY_CLUSTER_READY"
        except Exception:
            return rows, "PCA_READY_CLUSTER_SKIPPED"
    except Exception as exc:
        return [], f"PCA_FAILED:{type(exc).__name__}"


def one_class_sandbox(feature_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    matrix = standardize([numeric_vector(row) for row in feature_rows])
    if not matrix:
        return [], "ONE_CLASS_SANDBOX_SKIPPED_NO_FEATURES"
    try:
        from sklearn.ensemble import IsolationForest

        model = IsolationForest(random_state=42, contamination="auto")
        model.fit(matrix)
        scores = model.score_samples(matrix)
        rows = []
        for source, score in zip(feature_rows, scores):
            rows.append(
                {
                    "anchor_id": source["anchor_id"],
                    "sandbox_method": "IsolationForest",
                    "one_class_score": f"{float(score):.8f}",
                    "sandbox_status": "INVALID_FOR_SCIENTIFIC_CLAIM",
                    "model_saved": "false",
                    "can_create_training_label": "false",
                    "can_train_model": "false",
                    "notes": "Score is local engineering telemetry only and is not a scientific claim.",
                }
            )
        return rows, "ONE_CLASS_SANDBOX_RAN_INVALID_FOR_CLAIM"
    except Exception as exc:
        return (
            [
                {
                    "anchor_id": "ALL",
                    "sandbox_method": "IsolationForest",
                    "one_class_score": "",
                    "sandbox_status": f"SKLEARN_UNAVAILABLE_OR_FAILED:{type(exc).__name__}",
                    "model_saved": "false",
                    "can_create_training_label": "false",
                    "can_train_model": "false",
                    "notes": "Sandbox did not run; boundary remains blocked.",
                }
            ],
            f"SKLEARN_UNAVAILABLE:{type(exc).__name__}",
        )


def registry_row(feature_rows: list[dict[str, Any]], controls: list[dict[str, str]], sandbox_status: str, prototype_status: str) -> dict[str, str]:
    minimum = "formal negatives; absence protocol; label governance; split/leakage protocol; independent validation metrics"
    return {
        "sandbox_id": "V1JK_REVIEW_ONLY_PROTOTYPE_ONE_CLASS_SANDBOX",
        "feature_count": str(len(FEATURE_FIELDS) - 1),
        "anchor_count": str(len(feature_rows)),
        "control_candidate_count": str(len(controls)),
        "dino_available_count": str(sum(1 for row in feature_rows if row.get("dino_euclidean_distance"))),
        "spectral_available_count": str(sum(1 for row in feature_rows if row.get("ndwi_delta") and row.get("ndbi_delta"))),
        "dem_available_count": str(sum(1 for row in feature_rows if row.get("dem_mean"))),
        "one_class_sandbox_status": sandbox_status,
        "prototype_analysis_status": prototype_status,
        "scientific_claim_status": "INVALID_FOR_SUPERVISED_CLAIM",
        "supervised_training_status": "SUPERVISED_TRAINING_BLOCKED_NO_NEGATIVES",
        "can_create_training_label": "false",
        "can_train_model": "false",
        "can_unfreeze_dino_for_scientific_claim": "false",
        "primary_blocker": "NO_FORMAL_NEGATIVES_NO_LABEL_PROTOCOL_NO_INDEPENDENT_VALIDATION",
        "minimum_evidence_needed": minimum,
        "notes": "Clusters/prototypes are review-only; one-class score is sandbox telemetry and not a class.",
    }


def public_text_has_private_path(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    return any(fragment in text for fragment in PRIVATE_FRAGMENTS)


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepare(args.force)
    anchors = read_csv(ANCHORS_PATH)
    patch_rows = read_csv(PATCH_PATH)
    dino_rows = read_csv(DINO_PATH)
    controls = read_csv(CONTROLS_PATH)
    boundary = read_csv(BOUNDARY_PATH)

    features = build_feature_table(anchors, patch_rows, dino_rows)
    distances = distance_distribution(features)
    ranking = change_ranking(features)
    pca_rows, prototype_status = pca_projection(features)
    sandbox_rows, one_class_status = one_class_sandbox(features)
    boundary_rows = [
        {
            "decision_id": "V1JK_BOUNDARY_AFTER_SANDBOX",
            "review_only_analytics_status": "REVIEW_ONLY_ANALYTICS_READY" if features and ranking else "REVIEW_ONLY_ANALYTICS_BLOCKED",
            "one_class_sandbox_status": one_class_status,
            "supervised_training_status": "SUPERVISED_TRAINING_BLOCKED_NO_NEGATIVES",
            "unfreeze_status": "UNFREEZE_BLOCKED_NO_LABELS_NO_SPLIT",
            "scientific_claim_status": "INVALID_FOR_SUPERVISED_CLAIM",
            "can_create_training_label": "false",
            "can_train_model": "false",
            "can_unfreeze_dino_for_scientific_claim": "false",
            "notes": "Sandbox is local engineering only; no weights are saved and no label is created.",
        }
    ]
    registry = registry_row(features, controls, one_class_status, prototype_status)

    write_csv(LOCAL_RUN_DIR / "v1jk_feature_table.csv", features, FEATURE_FIELDS)
    write_csv(
        LOCAL_RUN_DIR / "v1jk_dino_distance_distribution.csv",
        distances,
        ["pair_id", "left_anchor_id", "right_anchor_id", "dino_pair_metric_distance", "distance_basis", "notes"],
    )
    write_csv(LOCAL_RUN_DIR / "v1jk_anchor_change_ranking.csv", ranking, ["rank", "anchor_id", "dino_cosine_similarity", "dino_euclidean_distance", "ndwi_delta", "ndbi_delta", "review_outlier_status", "notes"])
    write_csv(LOCAL_RUN_DIR / "v1jk_pca_projection.csv", pca_rows, ["anchor_id", "pca_1", "pca_2", "projection_method", "exploratory_group", "notes"])
    write_csv(LOCAL_RUN_DIR / "v1jk_one_class_sandbox_log.csv", sandbox_rows, ["anchor_id", "sandbox_method", "one_class_score", "sandbox_status", "model_saved", "can_create_training_label", "can_train_model", "notes"])
    write_csv(LOCAL_RUN_DIR / "v1jk_training_boundary_after_sandbox.csv", boundary_rows, ["decision_id", "review_only_analytics_status", "one_class_sandbox_status", "supervised_training_status", "unfreeze_status", "scientific_claim_status", "can_create_training_label", "can_train_model", "can_unfreeze_dino_for_scientific_claim", "notes"])

    write_csv(DATASETS_DIR / "review_only_multimodal_sandbox_registry.csv", [registry], REGISTRY_FIELDS)
    write_schema(SCHEMAS_DIR / "review_only_multimodal_sandbox_schema.csv", REGISTRY_FIELDS, "REV-P v1jk review-only sandbox field")

    qa_rows = [
        {"check": "feature_table_created", "status": "PASS" if len(features) == len(anchors) and features else "FAIL", "detail": str(len(features))},
        {"check": "prototypes_not_classes", "status": "PASS" if all("class" not in row.get("exploratory_group", "").lower() for row in pca_rows) else "FAIL", "detail": prototype_status},
        {"check": "sandbox_invalid_for_claim", "status": "PASS" if "INVALID_FOR_CLAIM" in one_class_status or "SKLEARN_UNAVAILABLE" in one_class_status else "FAIL", "detail": one_class_status},
        {"check": "can_train_model_false", "status": "PASS" if registry["can_train_model"] == "false" else "FAIL", "detail": registry["can_train_model"]},
        {"check": "can_create_training_label_false", "status": "PASS" if registry["can_create_training_label"] == "false" else "FAIL", "detail": registry["can_create_training_label"]},
        {"check": "can_unfreeze_false", "status": "PASS" if registry["can_unfreeze_dino_for_scientific_claim"] == "false" else "FAIL", "detail": registry["can_unfreeze_dino_for_scientific_claim"]},
        {"check": "no_private_path_in_public_outputs", "status": "PASS" if not public_text_has_private_path(DATASETS_DIR / "review_only_multimodal_sandbox_registry.csv") else "FAIL", "detail": "public registry checked"},
        {"check": "v1jj_boundary_read", "status": "PASS" if boundary else "FAIL", "detail": str(len(boundary))},
    ]
    write_csv(LOCAL_RUN_DIR / "v1jk_qa.csv", qa_rows, ["check", "status", "detail"])

    summary = {
        "stage": "v1jk",
        "timestamp": utc_now(),
        "feature_rows": len(features),
        "anchor_count": len(anchors),
        "control_candidate_count": len(controls),
        "dino_available_count": int(registry["dino_available_count"]),
        "spectral_available_count": int(registry["spectral_available_count"]),
        "dem_available_count": int(registry["dem_available_count"]),
        "distance_pair_count": len(distances),
        "prototype_analysis_status": prototype_status,
        "one_class_sandbox_status": one_class_status,
        "scientific_claim_status": "INVALID_FOR_SUPERVISED_CLAIM",
        "supervised_training_status": "SUPERVISED_TRAINING_BLOCKED_NO_NEGATIVES",
        "can_create_training_label": False,
        "can_train_model": False,
        "can_unfreeze_dino_for_scientific_claim": False,
    }
    write_json(LOCAL_RUN_DIR / "v1jk_summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--read-v1ji-batch", action="store_true")
    parser.add_argument("--read-v1jj-boundary", action="store_true")
    parser.add_argument("--build-feature-table", action="store_true")
    parser.add_argument("--run-review-only-prototypes", action="store_true")
    parser.add_argument("--run-one-class-sandbox", action="store_true")
    parser.add_argument("--emit-sandbox-report", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print("REV-P v1jk REVIEW-ONLY PROTOTYPE AND ONE-CLASS SANDBOX")
    print(f"Feature rows: {summary['feature_rows']}")
    print(f"DINO available: {summary['dino_available_count']}")
    print(f"Spectral available: {summary['spectral_available_count']}")
    print(f"DEM available: {summary['dem_available_count']}")
    print(f"Distance pairs: {summary['distance_pair_count']}")
    print(f"Prototype status: {summary['prototype_analysis_status']}")
    print(f"One-class sandbox: {summary['one_class_sandbox_status']}")
    print(f"Training allowed: {summary['can_train_model']}")
    print("No git add, commit, or push was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
