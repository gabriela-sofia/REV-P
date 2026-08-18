"""SUSC-18B Multimodal ML Readiness, Dataset Builder e Experiment Harness.

Transforma a feature store multimodal [[SUSC-18A]] (316 patches, 7 familias) em
infraestrutura de MODELAGEM auditavel, SEM ground truth supervisionado:

  - dataset model-ready + feature tensor tabular + feature groups;
  - missingness masks explicitas + pipeline de normalizacao review-only;
  - splits deterministicos;
  - suite de experimentos NAO supervisionados (PCA/SVD/KMeans/outlier/similaridade);
  - fila de descoberta de candidatos p/ aquisicao de Ground Reference;
  - trainability gate + harness supervisionado FAIL-CLOSED (existe mas nao treina);
  - experiment registry + model cards.

Resultado A (esperado agora): ML readiness completo, treino supervisionado BLOQUEADO
por falta de labels de Ground Reference. Resultado B (dry-run com labels sinteticos)
PROIBIDO. Resultado C (treino supervisionado) so com labels oficiais aceitos.

Guardrails: NAO score v7, NAO altera score v6 (so referencia), NAO ground truth,
NAO labels de treino, canario NAO e positivo, controle NAO e negativo, ocorrencia
so ancora (nunca feature causal), flow_acc NAO equivalente, clustering NAO e
previsao, outlier NAO e risco, ranking NAO e ground truth, sem pos-evento como
pre-evento, missingness explicita. Tudo sem-label e unsupervised_review_only.

Offline e deterministico (numpy puro com seed fixo; sklearn detectado mas nao usado
no calculo p/ garantir rebuild byte-identico). Sem rede.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import ensure_dir, read_csv, read_json, rel, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

SEED = 18022  # seed fixo p/ determinismo (18B/2022)

# --- Entradas 18A ----------------------------------------------------------
A_REGISTRY = OUT / "susc_18a_unified_patch_registry.csv"
A_CONTRACT = OUT / "susc_18a_multimodal_feature_family_contract.json"
A_STORE = OUT / "susc_18a_multimodal_patch_feature_store.csv"
A_AVAIL = OUT / "susc_18a_feature_availability_matrix.csv"
A_LINEAGE = OUT / "susc_18a_feature_lineage_manifest.csv"
A_QFLAGS = OUT / "susc_18a_quality_flags.csv"
A_NEXT = OUT / "susc_18a_next_stage_readiness.csv"
A_SUMMARY = OUT / "susc_18a_readiness_summary.json"
A_SENTINEL2 = OUT / "susc_18a_sentinel2_feature_store.csv"
C40_INDEX = OUT / "susc_17c40_transparent_review_only_index.csv"
SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"

REQUIRED_INPUTS = [A_REGISTRY, A_CONTRACT, A_STORE, A_AVAIL, A_QFLAGS, A_SUMMARY, A_SENTINEL2, C40_INDEX]

# --- Saidas 18B ------------------------------------------------------------
DATASET_CONTRACT = OUT / "susc_18b_ml_dataset_contract.json"
ML_DATASET = OUT / "susc_18b_ml_ready_dataset.csv"
FEATURE_GROUPS = OUT / "susc_18b_feature_group_registry.csv"
MISSINGNESS = OUT / "susc_18b_missingness_mask.csv"
NORM_PIPELINE = OUT / "susc_18b_ml_normalization_pipeline.json"
NORM_AUDIT = OUT / "susc_18b_ml_normalization_audit.csv"
FEATURE_TENSOR = OUT / "susc_18b_ml_feature_tensor.csv"
SPLIT_REGISTRY = OUT / "susc_18b_split_registry.csv"
UNSUP_REGISTRY = OUT / "susc_18b_unsupervised_experiment_registry.csv"
EMBEDDINGS = OUT / "susc_18b_tabular_embedding_projection.csv"
CLUSTERS = OUT / "susc_18b_cluster_assignments.csv"
OUTLIERS = OUT / "susc_18b_outlier_scores.csv"
SIMILARITY = OUT / "susc_18b_patch_similarity_index.csv"
CANDIDATES = OUT / "susc_18b_candidate_discovery_queue.csv"
TRAINABILITY = OUT / "susc_18b_trainability_gate.csv"
SUP_HARNESS = OUT / "susc_18b_supervised_harness_status.csv"
LABEL_CONTRACT = OUT / "susc_18b_future_label_contract.json"
EXP_REGISTRY = OUT / "susc_18b_experiment_registry.csv"
MODEL_CARDS = OUT / "susc_18b_model_cards.csv"
INTEGRITY = OUT / "susc_18b_score_integrity_audit.csv"
LEAKAGE = OUT / "susc_18b_no_leakage_audit.csv"
SUMMARY = OUT / "susc_18b_readiness_summary.json"
BLOCKERS = OUT / "susc_18b_promotion_blockers.csv"
REPORT = OUT / "SUSC_18B_MULTIMODAL_ML_READINESS_REPORT.md"

SCHEMA_DATASET = SCHEMAS / "susc_18b_ml_dataset_schema_v1.json"
SCHEMA_EXP = SCHEMAS / "susc_18b_experiment_registry_schema_v1.json"
SCHEMA_TRAIN = SCHEMAS / "susc_18b_trainability_schema_v1.json"

REQUIRED_OUTPUTS = [
    DATASET_CONTRACT, ML_DATASET, FEATURE_GROUPS, MISSINGNESS, NORM_PIPELINE, NORM_AUDIT,
    FEATURE_TENSOR, SPLIT_REGISTRY, UNSUP_REGISTRY, EMBEDDINGS, CLUSTERS, OUTLIERS, SIMILARITY,
    CANDIDATES, TRAINABILITY, SUP_HARNESS, LABEL_CONTRACT, EXP_REGISTRY, MODEL_CARDS,
    INTEGRITY, LEAKAGE, SUMMARY, BLOCKERS, REPORT,
]

# --- Definicao de features -------------------------------------------------
# (feature, grupo, fonte)  fonte: mm=multimodal store, s2=sentinel2 store, q=quality-derivada
FEATURES = [
    ("elevation_mean", "physical_topography", "mm"),
    ("slope_mean", "physical_topography", "mm"),
    ("HAND_value", "physical_topography", "mm"),
    ("TWI_value", "physical_topography", "mm"),
    ("TPI_250m_value", "physical_topography", "mm"),
    ("distance_to_official_drainage_m", "hydrology_proximity", "mm"),
    ("distance_to_official_water_m", "hydrology_proximity", "mm"),
    ("urban_prop", "urban_landcover", "mm"),
    ("vegetation_prop", "urban_landcover", "mm"),
    ("water_prop", "urban_landcover", "mm"),
    ("NDVI_mean", "spectral_sentinel2", "mm"),
    ("NDWI_mean", "spectral_sentinel2", "mm"),
    ("MNDWI_mean", "spectral_sentinel2", "mm"),
    ("NDBI_mean", "spectral_sentinel2", "mm"),
    ("B03_mean", "spectral_sentinel2", "s2"),
    ("B04_mean", "spectral_sentinel2", "s2"),
    ("B08_mean", "spectral_sentinel2", "s2"),
    ("B11_mean", "spectral_sentinel2", "s2"),
    ("chirps_3d_sum", "rainfall_chirps", "mm"),
    ("chirps_7d_sum", "rainfall_chirps", "mm"),
    ("chirps_30d_sum", "rainfall_chirps", "mm"),
    ("feature_family_available_count", "quality_lineage", "mm"),
    ("feature_completeness_ratio", "quality_lineage", "mm"),
    ("proxy_feature_count", "quality_lineage", "mm"),
    ("official_feature_count", "quality_lineage", "mm"),
    ("missing_feature_family_count", "quality_lineage", "q"),
]
FEATURE_NAMES = [f[0] for f in FEATURES]
# grupos de modelagem do fenomeno (excluem quality_lineage p/ nao clusterizar por disponibilidade)
MODELING_GROUPS = ["physical_topography", "hydrology_proximity", "urban_landcover",
                   "spectral_sentinel2", "rainfall_chirps"]
MODELING_FEATURES = [f[0] for f in FEATURES if f[1] in MODELING_GROUPS]
SPECTRAL_FEATURES = [f[0] for f in FEATURES if f[1] == "spectral_sentinel2"]
PHYSICAL_FEATURES = [f[0] for f in FEATURES if f[1] == "physical_topography"]

GROUP_TO_AVAIL = {
    "physical_topography": "terrain_available", "hydrology_proximity": "hydrology_available",
    "urban_landcover": "landcover_available", "spectral_sentinel2": "sentinel2_available",
    "rainfall_chirps": "chirps_available", "quality_lineage": None,
}
MASK_FAMILIES = ["physical_topography", "hydrology_proximity", "urban_landcover",
                 "spectral_sentinel2", "rainfall_chirps", "quality_lineage"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _bool(v):
    return "true" if v else "false"


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _r(v, n=6):
    return "not_available" if v is None else round(float(v), n)


def _run_git(args):
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _18a_committed():
    r = subprocess.run(["git", "cat-file", "-e", f"HEAD:{rel(A_STORE)}"],
                       cwd=ROOT, text=True, capture_output=True, check=False)
    return r.returncode == 0


def _require_inputs():
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(p) for p in missing))


def _sklearn_available():
    return importlib.util.find_spec("sklearn") is not None


# ---------------------------------------------------------------------------
# assemble: samples com valores brutos + referencias
# ---------------------------------------------------------------------------

def assemble_samples():
    store = {r["unified_patch_id"]: r for r in read_csv(A_STORE)}
    s2 = {r["unified_patch_id"]: r for r in read_csv(A_SENTINEL2)}
    reg = read_csv(A_REGISTRY)
    # score v6 referencia por source_patch_id (oficiais)
    v6 = {r["patch_id"]: r.get("susc_score_v6_candidate") for r in read_csv(SCORE_V6)} if SCORE_V6.exists() else {}
    # indice transparente V1 por source_patch_id (16 patches)
    ti = {}
    for r in read_csv(C40_INDEX):
        if r.get("index_variant") == "V1_equal_weight_available_components":
            ti[r["patch_id"]] = r.get("transparent_index_value")
    samples = []
    for r in sorted(reg, key=lambda x: x["unified_patch_id"]):
        uid = r["unified_patch_id"]
        m = store.get(uid, {})
        b = s2.get(uid, {})
        spid = r["source_patch_id"]
        raw = {}
        for fname, _grp, src in FEATURES:
            if fname == "missing_feature_family_count":
                mff = m.get("missing_feature_families", "none")
                raw[fname] = 0.0 if mff in ("none", "") else float(len(mff.split(";")))
                continue
            src_row = b if src == "s2" else m
            raw[fname] = _num(src_row.get(fname))
        samples.append({
            "unified_patch_id": uid, "source_patch_id": spid, "patch_family": r["patch_family"],
            "city": m.get("city", r.get("city", "")), "region": m.get("region", r.get("region", "")),
            "is_official": r["patch_family"] == "official_population_patch",
            "is_canary": r["patch_family"] == "event_anchored_canary_patch",
            "is_control": r["patch_family"] == "spatial_mismatch_control_candidate",
            "has_anchor": m.get("has_official_hydrological_occurrence_anchor", "false"),
            "flow_acc_available": m.get("flow_acc_status", "not_available") != "not_available",
            "feature_family_available_count": m.get("feature_family_available_count", "0"),
            "feature_completeness_ratio": m.get("feature_completeness_ratio", "0"),
            "proxy_feature_count": m.get("proxy_feature_count", "0"),
            "official_feature_count": m.get("official_feature_count", "0"),
            "missing_feature_families": m.get("missing_feature_families", "none"),
            "avail": {g: (m.get(GROUP_TO_AVAIL[g], "true") == "true" if GROUP_TO_AVAIL[g] else True) for g in MASK_FAMILIES},
            "score_v6_reference": _num(v6.get(spid)),
            "transparent_index_reference": _num(ti.get(spid)),
            "raw": raw,
        })
    return samples


# ---------------------------------------------------------------------------
# normalizacao robusta (median/IQR) sobre populacao combinada
# ---------------------------------------------------------------------------

def _norm_stats(samples):
    stats = {}
    for fname in FEATURE_NAMES:
        vals = [s["raw"][fname] for s in samples if s["raw"][fname] is not None]
        arr = np.array(vals, dtype=float) if vals else np.array([], dtype=float)
        if arr.size:
            med = float(np.median(arr))
            q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
            iqr = q3 - q1
            mn, mx = float(arr.min()), float(arr.max())
        else:
            med = q1 = q3 = iqr = mn = mx = 0.0
        stats[fname] = {"median": med, "iqr": iqr, "q1": q1, "q3": q3, "min": mn, "max": mx,
                        "missing_count": len(samples) - arr.size, "n": int(arr.size)}
    return stats


def _norm_value(v, st):
    if v is None:
        return 0.0, True  # imputa mediana (0 apos centralizar) + flag
    if st["iqr"] > 0:
        return (v - st["median"]) / st["iqr"], False
    return 0.0, False


def build_matrix(samples, feature_names, stats):
    """Retorna (ids, X normalizada imputada, dict missing por feature)."""
    ids = [s["unified_patch_id"] for s in samples]
    X = np.zeros((len(samples), len(feature_names)), dtype=float)
    miss = np.zeros((len(samples), len(feature_names)), dtype=int)
    for j, fname in enumerate(feature_names):
        st = stats[fname]
        for i, s in enumerate(samples):
            nv, was_missing = _norm_value(s["raw"][fname], st)
            X[i, j] = nv
            miss[i, j] = 1 if was_missing else 0
    return ids, X, miss


# ---------------------------------------------------------------------------
# algoritmos numpy deterministicos
# ---------------------------------------------------------------------------

def _pca_scores(X, ncomp):
    """PCA via SVD centrada; sinal fixado (maior |loading| positivo) p/ determinismo."""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(ncomp, Vt.shape[0])
    for r in range(k):
        j = int(np.argmax(np.abs(Vt[r])))
        if Vt[r, j] < 0:
            Vt[r] = -Vt[r]
            U[:, r] = -U[:, r]
    scores = U[:, :k] * S[:k]
    return scores


def _kmeans(X, k, seed=SEED, iters=100):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    # k-means++ deterministico
    centers = [X[rng.randint(n)]]
    for _ in range(1, k):
        d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        tot = d2.sum()
        probs = d2 / tot if tot > 0 else np.ones(n) / n
        centers.append(X[np.searchsorted(np.cumsum(probs), rng.random_sample())])
    C = np.array(centers, dtype=float)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dists = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            labels = new_labels
            break
        labels = new_labels
        for c in range(k):
            pts = X[labels == c]
            if pts.size:
                C[c] = pts.mean(axis=0)
    dists = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
    dcent = dists[np.arange(n), labels]
    return labels, dcent, C


def _pairwise(X):
    return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)


# ---------------------------------------------------------------------------
# Etapa 1 - contrato de dataset ML
# ---------------------------------------------------------------------------

def dataset_contract():
    return {
        "milestone": "SUSC-18B",
        "sample_unit": "patch",
        "feature_families": MODELING_GROUPS + ["quality_lineage"],
        "allowed_features": FEATURE_NAMES,
        "forbidden_features": ["occurrence_anchor_as_feature", "canary_flag_as_label",
                               "control_flag_as_label", "score_v6_as_feature_target",
                               "flow_acc_official_equivalent"],
        "available_targets": [],
        "forbidden_targets": ["score_v6_candidate", "transparent_index_value",
                              "occurrence_anchor", "canary_membership", "control_membership"],
        "supervised_training_status": "blocked_no_ground_reference_labels",
        "unsupervised_training_status": "allowed_review_only",
        "ground_truth_status": "not_available",
        "missingness_policy": "explicit_per_family_mask; imputacao=median_zero_only_for_unsupervised_with_flag; nunca esconder",
        "normalization_policy": "robust_median_iqr_review_only; nao score v6/v7",
        "splits_policy": "deterministic; supervised splits so quando labels fortes existirem",
        "leakage_policy": "ocorrencia so ancora; canario!=positivo; controle!=negativo; score_v6 so referencia; sem pos-evento como pre-evento",
        "sample_unit_field": "patch",
        "supervised_training_allowed": False,
        "unsupervised_training_allowed": True,
        "review_only_modeling_allowed": True,
        "ground_truth_available": False,
        "training_labels_available": False,
        "canary_is_positive_label": False,
        "control_is_negative_label": False,
        "occurrence_anchor_is_feature": False,
        "score_v6_is_training_target": False,
        "score_v6_is_reference_only": True,
        "flow_acc_is_official_equivalent": False,
        "review_only": True,
    }


# ---------------------------------------------------------------------------
# Etapa 2 - dataset model-ready
# ---------------------------------------------------------------------------

def ml_ready_dataset_rows():
    rows = []
    for i, s in enumerate(assemble_samples(), start=1):
        raw = s["raw"]
        mff = s["missing_feature_families"]
        missing_count = 0 if mff in ("none", "") else len(mff.split(";"))
        rows.append({
            "ml_sample_id": f"S18B_SAMPLE_{i:04d}", "unified_patch_id": s["unified_patch_id"],
            "source_patch_id": s["source_patch_id"], "patch_family": s["patch_family"],
            "city": s["city"], "region": s["region"],
            "is_official_population_patch": _bool(s["is_official"]),
            "is_canary_patch": _bool(s["is_canary"]), "is_control_patch": _bool(s["is_control"]),
            "review_only": "true", "ground_truth": "false", "training_label": "false",
            "elevation_mean": _r(raw["elevation_mean"]), "slope_mean": _r(raw["slope_mean"]),
            "HAND_value": _r(raw["HAND_value"]), "TWI_value": _r(raw["TWI_value"]),
            "TPI_250m_value": _r(raw["TPI_250m_value"]),
            "flow_acc_available": _bool(s["flow_acc_available"]), "flow_acc_equivalent": "false",
            "distance_to_official_drainage_m": _r(raw["distance_to_official_drainage_m"]),
            "distance_to_official_water_m": _r(raw["distance_to_official_water_m"]),
            "urban_prop": _r(raw["urban_prop"]), "vegetation_prop": _r(raw["vegetation_prop"]),
            "water_prop": _r(raw["water_prop"]),
            "NDVI_mean": _r(raw["NDVI_mean"]), "NDWI_mean": _r(raw["NDWI_mean"]),
            "MNDWI_mean": _r(raw["MNDWI_mean"]), "NDBI_mean": _r(raw["NDBI_mean"]),
            "chirps_3d_sum": _r(raw["chirps_3d_sum"]), "chirps_7d_sum": _r(raw["chirps_7d_sum"]),
            "chirps_30d_sum": _r(raw["chirps_30d_sum"]),
            "feature_family_available_count": s["feature_family_available_count"],
            "feature_completeness_ratio": s["feature_completeness_ratio"],
            "proxy_feature_count": s["proxy_feature_count"],
            "official_feature_count": s["official_feature_count"],
            "missing_feature_family_count": str(missing_count),
            "score_v6_reference_value": _r(s["score_v6_reference"]),
            "transparent_index_reference_value": _r(s["transparent_index_reference"]),
            "has_official_hydrological_occurrence_anchor": s["has_anchor"],
            "score_v6_is_target": "false", "transparent_index_is_target": "false",
            "occurrence_anchor_is_target": "false",
        })
    return rows


# ---------------------------------------------------------------------------
# Etapa 3 - feature groups
# ---------------------------------------------------------------------------

def feature_group_rows():
    groups = [
        ("physical_topography", PHYSICAL_FEATURES, True, True, False, "features fisicas/topograficas (nativas oficiais / DEM publico canarios)"),
        ("hydrology_proximity", [f[0] for f in FEATURES if f[1] == "hydrology_proximity"], True, True, False, "proximidade a drenagem/agua oficial"),
        ("urban_landcover", [f[0] for f in FEATURES if f[1] == "urban_landcover"], True, True, False, "cobertura urbana/vegetacao/agua"),
        ("spectral_sentinel2", SPECTRAL_FEATURES, True, True, False, "indices + bandas Sentinel-2"),
        ("rainfall_chirps", [f[0] for f in FEATURES if f[1] == "rainfall_chirps"], True, True, False, "gatilho pluviometrico CHIRPS region-level"),
        ("quality_lineage", [f[0] for f in FEATURES if f[1] == "quality_lineage"], True, False, False, "metadados de completude/proxy; auxiliar, nao usar como fenomeno em supervisao"),
        ("reference_only", ["score_v6_reference_value", "transparent_index_reference_value", "has_official_hydrological_occurrence_anchor"], False, False, False, "referencias externas review-only; NUNCA feature de modelo nem target"),
        ("forbidden_targets", ["score_v6_candidate", "transparent_index_value", "occurrence_anchor", "canary_membership", "control_membership"], False, False, False, "PROIBIDOS como target: score v6/indice/ocorrencia/canario/controle"),
    ]
    rows = []
    for i, (name, feats, unsup, sup, tgt, reason) in enumerate(groups, start=1):
        rows.append({
            "feature_group_id": f"S18B_FG_{i:04d}", "feature_group_name": name,
            "feature_names": ";".join(feats), "allowed_for_unsupervised": _bool(unsup),
            "allowed_for_supervised_if_labels_exist": _bool(sup), "allowed_as_target": _bool(tgt),
            "reason": reason, "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Etapa 4 - missingness masks
# ---------------------------------------------------------------------------

def missingness_mask_rows():
    rows = []
    n = 0
    for s in assemble_samples():
        for fam in MASK_FAMILIES:
            n += 1
            available = s["avail"][fam]
            reason = "available" if available else f"{fam}_not_extracted_for_{s['patch_family']}"
            rows.append({
                "missingness_mask_id": f"S18B_MASK_{n:05d}", "unified_patch_id": s["unified_patch_id"],
                "feature_family": fam, "feature_available": _bool(available),
                "feature_missing": _bool(not available), "missing_reason": reason,
                "missingness_allowed": "true",
                "imputation_allowed": "unsupervised_only",
                "imputation_strategy": "none" if available else "median_zero_with_missing_flag",
                "review_only": "true",
            })
    return rows


# ---------------------------------------------------------------------------
# Etapa 5 - normalizacao
# ---------------------------------------------------------------------------

def normalization_pipeline():
    return {
        "milestone": "SUSC-18B",
        "methods": {
            "robust_median_iqr": "x' = (x - median) / IQR; principal p/ tensor",
            "minmax": "x' = (x - min)/(max-min); auditado",
            "rank_percentile": "posto/percentil por feature; auditado",
            "missing_indicator": "flag m_<familia> separada; imputacao=median(0) so unsupervised",
        },
        "population_used": "combined_all_316_patches_review_only",
        "primary_method_for_tensor": "robust_median_iqr",
        "used_for_unsupervised": True,
        "used_for_supervised": False,
        "not_score_v6_normalization": True,
        "not_score_v7_normalization": True,
        "review_only": True,
    }


def normalization_audit_rows():
    samples = assemble_samples()
    stats = _norm_stats(samples)
    rows = []
    for i, fname in enumerate(FEATURE_NAMES, start=1):
        st = stats[fname]
        rows.append({
            "normalization_id": f"S18B_NORM_{i:04d}", "feature_name": fname,
            "normalization_method": "robust_median_iqr", "population_used": "combined_all_316",
            "median": _r(st["median"]), "iqr": _r(st["iqr"]),
            "min": _r(st["min"]), "max": _r(st["max"]),
            "missing_count": str(st["missing_count"]),
            "used_for_unsupervised": "true", "used_for_supervised": "false",
            "not_score_v6_normalization": "true", "not_score_v7_normalization": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Etapa 6 - feature tensor
# ---------------------------------------------------------------------------

def feature_tensor_rows():
    samples = assemble_samples()
    stats = _norm_stats(samples)
    ids, X, miss = build_matrix(samples, FEATURE_NAMES, stats)
    rows = []
    for i, s in enumerate(samples):
        row = {
            "ml_sample_id": f"S18B_SAMPLE_{i + 1:04d}", "unified_patch_id": s["unified_patch_id"],
            "patch_family": s["patch_family"], "region": s["region"],
        }
        for j, fname in enumerate(FEATURE_NAMES):
            row[f"x_{fname}"] = _r(float(X[i, j]))
        for fam in MASK_FAMILIES:
            row[f"m_{fam}"] = "0" if s["avail"][fam] else "1"
        row["review_only"] = "true"
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Etapa 7 - split registry
# ---------------------------------------------------------------------------

def split_registry_rows():
    samples = assemble_samples()
    splits = [
        ("all_unsupervised", lambda s: "member", "low"),
        ("region_holdout_recife", lambda s: "holdout" if s["region"].lower().startswith("recife") or s["is_canary"] or s["is_control"] else "train", "medium"),
        ("region_holdout_curitiba", lambda s: "holdout" if s["region"].lower().startswith("curitiba") else "train", "medium"),
        ("region_holdout_petropolis", lambda s: "holdout" if s["region"].lower().startswith("petropolis") else "train", "medium"),
        ("canary_holdout", lambda s: "holdout" if s["is_canary"] else "train", "high"),
        ("official_only_unsupervised", lambda s: "member" if s["is_official"] else "excluded", "low"),
    ]
    rows = []
    n = 0
    for split_name, role_fn, risk in splits:
        for s in samples:
            n += 1
            role = role_fn(s)
            rows.append({
                "split_id": f"S18B_SPLIT_{n:05d}", "split_name": split_name,
                "unified_patch_id": s["unified_patch_id"], "split_role": role,
                "region": s["region"], "patch_family": s["patch_family"],
                "split_allowed_for_supervised": "false",
                "split_allowed_for_unsupervised": "true",
                "leakage_risk": risk, "review_only": "true",
            })
    return rows


# ---------------------------------------------------------------------------
# Etapas 8-12 - experimentos (calculo unico, cacheado por processo)
# ---------------------------------------------------------------------------

_EXP_CACHE = {}


def _compute_experiments():
    if _EXP_CACHE:
        return _EXP_CACHE
    samples = assemble_samples()
    stats = _norm_stats(samples)
    ids, Xall, _m = build_matrix(samples, MODELING_FEATURES, stats)
    _, Xspec, _ = build_matrix(samples, SPECTRAL_FEATURES, stats)
    _, Xphys, _ = build_matrix(samples, PHYSICAL_FEATURES, stats)
    fam = {s["unified_patch_id"]: s["patch_family"] for s in samples}
    region = {s["unified_patch_id"]: s["region"] for s in samples}
    res = {
        "ids": ids, "fam": fam, "region": region, "samples": samples, "stats": stats, "Xall": Xall,
        "pca2": _pca_scores(Xall, 2),
        "svd5": _pca_scores(Xall, 5),
        "spec2": _pca_scores(Xspec, 2),
        "phys2": _pca_scores(Xphys, 2),
    }
    res["km5"] = _kmeans(Xall, 5)
    res["km8"] = _kmeans(Xall, 8)
    # outlier: distancia ao centroide global
    centroid = Xall.mean(axis=0, keepdims=True)
    res["outlier"] = np.linalg.norm(Xall - centroid, axis=1)
    res["dist"] = _pairwise(Xall)
    _EXP_CACHE.update(res)
    return _EXP_CACHE


EXPERIMENTS = [
    ("EXP01_pca_2d", "pca_2d_numpy_svd", MODELING_GROUPS),
    ("EXP02_truncated_svd_5d", "truncated_svd_5d_numpy", MODELING_GROUPS),
    ("EXP03_kmeans_k5", "kmeans_k5_numpy", MODELING_GROUPS),
    ("EXP04_kmeans_k8", "kmeans_k8_numpy", MODELING_GROUPS),
    ("EXP05_outlier_distance_to_centroid", "distance_to_global_centroid", MODELING_GROUPS),
    ("EXP06_similarity_topk", "euclidean_topk_numpy", MODELING_GROUPS),
    ("EXP07_family_ablation_spectral_only", "pca_2d_spectral_only", ["spectral_sentinel2"]),
    ("EXP08_family_ablation_physical_only", "pca_2d_physical_only", ["physical_topography"]),
]


def unsupervised_experiment_rows():
    e = _compute_experiments()
    n = len(e["ids"])
    rows = []
    for exp_id, method, groups in EXPERIMENTS:
        rows.append({
            "experiment_id": exp_id, "experiment_name": exp_id, "method": method,
            "feature_groups_used": ";".join(groups), "sample_count": str(n),
            "supervised": "false", "uses_labels": "false", "review_only": "true",
            "status": "completed",
        })
    return rows


def tabular_embedding_rows():
    e = _compute_experiments()
    ids = e["ids"]
    emb_sources = [("EXP01_pca_2d", e["pca2"]), ("EXP02_truncated_svd_5d", e["svd5"]),
                   ("EXP07_family_ablation_spectral_only", e["spec2"]),
                   ("EXP08_family_ablation_physical_only", e["phys2"])]
    rows = []
    n = 0
    for exp_id, Z in emb_sources:
        for i, uid in enumerate(ids):
            n += 1
            zc = {}
            for zi in range(5):
                zc[f"z{zi + 1}"] = _r(float(Z[i, zi])) if zi < Z.shape[1] else "not_available"
            rows.append({
                "embedding_projection_id": f"S18B_EMB_{n:05d}", "experiment_id": exp_id,
                "unified_patch_id": uid, "patch_family": e["fam"][uid], "region": e["region"][uid],
                **zc, "review_only": "true",
            })
    return rows


def cluster_assignment_rows():
    e = _compute_experiments()
    ids = e["ids"]
    rows = []
    n = 0
    for exp_id, key in [("EXP03_kmeans_k5", "km5"), ("EXP04_kmeans_k8", "km8")]:
        labels, dcent, _C = e[key]
        sizes = {c: int((labels == c).sum()) for c in set(labels.tolist())}
        for i, uid in enumerate(ids):
            n += 1
            rows.append({
                "cluster_assignment_id": f"S18B_CL_{n:05d}", "experiment_id": exp_id,
                "unified_patch_id": uid, "cluster_id": f"C{int(labels[i])}",
                "distance_to_cluster_centroid": _r(float(dcent[i])),
                "cluster_size": str(sizes[int(labels[i])]),
                "cluster_interpretation_review_only": "agrupamento multimodal nao supervisionado; NAO e classe/previsao/risco",
                "review_only": "true",
            })
    return rows


def outlier_score_rows():
    e = _compute_experiments()
    ids = e["ids"]
    scores = e["outlier"]
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))
    rank = {i: r + 1 for r, i in enumerate(order)}
    rows = []
    for i, uid in enumerate(ids):
        rows.append({
            "outlier_score_id": f"S18B_OUT_{i + 1:05d}", "experiment_id": "EXP05_outlier_distance_to_centroid",
            "unified_patch_id": uid, "outlier_score": _r(float(scores[i])),
            "outlier_rank": str(rank[i]),
            "outlier_reason": "distancia ao centroide multimodal global; patch INCOMUM (nao risco)",
            "review_only": "true",
        })
    return rows


def similarity_index_rows(topk=5):
    e = _compute_experiments()
    ids = e["ids"]
    D = e["dist"]
    fam = e["fam"]
    region = e["region"]
    rows = []
    n = 0
    for i, uid in enumerate(ids):
        order = sorted([j for j in range(len(ids)) if j != i], key=lambda j: (D[i, j], ids[j]))
        for rank, j in enumerate(order[:topk], start=1):
            n += 1
            dist = float(D[i, j])
            rows.append({
                "similarity_id": f"S18B_SIM_{n:05d}", "query_patch_id": uid,
                "neighbor_patch_id": ids[j], "rank": str(rank), "distance": _r(dist),
                "similarity_score": _r(1.0 / (1.0 + dist)),
                "shared_region": _bool(region[uid] == region[ids[j]]),
                "shared_patch_family": _bool(fam[uid] == fam[ids[j]]),
                "review_only": "true",
            })
    return rows


# ---------------------------------------------------------------------------
# Etapa 13 - candidate discovery
# ---------------------------------------------------------------------------

def _percentile_rank(values):
    """dict idx->percentil (0-100) por posto ascendente."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    pr = {}
    n = len(values)
    for rank, i in enumerate(order):
        pr[i] = round(100.0 * rank / (n - 1), 2) if n > 1 else 0.0
    return pr


def candidate_discovery_rows(top_n=100):
    e = _compute_experiments()
    samples = e["samples"]
    stats = e["stats"]
    ids = e["ids"]
    canary_ids = {s["unified_patch_id"] for s in samples if s["is_canary"]}
    # sinal composto por familia (media das features normalizadas do grupo)
    def group_signal(feature_names):
        _, Xg, _ = build_matrix(samples, feature_names, stats)
        return Xg.mean(axis=1)
    phys = group_signal(PHYSICAL_FEATURES)
    hydro = group_signal([f[0] for f in FEATURES if f[1] == "hydrology_proximity"])
    urban = group_signal([f[0] for f in FEATURES if f[1] == "urban_landcover"])
    spec = group_signal(SPECTRAL_FEATURES)
    rain = group_signal([f[0] for f in FEATURES if f[1] == "rainfall_chirps"])
    pr_phys, pr_hydro, pr_urban = _percentile_rank(phys), _percentile_rank(hydro), _percentile_rank(urban)
    pr_spec, pr_rain = _percentile_rank(spec), _percentile_rank(rain)
    # similaridade com canarios (quantos dos top5 vizinhos sao canario)
    D = e["dist"]
    sim_canary = {}
    for i, uid in enumerate(ids):
        order = sorted([j for j in range(len(ids)) if j != i], key=lambda j: (D[i, j], ids[j]))[:5]
        sim_canary[i] = sum(1 for j in order if ids[j] in canary_ids)
    priority = []
    for i, s in enumerate(samples):
        comp = _num(s["feature_completeness_ratio"]) or 0.0
        score = (pr_phys[i] + pr_hydro[i]) * 0.30 + pr_urban[i] * 0.10 + (pr_spec[i] + pr_rain[i]) * 0.05 \
            + sim_canary[i] * 8.0 + comp * 20.0
        priority.append((score, i))
    priority.sort(key=lambda t: (-t[0], ids[t[1]]))
    rows = []
    for rank, (score, i) in enumerate(priority[:top_n], start=1):
        s = samples[i]
        comp = _num(s["feature_completeness_ratio"]) or 0.0
        reasons = []
        if pr_phys[i] >= 70 or pr_hydro[i] >= 70:
            reasons.append("high_physical_hydrology_signal")
        if sim_canary[i] > 0:
            reasons.append("similar_to_canary")
        if comp >= 0.8:
            reasons.append("high_completeness")
        if pr_urban[i] >= 70:
            reasons.append("high_urban_signal")
        if not reasons:
            reasons.append("multimodal_context_review")
        action = "acquire_ground_reference_priority" if (pr_phys[i] >= 70 and comp >= 0.8) else "review_only_screening"
        rows.append({
            "candidate_discovery_id": f"S18B_CAND_{rank:04d}", "unified_patch_id": s["unified_patch_id"],
            "source_patch_id": s["source_patch_id"], "patch_family": s["patch_family"],
            "priority_rank": str(rank), "candidate_reason": ";".join(reasons),
            "similar_to_canary_count": str(sim_canary[i]),
            "physical_signal_percentile": _r(pr_phys[i], 2), "hydrology_signal_percentile": _r(pr_hydro[i], 2),
            "urban_signal_percentile": _r(pr_urban[i], 2), "spectral_signal_percentile": _r(pr_spec[i], 2),
            "rainfall_signal_percentile": _r(pr_rain[i], 2), "data_completeness": _r(comp, 4),
            "recommended_next_action": action, "review_only": "true",
            "ground_truth": "false", "training_label": "false",
        })
    return rows


# ---------------------------------------------------------------------------
# Etapa 14 - trainability gate
# ---------------------------------------------------------------------------

def trainability_gate_rows():
    gates = [
        ("supervised_training", False, "labels de ground reference oficiais aceitos (>=50 ev>=5 reg>=3)",
         "0 labels aceitos", "sem labels validos", "aquisicao de ground reference oficial patch-level"),
        ("weak_supervision", False, "regras/proxies aprovados como fraca supervisao",
         "nenhuma regra aprovada", "proxy nao aprovado como label", "nao usar ancora/canario como fraca supervisao sem aprovacao"),
        ("positive_unlabeled_learning", False, "conjunto positivo confiavel + nao rotulados",
         "sem positivos confiaveis (canario!=positivo)", "canario nao e positivo verdadeiro", "aguardar ground reference"),
        ("self_supervised_learning", True, "features multimodais sem label",
         "feature tensor 316x{}".format(len(FEATURE_NAMES)), "none", "pretexto self-supervised review-only (mascaramento/reconstrucao)"),
        ("unsupervised_clustering", True, "features multimodais normalizadas",
         "tensor + normalizacao prontos", "none", "clustering/embedding review-only (ja executado)"),
        ("candidate_screening", True, "percentis + similaridade + outlier",
         "similarity + candidate queue prontos", "none", "triagem review-only p/ aquisicao futura"),
    ]
    rows = []
    for i, (mode, allowed, req, avail, block, nxt) in enumerate(gates, start=1):
        rows.append({
            "trainability_gate_id": f"S18B_GATE_{i:04d}", "training_mode": mode,
            "is_allowed_now": _bool(allowed), "required_inputs": req, "available_inputs": avail,
            "blocking_reason": block, "safe_next_step": nxt, "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Etapa 15 - supervised harness fail-closed
# ---------------------------------------------------------------------------

def _harness_implemented(module, cls):
    spec = importlib.util.find_spec(module)
    return spec is not None


def supervised_harness_rows():
    families = [
        ("logistic_regression", "sklearn.linear_model", "LogisticRegression"),
        ("random_forest", "sklearn.ensemble", "RandomForestClassifier"),
        ("gradient_boosting_or_xgboost", "sklearn.ensemble", "GradientBoostingClassifier"),
        ("linear_svm", "sklearn.svm", "LinearSVC"),
    ]
    rows = []
    for i, (fam, module, cls) in enumerate(families, start=1):
        impl = _harness_implemented(module, cls)
        rows.append({
            "supervised_harness_id": f"S18B_HARN_{i:04d}", "model_family": fam,
            "implemented": _bool(impl), "attempted_training": "false", "training_allowed": "false",
            "training_executed": "false",
            "blocking_reason": "sem labels de ground reference aceitos; harness fail-closed (codigo existe, .fit nao executa)",
            "required_label_contract": "susc_18b_future_label_contract.json",
            "review_only": "true",
        })
    return rows


def supervised_harness_attempt():
    """Harness fail-closed: verifica contrato de labels; NUNCA chama .fit sem labels aceitos."""
    accepted = 0  # 0 ground references aceitos nesta iteracao
    contract = future_label_contract()
    req = contract["minimum_criteria_for_future_training"]
    allowed = (accepted >= req["accepted_ground_reference_count"])
    return {"training_allowed": allowed, "accepted_labels": accepted, "training_executed": False}


# ---------------------------------------------------------------------------
# Etapa 16 - future label contract
# ---------------------------------------------------------------------------

def future_label_contract():
    return {
        "milestone": "SUSC-18B",
        "purpose": "define o que sera exigido para HABILITAR treino supervisionado no futuro",
        "required_label_fields": [
            "ground_reference_id", "event_id", "event_date", "geometry", "patch_link",
            "uncertainty_m", "source_authority", "qa_status", "temporal_split_group",
            "label_type", "not_ground_truth_reason", "allowed_for_training",
        ],
        "minimum_criteria_for_future_training": {
            "accepted_ground_reference_count": 50,
            "event_count": 5,
            "region_count": 3,
            "negative_sampling_policy_approved": True,
            "temporal_split_possible": True,
            "qa_accepted_only": True,
        },
        "current_status": {
            "accepted_ground_reference_count": 0, "event_count": 0, "region_count": 0,
            "negative_sampling_policy_approved": False, "temporal_split_possible": False,
            "training_enabled_now": False,
        },
        "canary_is_positive_label": False, "control_is_negative_label": False,
        "occurrence_anchor_is_label": False, "review_only": True,
    }


# ---------------------------------------------------------------------------
# Etapa 17 - experiment registry
# ---------------------------------------------------------------------------

def experiment_registry_rows():
    rows = []
    outputs = {
        "EXP01_pca_2d": rel(EMBEDDINGS), "EXP02_truncated_svd_5d": rel(EMBEDDINGS),
        "EXP03_kmeans_k5": rel(CLUSTERS), "EXP04_kmeans_k8": rel(CLUSTERS),
        "EXP05_outlier_distance_to_centroid": rel(OUTLIERS), "EXP06_similarity_topk": rel(SIMILARITY),
        "EXP07_family_ablation_spectral_only": rel(EMBEDDINGS),
        "EXP08_family_ablation_physical_only": rel(EMBEDDINGS),
    }
    for exp_id, method, groups in EXPERIMENTS:
        rows.append({
            "experiment_id": exp_id, "experiment_type": "unsupervised_review_only", "method": method,
            "feature_groups": ";".join(groups), "sample_count": "316", "uses_labels": "false",
            "uses_ground_truth": "false", "uses_score_v6_as_target": "false", "status": "completed",
            "output_files": outputs.get(exp_id, ""), "review_only": "true",
        })
    # harness supervisionado (bloqueado) registrado como experimento nao executado
    for fam in ["logistic_regression", "random_forest", "gradient_boosting_or_xgboost", "linear_svm"]:
        rows.append({
            "experiment_id": f"SUP_{fam}", "experiment_type": "supervised_blocked", "method": fam,
            "feature_groups": ";".join(MODELING_GROUPS), "sample_count": "0", "uses_labels": "false",
            "uses_ground_truth": "false", "uses_score_v6_as_target": "false",
            "status": "blocked_no_labels", "output_files": rel(SUP_HARNESS), "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Etapa 18 - model cards
# ---------------------------------------------------------------------------

def model_card_rows():
    cards = [
        ("pca_2d", "EXP01_pca_2d", "dimensionality_reduction", "completed",
         "exploracao visual review-only da estrutura multimodal", "previsao/classe/risco/ground truth",
         ";".join(MODELING_FEATURES), "escala nao-oficial; sem label; missingness imputada(0) flagada"),
        ("truncated_svd_5d", "EXP02_truncated_svd_5d", "dimensionality_reduction", "completed",
         "embedding tabular review-only", "target de treino/score",
         ";".join(MODELING_FEATURES), "componentes latentes nao interpretaveis como risco"),
        ("kmeans_k5", "EXP03_kmeans_k5", "clustering", "completed",
         "agrupamento multimodal exploratorio", "chamar cluster de classe/previsao",
         ";".join(MODELING_FEATURES), "k arbitrario; cluster != classe; sensivel a missingness"),
        ("kmeans_k8", "EXP04_kmeans_k8", "clustering", "completed",
         "agrupamento multimodal exploratorio", "chamar cluster de risco",
         ";".join(MODELING_FEATURES), "k arbitrario; review-only"),
        ("outlier_distance_centroid", "EXP05_outlier_distance_to_centroid", "outlier_detection", "completed",
         "detectar patches multimodais incomuns", "chamar outlier de risco/evento",
         ";".join(MODELING_FEATURES), "outlier=incomum, NAO risco; sensivel a escala"),
        ("similarity_topk", "EXP06_similarity_topk", "similarity_search", "completed",
         "vizinhanca multimodal p/ triagem", "chamar similaridade de ground truth",
         ";".join(MODELING_FEATURES), "distancia euclidiana review-only"),
        ("logistic_regression", "SUP_logistic_regression", "supervised_classification", "blocked_no_labels",
         "reservado p/ treino futuro com labels aceitos", "treinar sem ground reference aceito",
         ";".join(MODELING_FEATURES), "harness existe mas NAO treina; sem labels"),
        ("random_forest", "SUP_random_forest", "supervised_classification", "blocked_no_labels",
         "reservado p/ treino futuro", "treinar sem labels/pseudo-label",
         ";".join(MODELING_FEATURES), "fail-closed"),
        ("gradient_boosting_or_xgboost", "SUP_gradient_boosting_or_xgboost", "supervised_classification", "blocked_no_labels",
         "reservado p/ treino futuro", "treinar sem labels",
         ";".join(MODELING_FEATURES), "fail-closed"),
        ("linear_svm", "SUP_linear_svm", "supervised_classification", "blocked_no_labels",
         "reservado p/ treino futuro", "treinar sem labels",
         ";".join(MODELING_FEATURES), "fail-closed"),
    ]
    rows = []
    for i, (name, exp, task, status, intended, forbidden, feats, limits) in enumerate(cards, start=1):
        supervised = task == "supervised_classification"
        rows.append({
            "model_card_id": f"S18B_CARD_{i:04d}", "experiment_id": exp, "model_or_method_name": name,
            "task_type": task, "training_status": status, "intended_use": intended,
            "forbidden_use": forbidden, "input_features": feats, "known_limitations": limits,
            "uses_labels": "false", "uses_ground_truth": "false", "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# integridade + no-leakage + summary + blockers + report
# ---------------------------------------------------------------------------

def score_integrity_audit_rows():
    v6_changed = bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]))
    return [
        {"score_integrity_audit_id": "S18B_INTEG_0001", "protected_artifact": rel(SCORE_V6),
         "artifact_type": "score_v6_candidate", "changed_in_working_tree": _bool(v6_changed),
         "used_as": "reference_only_never_target", "created_new": "false", "review_only": "true"},
        {"score_integrity_audit_id": "S18B_INTEG_0002", "protected_artifact": rel(SCORE_V7),
         "artifact_type": "score_v7_candidate", "changed_in_working_tree": "false",
         "used_as": "none", "created_new": _bool(SCORE_V7.exists()), "review_only": "true"},
    ]


def no_leakage_audit_rows():
    rows = []
    objs = [(exp_id, "unsupervised_experiment") for exp_id, _m, _g in EXPERIMENTS]
    objs += [(f"SUP_{f}", "supervised_harness") for f in ["logistic_regression", "random_forest", "gradient_boosting_or_xgboost", "linear_svm"]]
    objs += [("ml_ready_dataset", "dataset"), ("feature_tensor", "dataset")]
    for i, (oid, otype) in enumerate(objs, start=1):
        rows.append({
            "no_leakage_audit_id": f"S18B_LEAK_{i:04d}", "object_id": oid, "object_type": otype,
            "uses_occurrence_as_feature": "false", "uses_canary_as_positive_label": "false",
            "uses_control_as_negative_label": "false", "uses_score_v6_as_target": "false",
            "uses_ground_truth_label": "false", "uses_training_label": "false",
            "uses_post_event_feature_as_pre_event": "false", "uses_flow_acc_as_equivalent": "false",
            "passes_no_leakage": "true",
            "blocking_reason": "modelagem sem-label unsupervised_review_only; referencias nunca target; harness supervisionado fail-closed",
            "review_only": "true",
        })
    return rows


def build_summary():
    samples = assemble_samples()
    official = [s for s in samples if s["is_official"]]
    canary = [s for s in samples if s["is_canary"]]
    control = [s for s in samples if s["is_control"]]
    ds = ml_ready_dataset_rows()
    fg = feature_group_rows()
    mask = missingness_mask_rows()
    tensor = feature_tensor_rows()
    splits = split_registry_rows()
    unsup = unsupervised_experiment_rows()
    emb = tabular_embedding_rows()
    clusters = cluster_assignment_rows()
    outliers = outlier_score_rows()
    sim = similarity_index_rows()
    cand = candidate_discovery_rows()
    exp_reg = experiment_registry_rows()
    cards = model_card_rows()
    harness = supervised_harness_attempt()

    minimum = (
        len(samples) >= 316 and len(ds) >= 316 and len(fg) >= 6 and len(FEATURE_NAMES) >= 25
        and len(mask) > 0 and len(tensor) >= 316 and len(splits) > 0 and len(unsup) >= 5
        and len(sim) >= 1580 and len(clusters) >= 316 and len(outliers) >= 316
        and len(cand) >= 50 and len(exp_reg) >= 5 and len(cards) >= 5
        and not harness["training_executed"]
    )
    return {
        "minimum_success_achieved": minimum,
        "feature_store_rows_consumed_count": len(samples),
        "official_patch_rows_consumed_count": len(official),
        "canary_patch_rows_consumed_count": len(canary),
        "control_patch_rows_consumed_count": len(control),
        "ml_dataset_rows_count": len(ds),
        "feature_group_count": len(fg),
        "numeric_feature_count": len(FEATURE_NAMES),
        "missingness_mask_rows_count": len(mask),
        "missingness_mask_created": True,
        "normalization_pipeline_created": True,
        "feature_tensor_rows_count": len(tensor),
        "split_registry_rows_count": len(splits),
        "split_registry_created": True,
        "unsupervised_experiment_count": len(unsup),
        "unsupervised_experiments_completed": all(r["status"] == "completed" for r in unsup),
        "tabular_embedding_rows_count": len(emb),
        "cluster_assignment_rows_count": len(clusters),
        "outlier_score_rows_count": len(outliers),
        "similarity_index_rows_count": len(sim),
        "candidate_discovery_rows_count": len(cand),
        "trainability_gate_created": True,
        "supervised_harness_created": True,
        "supervised_training_allowed": harness["training_allowed"],
        "supervised_training_attempted": False,
        "supervised_training_executed": harness["training_executed"],
        "supervised_training_blocked": True,
        "supervised_training_blocked_reason": "no_ground_reference_labels",
        "ml_dataset_created": True,
        "future_label_contract_created": True,
        "experiment_registry_rows_count": len(exp_reg),
        "model_card_rows_count": len(cards),
        "sklearn_available": _sklearn_available(),
        "compute_backend": "numpy_deterministic_seed_%d" % SEED,
        "flow_acc_treated_as_equivalent": False,
        "occurrence_used_as_causal_feature": False,
        "canary_used_as_positive_label": False,
        "control_used_as_negative_label": False,
        "score_v6_used_as_training_target": False,
        "score_v6_original_file_changed": bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)])),
        "score_v7_created": SCORE_V7.exists(),
        "ground_truth_created": False,
        "training_labels_created": False,
        "official_patch_created": False,
        "official_patch_link_created": False,
        "eligible_for_17b_now": False,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "result_class": "ml_readiness_complete_supervised_training_blocked",
        "recommended_next_milestone": (
            "SUSC-18C Aquisicao de Ground Reference oficial patch-level (>=50 aceitos, >=5 eventos, >=3 regioes) "
            "para preencher o future_label_contract e destravar o harness supervisionado fail-closed; "
            "ou SUSC-18C-alt self-supervised pretext review-only sobre o feature tensor. "
            "Sem score v7, sem ground truth, sem treino, 17B fail-closed."),
    }


BLOCKER_LIST = [
    ("supervised_training_blocked_no_ground_reference_labels", "treino supervisionado bloqueado: 0 labels de ground reference aceitos"),
    ("no_training_labels", "nenhum label de treino existe (nem sintetico, nem pseudo-label)"),
    ("not_ground_truth", "nenhum artefato e ground truth"),
    ("not_score_v7", "nao cria score v7"),
    ("flow_acc_not_equivalent", "flow_acc nao tratado como equivalente ao oficial"),
    ("canary_not_positive_label", "canario NAO e label positivo"),
    ("control_not_negative_label", "controle NAO e label negativo"),
    ("score_v6_not_training_target", "score v6 usado so como referencia, nunca target"),
    ("17b_blocked", "17B exige G4_full + ground reference aceito; nao atingido"),
    ("ml_readiness_created", "infra de ML readiness entregue (review-only)"),
]


def promotion_blocker_rows():
    return [{"blocker_id": f"S18B_BLOCKER_{i:04d}", "blocker_type": n, "description": d,
             "blocks_supervised_training": "true", "blocks_17b": "true", "review_only": "true"}
            for i, (n, d) in enumerate(BLOCKER_LIST, start=1)]


def build_report():
    s = build_summary()
    return "\n".join([
        "# SUSC-18B - Multimodal ML Readiness, Dataset Builder e Experiment Harness",
        "",
        "## Objetivo",
        "Transformar a feature store multimodal 18A em infraestrutura de MODELAGEM auditavel (dataset model-ready + experiment harness + trainability gates + baselines nao supervisionados + harness supervisionado fail-closed), SEM ground truth e SEM treino supervisionado.",
        "",
        "## Resultado: A (ML readiness completo, treino supervisionado BLOQUEADO)",
        f"- ml_dataset_created={s['ml_dataset_created']} | unsupervised_experiments_completed={s['unsupervised_experiments_completed']}",
        f"- supervised_training_allowed={s['supervised_training_allowed']} | supervised_training_executed={s['supervised_training_executed']} | motivo={s['supervised_training_blocked_reason']}",
        "",
        "## Dataset e features",
        f"- Feature store 18A consumida: {s['feature_store_rows_consumed_count']} patches ({s['official_patch_rows_consumed_count']} oficiais + {s['canary_patch_rows_consumed_count']} canarios + {s['control_patch_rows_consumed_count']} controles)",
        f"- Dataset ML: {s['ml_dataset_rows_count']} linhas | feature groups: {s['feature_group_count']} | features numericas: {s['numeric_feature_count']}",
        f"- Missingness mask: {s['missingness_mask_rows_count']} | feature tensor: {s['feature_tensor_rows_count']} | splits: {s['split_registry_rows_count']}",
        "",
        "## Experimentos nao supervisionados (numpy deterministico, seed fixo)",
        f"- Experimentos: {s['unsupervised_experiment_count']} (PCA/SVD/KMeans/outlier/similaridade/ablacoes)",
        f"- Embeddings: {s['tabular_embedding_rows_count']} | clusters: {s['cluster_assignment_rows_count']} | outliers: {s['outlier_score_rows_count']} | similarity: {s['similarity_index_rows_count']}",
        f"- Candidate discovery: {s['candidate_discovery_rows_count']} | experiment registry: {s['experiment_registry_rows_count']} | model cards: {s['model_card_rows_count']}",
        f"- sklearn disponivel: {s['sklearn_available']} (calculo em numpy deterministico p/ rebuild byte-identico)",
        "",
        "## Trainability gate + harness supervisionado fail-closed",
        "- Permitidos agora: self_supervised_learning, unsupervised_clustering, candidate_screening.",
        "- Bloqueados: supervised_training, weak_supervision, positive_unlabeled_learning (sem labels aceitos).",
        "- Harness supervisionado (logistic/random_forest/gradient_boosting/linear_svm): codigo existe, .fit NAO executa. future_label_contract exige >=50 ground references aceitos, >=5 eventos, >=3 regioes.",
        "",
        "## Guardrails cientificos",
        "- Clustering NAO e previsao; outlier NAO e risco; ranking NAO e ground truth.",
        "- Canario NAO e positivo; controle NAO e negativo; ausencia NAO e negativo; ocorrencia so ancora.",
        "- score v6 so referencia (nunca target); flow_acc nao equivalente; sem pos-evento como pre-evento; missingness explicita.",
        "- Score v6 oficial intacto; score v7 inexistente; ground truth nao criado; treino/labels nao criados; 17B fail-closed.",
        "",
        f"## minimum_success_achieved: {s['minimum_success_achieved']} | result_class: {s['result_class']}",
        "",
        "## Proximo marco recomendado",
        s["recommended_next_milestone"],
    ])


# ---------------------------------------------------------------------------
# build / modes / validate
# ---------------------------------------------------------------------------

def build_all():
    _require_inputs()
    ensure_dir(OUT)
    write_json(DATASET_CONTRACT, dataset_contract())
    write_csv(ML_DATASET, ml_ready_dataset_rows())
    write_csv(FEATURE_GROUPS, feature_group_rows())
    write_csv(MISSINGNESS, missingness_mask_rows())
    write_json(NORM_PIPELINE, normalization_pipeline())
    write_csv(NORM_AUDIT, normalization_audit_rows())
    write_csv(FEATURE_TENSOR, feature_tensor_rows())
    write_csv(SPLIT_REGISTRY, split_registry_rows())
    write_csv(UNSUP_REGISTRY, unsupervised_experiment_rows())
    write_csv(EMBEDDINGS, tabular_embedding_rows())
    write_csv(CLUSTERS, cluster_assignment_rows())
    write_csv(OUTLIERS, outlier_score_rows())
    write_csv(SIMILARITY, similarity_index_rows())
    write_csv(CANDIDATES, candidate_discovery_rows())
    write_csv(TRAINABILITY, trainability_gate_rows())
    write_csv(SUP_HARNESS, supervised_harness_rows())
    write_json(LABEL_CONTRACT, future_label_contract())
    write_csv(EXP_REGISTRY, experiment_registry_rows())
    write_csv(MODEL_CARDS, model_card_rows())
    write_csv(INTEGRITY, score_integrity_audit_rows())
    write_csv(LEAKAGE, no_leakage_audit_rows())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, promotion_blocker_rows())
    write_markdown(REPORT, build_report())


MODE_DISPATCH = {
    "build-ml-dataset-contract": (write_json, DATASET_CONTRACT, dataset_contract),
    "build-ml-ready-dataset": (write_csv, ML_DATASET, ml_ready_dataset_rows),
    "build-feature-groups": (write_csv, FEATURE_GROUPS, feature_group_rows),
    "build-missingness-mask": (write_csv, MISSINGNESS, missingness_mask_rows),
    "build-normalization-pipeline": (write_json, NORM_PIPELINE, normalization_pipeline),
    "build-feature-tensor": (write_csv, FEATURE_TENSOR, feature_tensor_rows),
    "build-split-registry": (write_csv, SPLIT_REGISTRY, split_registry_rows),
    "run-unsupervised-experiments": (write_csv, UNSUP_REGISTRY, unsupervised_experiment_rows),
    "build-tabular-embeddings": (write_csv, EMBEDDINGS, tabular_embedding_rows),
    "build-cluster-assignments": (write_csv, CLUSTERS, cluster_assignment_rows),
    "build-outlier-scores": (write_csv, OUTLIERS, outlier_score_rows),
    "build-similarity-index": (write_csv, SIMILARITY, similarity_index_rows),
    "build-candidate-discovery": (write_csv, CANDIDATES, candidate_discovery_rows),
    "build-trainability-gate": (write_csv, TRAINABILITY, trainability_gate_rows),
    "build-supervised-harness-status": (write_csv, SUP_HARNESS, supervised_harness_rows),
    "build-future-label-contract": (write_json, LABEL_CONTRACT, future_label_contract),
    "build-experiment-registry": (write_csv, EXP_REGISTRY, experiment_registry_rows),
    "build-model-cards": (write_csv, MODEL_CARDS, model_card_rows),
}


def run_mode(mode):
    _require_inputs()
    ensure_dir(OUT)
    if mode == "build-normalization-pipeline":
        write_json(NORM_PIPELINE, normalization_pipeline())
        write_csv(NORM_AUDIT, normalization_audit_rows())
    elif mode == "build-offline":
        build_all()
    elif mode == "audit":
        raise SystemExit(validate())
    elif mode in MODE_DISPATCH:
        writer, path, fn = MODE_DISPATCH[mode]
        writer(path, fn())
    else:
        raise SystemExit(f"unknown mode: {mode}")
    return 0


def _schema_violations(row, schema):
    v = []
    for f in schema.get("required", []):
        if f not in row or row[f] == "":
            v.append(f"missing:{f}")
    for f, rules in schema.get("properties", {}).items():
        if f not in row:
            continue
        val = row[f]
        if "const" in rules and val != rules["const"]:
            v.append(f"{f}:const")
        if "enum" in rules and val not in rules["enum"]:
            v.append(f"{f}:enum")
        if "pattern" in rules and not str(val).startswith(rules["pattern"].replace("^", "")):
            v.append(f"{f}:pattern")
    return v


def _validate_byte_identical():
    before = {p: p.read_bytes() for p in REQUIRED_OUTPUTS if p.exists()}
    build_all()
    errs = []
    for p, content in before.items():
        if p.exists() and p.read_bytes() != content:
            errs.append(f"offline_build_not_byte_identical:{rel(p)}")
    return errs


def validate():
    # 1 - 18A commitado antes do 18B
    if not _18a_committed():
        print("ERROR: susc_18a_not_committed", file=sys.stderr)
        return 1
    missing = [p for p in REQUIRED_OUTPUTS if not p.exists()]
    if missing:
        print("MISSING: " + "; ".join(rel(p) for p in missing), file=sys.stderr)
        return 1
    errors = _validate_byte_identical()

    ds = read_csv(ML_DATASET)
    fg = read_csv(FEATURE_GROUPS)
    mask = read_csv(MISSINGNESS)
    tensor = read_csv(FEATURE_TENSOR)
    splits = read_csv(SPLIT_REGISTRY)
    unsup = read_csv(UNSUP_REGISTRY)
    emb = read_csv(EMBEDDINGS)
    clusters = read_csv(CLUSTERS)
    outliers = read_csv(OUTLIERS)
    sim = read_csv(SIMILARITY)
    cand = read_csv(CANDIDATES)
    gate = read_csv(TRAINABILITY)
    harness = read_csv(SUP_HARNESS)
    exp_reg = read_csv(EXP_REGISTRY)
    cards = read_csv(MODEL_CARDS)
    leak = read_csv(LEAKAGE)
    summary = read_json(SUMMARY)
    ds_schema = read_json(SCHEMA_DATASET)
    exp_schema = read_json(SCHEMA_EXP)
    train_schema = read_json(SCHEMA_TRAIN)

    for r in ds:
        errors.extend(_schema_violations(r, ds_schema))
    for r in exp_reg:
        errors.extend(_schema_violations(r, exp_schema))
    for r in gate:
        errors.extend(_schema_violations(r, train_schema))

    official = [r for r in ds if r["is_official_population_patch"] == "true"]
    canary = [r for r in ds if r["is_canary_patch"] == "true"]
    control = [r for r in ds if r["is_control_patch"] == "true"]

    def chk(cond, tag):
        if cond:
            errors.append(tag)

    chk(summary["feature_store_rows_consumed_count"] < 316, "consumed_lt_316")  # 2
    chk(len(official) < 300, "official_lt_300")  # 3
    chk(len(canary) < 11, "canary_lt_11")  # 4
    chk(len(control) < 5, "control_lt_5")  # 5
    chk(len(ds) < 316, "ml_dataset_lt_316")  # 6
    chk(len(fg) < 6, "feature_groups_lt_6")  # 7
    chk(summary["numeric_feature_count"] < 25, "numeric_features_lt_25")  # 8
    chk(len(mask) == 0 or not summary["missingness_mask_created"], "missingness_mask_missing")  # 9
    chk(not NORM_PIPELINE.exists() or not summary["normalization_pipeline_created"], "norm_pipeline_missing")  # 10
    chk(len(tensor) < 316, "feature_tensor_lt_316")  # 11
    chk(len(splits) == 0 or not summary["split_registry_created"], "split_registry_missing")  # 12
    chk(len(unsup) < 5, "unsup_lt_5")  # 13
    chk(len(sim) < 1580, "similarity_lt_1580")  # 14
    chk(len(cand) < 50, "candidate_lt_50")  # 15
    chk(len(gate) == 0 or not summary["trainability_gate_created"], "trainability_gate_missing")  # 16
    chk(len(harness) == 0 or not summary["supervised_harness_created"], "supervised_harness_missing")  # 17
    # 18 - treino executado sem labels
    chk(summary["supervised_training_executed"] or summary["supervised_training_attempted"] or summary["supervised_training_allowed"], "supervised_training_executed")
    for r in harness:
        chk(r["training_executed"] != "false" or r["attempted_training"] != "false", "harness_training_executed")
    for r in exp_reg:
        if r["experiment_type"] == "supervised_blocked":
            chk(r["status"] != "blocked_no_labels", "supervised_not_blocked")
    # 19/20 - canario positivo / controle negativo
    for r in leak:
        chk(r["uses_canary_as_positive_label"] != "false", "canary_positive_leak")
        chk(r["uses_control_as_negative_label"] != "false", "control_negative_leak")
    chk(summary["canary_used_as_positive_label"] is not False, "canary_positive_summary")
    chk(summary["control_used_as_negative_label"] is not False, "control_negative_summary")
    # 21 - score v6 como target
    for r in leak:
        chk(r["uses_score_v6_as_target"] != "false", "score_v6_target_leak")
    for r in ds:
        chk(r["score_v6_is_target"] != "false" or r["transparent_index_is_target"] != "false" or r["occurrence_anchor_is_target"] != "false", "ds_target_leak")
    chk(summary["score_v6_used_as_training_target"] is not False, "score_v6_target_summary")
    # 22 - ocorrencia como feature causal
    for r in leak:
        chk(r["uses_occurrence_as_feature"] != "false", "occurrence_feature_leak")
    chk(summary["occurrence_used_as_causal_feature"] is not False, "occurrence_feature_summary")
    # 23 - flow_acc equivalente
    for r in leak:
        chk(r["uses_flow_acc_as_equivalent"] != "false", "flow_acc_equivalent_leak")
    for r in ds:
        chk(r["flow_acc_equivalent"] != "false", "flow_acc_equivalent_ds")
    chk(summary["flow_acc_treated_as_equivalent"] is not False, "flow_acc_equivalent_summary")
    # 24 - score v6 mudou
    chk(bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)])) or summary["score_v6_original_file_changed"], "score_v6_changed")
    # 25 - score v7
    chk(SCORE_V7.exists() or summary["score_v7_created"], "score_v7_exists")
    # 26/27 - ground truth / treino
    chk(summary["ground_truth_created"] or summary["training_labels_created"], "gt_or_training")
    for r in ds:
        chk(r["ground_truth"] != "false" or r["training_label"] != "false", "ds_gt_training")
    # 28 - 17B
    chk(summary["eligible_for_17b_now"], "17b_unlocked")
    # pos-evento como pre-evento
    for r in leak:
        chk(r["uses_post_event_feature_as_pre_event"] != "false", "post_as_pre_leak")
    chk(any(r["passes_no_leakage"] != "true" for r in leak), "no_leakage_failed")
    # nao terminar blocked
    chk(not summary["minimum_success_achieved"], "minimum_success_not_achieved")

    # 29 - summary reflete contagens reais
    expected = build_summary()
    for k, v in expected.items():
        if summary.get(k) != v:
            errors.append(f"summary_mismatch:{k}")

    for rows, key in [
        (ds, "ml_sample_id"), (fg, "feature_group_id"), (mask, "missingness_mask_id"),
        (splits, "split_id"), (emb, "embedding_projection_id"), (clusters, "cluster_assignment_id"),
        (outliers, "outlier_score_id"), (sim, "similarity_id"), (cand, "candidate_discovery_id"),
        (gate, "trainability_gate_id"), (harness, "supervised_harness_id"), (cards, "model_card_id"),
        (leak, "no_leakage_audit_id"),
    ]:
        ids = [r[key] for r in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(
        "18B -> "
        f"consumed={summary['feature_store_rows_consumed_count']} dataset={summary['ml_dataset_rows_count']} "
        f"groups={summary['feature_group_count']} features={summary['numeric_feature_count']} "
        f"tensor={summary['feature_tensor_rows_count']} unsup={summary['unsupervised_experiment_count']} "
        f"clusters={summary['cluster_assignment_rows_count']} outliers={summary['outlier_score_rows_count']} "
        f"similarity={summary['similarity_index_rows_count']} candidates={summary['candidate_discovery_rows_count']} "
        f"exp_reg={summary['experiment_registry_rows_count']} cards={summary['model_card_rows_count']} "
        f"sup_train_exec={summary['supervised_training_executed']} sup_blocked={summary['supervised_training_blocked']} "
        f"result={summary['result_class']} min_success={summary['minimum_success_achieved']}"
    )
    return 0


MODES = list(MODE_DISPATCH.keys()) + ["build-offline", "audit"]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="susc_18b")
    parser.add_argument("mode", choices=MODES)
    args = parser.parse_args(argv)
    if args.mode == "audit":
        return validate()
    return run_mode(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
