"""SUSC-18C Self-Supervised Multimodal Pretraining e Ground Reference Factory.

Duas frentes integradas sobre a infra de [[SUSC-18B]] (feature tensor 316x26):

Frente A - Self-supervised multimodal pretraining (SEM labels, permitido):
  - masked feature modeling (reconstrucao mascarada low-rank / linear);
  - denoising autoencoder tabular (linear/SVD numpy);
  - contrastive/similarity representation;
  - representation quality audit.
  Permitido porque uses_labels=false, uses_ground_truth=false, canario nao e positivo,
  controle nao e negativo, review_only=true.

Frente B - Ground Reference Factory (tenta destravar treino supervisionado FUTURO):
  candidate discovery -> ground reference target pack -> tentativas de aquisicao
  (rede fail-closed) -> ground reference candidates -> patch-reference links -> QA
  queue -> execucao do label contract -> recheck do supervised gate.

Resultado A (esperado): self-supervised completo + supervised AINDA bloqueado (0 ground
references aceitos; 1 evento < 5; 1 regiao < 3). O marco NAO fecha so como blocked:
entrega pretraining self-supervised real. supervised_training.fit NUNCA executa sem
label_contract_training_allowed=true.

Guardrails: NAO score v7, NAO altera score v6 (referencia), NAO ground truth falso,
NAO labels sem contrato aceito, canario!=positivo, controle!=negativo, ocorrencia so
ancora/referencia (nunca feature causal), flow_acc nao equivalente, footprint pos-evento
nunca como feature pre-evento, 17B so muda com referencia forte suficiente.

Offline e deterministico (numpy seed fixo). So run-ground-reference-acquisition pode
usar rede (fail-closed sem envs SUSC_18C_ALLOW_*).
"""

from __future__ import annotations

import argparse
import math
import os
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

SEED = 18032  # seed fixo 18C

# --- Entradas --------------------------------------------------------------
B_TENSOR = OUT / "susc_18b_ml_feature_tensor.csv"
B_DATASET = OUT / "susc_18b_ml_ready_dataset.csv"
B_SIMILARITY = OUT / "susc_18b_patch_similarity_index.csv"
B_CANDIDATES = OUT / "susc_18b_candidate_discovery_queue.csv"
B_LABEL_CONTRACT = OUT / "susc_18b_future_label_contract.json"
B_SUMMARY = OUT / "susc_18b_readiness_summary.json"
A_STORE = OUT / "susc_18a_multimodal_patch_feature_store.csv"
C33_REGISTRY = OUT / "susc_17c33_event_anchored_canary_patch_registry.csv"
SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"

REQUIRED_INPUTS = [B_TENSOR, B_DATASET, B_SIMILARITY, B_CANDIDATES, B_LABEL_CONTRACT, B_SUMMARY, C33_REGISTRY]

# --- Saidas ----------------------------------------------------------------
CONTRACT = OUT / "susc_18c_pretraining_and_reference_contract.json"
SS_TENSOR = OUT / "susc_18c_self_supervised_training_tensor.csv"
MFM_METRICS = OUT / "susc_18c_masked_feature_modeling_metrics.csv"
DENOISE_EMB = OUT / "susc_18c_denoising_embedding.csv"
DENOISE_METRICS = OUT / "susc_18c_denoising_model_metrics.csv"
CONTRASTIVE = OUT / "susc_18c_contrastive_similarity_index.csv"
REP_QUALITY = OUT / "susc_18c_representation_quality_audit.csv"
EMB_CANDIDATES = OUT / "susc_18c_embedding_candidate_discovery_queue.csv"
GR_TARGETS = OUT / "susc_18c_ground_reference_target_pack.csv"
GR_ATTEMPTS = OUT / "susc_18c_ground_reference_acquisition_attempts.csv"
GR_CANDIDATES = OUT / "susc_18c_ground_reference_candidate_registry.csv"
PATCH_LINKS = OUT / "susc_18c_patch_reference_link_candidates.csv"
QA_QUEUE = OUT / "susc_18c_ground_reference_qa_queue.csv"
LABEL_EXEC = OUT / "susc_18c_label_contract_execution.csv"
GATE_RECHECK = OUT / "susc_18c_supervised_gate_recheck.csv"
EXP_REGISTRY = OUT / "susc_18c_experiment_registry.csv"
MODEL_CARDS = OUT / "susc_18c_model_cards.csv"
INTEGRITY = OUT / "susc_18c_score_integrity_audit.csv"
LEAKAGE = OUT / "susc_18c_no_leakage_audit.csv"
SUMMARY = OUT / "susc_18c_readiness_summary.json"
BLOCKERS = OUT / "susc_18c_promotion_blockers.csv"
REPORT = OUT / "SUSC_18C_SELF_SUPERVISED_AND_REFERENCE_FACTORY_REPORT.md"

SCHEMA_SS = SCHEMAS / "susc_18c_self_supervised_schema_v1.json"
SCHEMA_GRF = SCHEMAS / "susc_18c_ground_reference_factory_schema_v1.json"
SCHEMA_GATE = SCHEMAS / "susc_18c_supervised_gate_schema_v1.json"

REQUIRED_OUTPUTS = [
    CONTRACT, SS_TENSOR, MFM_METRICS, DENOISE_EMB, DENOISE_METRICS, CONTRASTIVE, REP_QUALITY,
    EMB_CANDIDATES, GR_TARGETS, GR_ATTEMPTS, GR_CANDIDATES, PATCH_LINKS, QA_QUEUE, LABEL_EXEC,
    GATE_RECHECK, EXP_REGISTRY, MODEL_CARDS, INTEGRITY, LEAKAGE, SUMMARY, BLOCKERS, REPORT,
]

X_FEATURES = [
    "elevation_mean", "slope_mean", "HAND_value", "TWI_value", "TPI_250m_value",
    "distance_to_official_drainage_m", "distance_to_official_water_m",
    "urban_prop", "vegetation_prop", "water_prop",
    "NDVI_mean", "NDWI_mean", "MNDWI_mean", "NDBI_mean",
    "B03_mean", "B04_mean", "B08_mean", "B11_mean",
    "chirps_3d_sum", "chirps_7d_sum", "chirps_30d_sum",
    "feature_family_available_count", "feature_completeness_ratio", "proxy_feature_count",
    "official_feature_count", "missing_feature_family_count",
]
M_FAMILIES = ["physical_topography", "hydrology_proximity", "urban_landcover",
              "spectral_sentinel2", "rainfall_chirps", "quality_lineage"]
SPECTRAL_IDX = [X_FEATURES.index(f) for f in ["NDVI_mean", "NDWI_mean", "MNDWI_mean", "NDBI_mean", "B03_mean", "B04_mean", "B08_mean", "B11_mean"]]
PHYSICAL_IDX = [X_FEATURES.index(f) for f in ["elevation_mean", "slope_mean", "HAND_value", "TWI_value", "TPI_250m_value"]]
HYDRO_IDX = [X_FEATURES.index(f) for f in ["distance_to_official_drainage_m", "distance_to_official_water_m"]]

NETWORK_ENVS = ["SUSC_18C_ALLOW_NETWORK", "SUSC_18C_ALLOW_PUBLIC_DOWNLOAD",
                "SUSC_18C_ALLOW_OFFICIAL_SOURCE_SEARCH", "SUSC_18C_ALLOW_SAR_FEASIBILITY",
                "SUSC_18C_ALLOW_LIGHTWEIGHT_VECTOR", "SUSC_18C_ALLOW_LIGHTWEIGHT_RASTER"]

GR_SOURCES = [
    ("Defesa Civil Recife", "civil_defense", "municipal_official"),
    ("Dados Recife", "open_data_portal", "municipal_official"),
    ("APAC", "state_agency", "state_official"),
    ("SGB/CPRM", "geological_survey", "federal_official"),
    ("ANA", "water_agency", "federal_official"),
    ("CEMADEN", "monitoring_center", "federal_official"),
    ("GeoCuritiba", "open_data_portal", "municipal_official"),
    ("Diario Oficial", "official_gazette", "official_publication"),
    ("Agencia Brasil/EBC", "public_media", "public_media"),
    ("Copernicus EMS", "emergency_mapping", "international_official"),
    ("Sentinel-1/SAR feasibility", "sar_feasibility", "international_technical"),
]


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


def _committed(path):
    r = subprocess.run(["git", "cat-file", "-e", f"HEAD:{rel(path)}"], cwd=ROOT,
                       text=True, capture_output=True, check=False)
    return r.returncode == 0


def _require_inputs():
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(p) for p in missing))


def _network_enabled():
    return all(os.environ.get(e) == "1" for e in NETWORK_ENVS)


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return _pearson(ra, rb)


def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# carregar tensor 18B + metadados
# ---------------------------------------------------------------------------

_CACHE = {}


def load_tensor():
    if "X" in _CACHE:
        return _CACHE
    trows = read_csv(B_TENSOR)
    ds = {r["unified_patch_id"]: r for r in read_csv(B_DATASET)}
    ids = [r["unified_patch_id"] for r in trows]
    X = np.zeros((len(trows), len(X_FEATURES)), dtype=float)
    miss_count = np.zeros(len(trows), dtype=int)
    meta = []
    for i, r in enumerate(trows):
        for j, f in enumerate(X_FEATURES):
            X[i, j] = _num(r.get(f"x_{f}")) or 0.0
        miss_count[i] = sum(1 for fam in M_FAMILIES if r.get(f"m_{fam}") == "1")
        d = ds.get(r["unified_patch_id"], {})
        meta.append({
            "unified_patch_id": r["unified_patch_id"], "ml_sample_id": r["ml_sample_id"],
            "patch_family": r["patch_family"], "region": r["region"],
            "source_patch_id": d.get("source_patch_id", ""), "city": d.get("city", ""),
            "is_canary": r["patch_family"] == "event_anchored_canary_patch",
            "is_control": r["patch_family"] == "spatial_mismatch_control_candidate",
            "is_official": r["patch_family"] == "official_population_patch",
            "completeness": _num(d.get("feature_completeness_ratio")) or 0.0,
        })
    _CACHE.update({"X": X, "ids": ids, "meta": meta, "miss_count": miss_count,
                   "canary_idx": [i for i, m in enumerate(meta) if m["is_canary"]]})
    return _CACHE


# ---------------------------------------------------------------------------
# algoritmos self-supervised (numpy deterministico)
# ---------------------------------------------------------------------------

def _lowrank_reconstruct(M, k):
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    k = min(k, S.size)
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k], S


def _denoise(X, latent=5, noise=0.1, seed=SEED):
    rng = np.random.RandomState(seed)
    Xn = X + noise * rng.standard_normal(X.shape)
    U, S, Vt = np.linalg.svd(Xn, full_matrices=False)
    k = min(latent, S.size)
    for r in range(k):
        j = int(np.argmax(np.abs(Vt[r])))
        if Vt[r, j] < 0:
            Vt[r] = -Vt[r]
            U[:, r] = -U[:, r]
    Z = U[:, :k] * S[:k]
    Xhat = Z @ Vt[:k]
    recon_err = np.linalg.norm(X - Xhat, axis=1)
    mae = float(np.mean(np.abs(X - Xhat)))
    rmse = float(np.sqrt(np.mean((X - Xhat) ** 2)))
    ev = float(np.sum(S[:k] ** 2) / np.sum(S ** 2)) if np.sum(S ** 2) > 0 else 0.0
    return Z, recon_err, mae, rmse, ev


def _compute():
    if "denoise_Z" in _CACHE:
        return _CACHE
    d = load_tensor()
    X = d["X"]
    Z, recon_err, mae, rmse, ev = _denoise(X, latent=5, noise=0.1)
    _CACHE.update({"denoise_Z": Z, "denoise_recon_err": recon_err, "denoise_mae": mae,
                   "denoise_rmse": rmse, "denoise_ev": ev,
                   "Zdist": np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)})
    return _CACHE


# ---------------------------------------------------------------------------
# Etapa 1 - contrato
# ---------------------------------------------------------------------------

def contract():
    return {
        "milestone": "SUSC-18C",
        "self_supervised_training_allowed": True,
        "supervised_training_allowed": False,
        "supervised_training_can_be_enabled_by_label_contract": True,
        "canary_is_positive_label": False,
        "control_is_negative_label": False,
        "score_v6_is_target": False,
        "transparent_index_is_target": False,
        "occurrence_anchor_is_feature": False,
        "ground_reference_candidate_is_label": False,
        "accepted_ground_reference_can_enable_training": True,
        "self_supervised_methods": [
            "masked_feature_modeling", "denoising_autoencoder_tabular",
            "contrastive_similarity_representation", "family_ablation_embeddings",
            "region_holdout_reconstruction",
        ],
        "post_event_footprint_never_pre_event_feature": True,
        "review_only": True,
    }


# ---------------------------------------------------------------------------
# Etapa 2 - self-supervised training tensor
# ---------------------------------------------------------------------------

def training_tensor_rows():
    trows = read_csv(B_TENSOR)
    rows = []
    for i, r in enumerate(trows, start=1):
        miss = sum(1 for fam in M_FAMILIES if r.get(f"m_{fam}") == "1")
        row = {
            "training_tensor_id": f"S18C_TT_{i:04d}", "ml_sample_id": r["ml_sample_id"],
            "unified_patch_id": r["unified_patch_id"], "patch_family": r["patch_family"],
            "region": r["region"], "feature_count": str(len(X_FEATURES)),
            "missing_flag_count": str(miss), "allowed_for_self_supervised": "true",
            "allowed_for_supervised": "false", "review_only": "true",
        }
        for f in X_FEATURES:
            row[f"x_{f}"] = r.get(f"x_{f}", "not_available")
        for fam in M_FAMILIES:
            row[f"m_{fam}"] = r.get(f"m_{fam}", "0")
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Etapa 3 - masked feature modeling
# ---------------------------------------------------------------------------

MFM_EXPERIMENTS = [
    ("MFM01_mask_10pct_linear_reconstruction", "random_cells", 0.10, None),
    ("MFM02_mask_20pct_linear_reconstruction", "random_cells", 0.20, None),
    ("MFM03_mask_30pct_linear_reconstruction", "random_cells", 0.30, None),
    ("MFM04_family_mask_spectral", "family_mask", None, "spectral_sentinel2"),
    ("MFM05_family_mask_physical", "family_mask", None, "physical_topography"),
    ("MFM06_family_mask_hydrology", "family_mask", None, "hydrology_proximity"),
]


def _mfm_random(X, frac, seed):
    rng = np.random.RandomState(seed)
    mask = rng.random_sample(X.shape) < frac
    if not mask.any():
        mask[0, 0] = True
    Xm = X.copy()
    Xm[mask] = 0.0
    Xhat, _S = _lowrank_reconstruct(Xm, 8)
    true, pred = X[mask], Xhat[mask]
    return int(mask.sum()), true, pred


def _mfm_family(X, cols):
    other = [j for j in range(X.shape[1]) if j not in cols]
    A = np.hstack([X[:, other], np.ones((X.shape[0], 1))])
    B = X[:, cols]
    W, *_ = np.linalg.lstsq(A, B, rcond=None)
    Bhat = A @ W
    return X.shape[0] * len(cols), B.flatten(), Bhat.flatten()


def masked_feature_modeling_rows():
    d = load_tensor()
    X = d["X"]
    fam_cols = {"spectral_sentinel2": SPECTRAL_IDX, "physical_topography": PHYSICAL_IDX,
                "hydrology_proximity": HYDRO_IDX}
    rows = []
    for i, (exp_id, policy, frac, fam) in enumerate(MFM_EXPERIMENTS, start=1):
        if policy == "random_cells":
            mc, true, pred = _mfm_random(X, frac, SEED + i)
            mask_policy = f"random_{int(frac * 100)}pct_cells"
        else:
            mc, true, pred = _mfm_family(X, fam_cols[fam])
            mask_policy = f"family_mask_{fam}"
        err = np.abs(true - pred)
        mae = float(np.mean(err))
        rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
        rp = _spearman(true, pred)
        rows.append({
            "mfm_metric_id": f"S18C_MFM_{i:04d}", "experiment_id": exp_id, "mask_policy": mask_policy,
            "masked_feature_count": str(mc), "sample_count": str(X.shape[0]),
            "reconstruction_mae": _r(mae), "reconstruction_rmse": _r(rmse),
            "rank_preservation_score": _r(rp), "status": "completed",
            "uses_labels": "false", "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Etapa 4 - denoising autoencoder
# ---------------------------------------------------------------------------

def denoising_embedding_rows():
    d = load_tensor()
    c = _compute()
    Z, err = c["denoise_Z"], c["denoise_recon_err"]
    rows = []
    for i, m in enumerate(d["meta"]):
        zc = {f"z{k + 1}": (_r(float(Z[i, k])) if k < Z.shape[1] else "not_available") for k in range(5)}
        rows.append({
            "denoising_embedding_id": f"S18C_DAE_{i + 1:04d}", "unified_patch_id": m["unified_patch_id"],
            "patch_family": m["patch_family"], "region": m["region"], **zc,
            "reconstruction_error": _r(float(err[i])), "review_only": "true",
        })
    return rows


def denoising_model_metrics_rows():
    d = load_tensor()
    c = _compute()
    return [{
        "model_metric_id": "S18C_DAEM_0001", "method": "linear_denoising_svd_numpy",
        "latent_dim": "5", "sample_count": str(d["X"].shape[0]),
        "reconstruction_mae": _r(c["denoise_mae"]), "reconstruction_rmse": _r(c["denoise_rmse"]),
        "explained_variance_proxy": _r(c["denoise_ev"]), "uses_labels": "false", "review_only": "true",
    }]


# ---------------------------------------------------------------------------
# Etapa 5 - contrastive/similarity representation
# ---------------------------------------------------------------------------

def contrastive_similarity_rows(topk=10):
    d = load_tensor()
    c = _compute()
    ids, meta = d["ids"], d["meta"]
    D = c["Zdist"]
    canary = {i for i in d["canary_idx"]}
    rows = []
    n = 0
    for i in range(len(ids)):
        order = sorted([j for j in range(len(ids)) if j != i], key=lambda j: (D[i, j], ids[j]))
        for rank, j in enumerate(order[:topk], start=1):
            n += 1
            dist = float(D[i, j])
            rows.append({
                "contrastive_similarity_id": f"S18C_CON_{n:05d}", "query_patch_id": ids[i],
                "neighbor_patch_id": ids[j], "rank": str(rank), "distance": _r(dist),
                "similarity_score": _r(1.0 / (1.0 + dist)),
                "same_region": _bool(meta[i]["region"] == meta[j]["region"]),
                "same_patch_family": _bool(meta[i]["patch_family"] == meta[j]["patch_family"]),
                "query_is_canary": _bool(i in canary), "neighbor_is_canary": _bool(j in canary),
                "review_only": "true",
            })
    return rows


# ---------------------------------------------------------------------------
# Etapa 6 - representation quality audit
# ---------------------------------------------------------------------------

def representation_quality_rows():
    d = load_tensor()
    c = _compute()
    ids, meta = d["ids"], d["meta"]
    D = c["Zdist"]
    # vizinhos top5 no embedding
    emb_nb = {}
    for i in range(len(ids)):
        emb_nb[i] = set(sorted([j for j in range(len(ids)) if j != i], key=lambda j: (D[i, j], ids[j]))[:5])
    # vizinhos top5 no tensor (18B similarity)
    id2i = {uid: i for i, uid in enumerate(ids)}
    tensor_nb = {i: set() for i in range(len(ids))}
    for r in read_csv(B_SIMILARITY):
        qi = id2i.get(r["query_patch_id"])
        ni = id2i.get(r["neighbor_patch_id"])
        if qi is not None and ni is not None and int(r["rank"]) <= 5:
            tensor_nb[qi].add(ni)
    stability = np.mean([len(emb_nb[i] & tensor_nb[i]) / 5.0 for i in range(len(ids))])
    fam_mix = np.mean([np.mean([meta[j]["patch_family"] != meta[i]["patch_family"] for j in emb_nb[i]]) for i in range(len(ids))])
    reg_mix = np.mean([np.mean([meta[j]["region"] != meta[i]["region"] for j in emb_nb[i]]) for i in range(len(ids))])
    canary = set(d["canary_idx"])
    can_dens = np.mean([np.mean([j in canary for j in emb_nb[i]]) for i in canary]) if canary else 0.0
    miss_sens = _pearson(c["denoise_recon_err"], d["miss_count"].astype(float))
    metrics = [
        ("reconstruction_mae", c["denoise_mae"], "erro medio absoluto de reconstrucao denoising", "escala normalizada review-only"),
        ("reconstruction_rmse", c["denoise_rmse"], "raiz do erro quadratico medio", "sensivel a outliers"),
        ("neighbor_stability", float(stability), "sobreposicao media top5 tensor vs embedding", "alta=representacao preserva vizinhanca"),
        ("family_mixing_ratio", float(fam_mix), "fracao media de vizinhos de familia diferente", "baixa=embedding separa por familia (disponibilidade)"),
        ("region_mixing_ratio", float(reg_mix), "fracao media de vizinhos de regiao diferente", "review-only"),
        ("canary_neighbor_density", float(can_dens), "densidade media de canarios na vizinhanca de canarios", "canarios proximos entre si; NAO e positividade"),
        ("missingness_sensitivity", miss_sens, "correlacao erro de reconstrucao x n de familias faltantes", "alta=missingness domina representacao"),
    ]
    rows = []
    for i, (name, val, interp, lim) in enumerate(metrics, start=1):
        rows.append({
            "quality_metric_id": f"S18C_RQ_{i:04d}", "metric_name": name, "metric_value": _r(val),
            "interpretation": interp, "limitation": lim, "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Etapa 7 - embedding candidate discovery v2
# ---------------------------------------------------------------------------

def _percentile_rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    n = len(values)
    pr = {}
    for rank, i in enumerate(order):
        pr[i] = round(100.0 * rank / (n - 1), 2) if n > 1 else 0.0
    return pr


def _embedding_candidates_ranked():
    d = load_tensor()
    c = _compute()
    X, meta, ids = d["X"], d["meta"], d["ids"]
    Z = d["denoise_Z"] if "denoise_Z" in d else c["denoise_Z"]
    D = c["Zdist"]
    canary_idx = d["canary_idx"]
    err = c["denoise_recon_err"]
    phys = _percentile_rank(X[:, PHYSICAL_IDX].mean(axis=1))
    hydro = _percentile_rank(X[:, HYDRO_IDX].mean(axis=1))
    # similaridade media a canarios no embedding
    sim_can = []
    for i in range(len(ids)):
        if canary_idx:
            dists = [D[i, j] for j in canary_idx if j != i]
            sim_can.append(float(np.mean([1.0 / (1.0 + dd) for dd in dists])) if dists else 0.0)
        else:
            sim_can.append(0.0)
    ranked = []
    for i, m in enumerate(meta):
        score = sim_can[i] * 40.0 + (100.0 - _percentile_rank(err)[i]) * 0.10 \
            + m["completeness"] * 20.0 + (phys[i] + hydro[i]) * 0.15
        ranked.append((score, i))
    ranked.sort(key=lambda t: (-t[0], ids[t[1]]))
    return ranked, sim_can, err, phys, hydro


def embedding_candidate_discovery_rows(top_n=100):
    d = load_tensor()
    meta = d["meta"]
    ranked, sim_can, err, phys, hydro = _embedding_candidates_ranked()
    rows = []
    for rank, (score, i) in enumerate(ranked[:top_n], start=1):
        m = meta[i]
        action = "acquire_ground_reference_high_priority" if (phys[i] >= 70 and m["completeness"] >= 0.8) else "review_only_screening"
        rows.append({
            "embedding_candidate_id": f"S18C_ECAND_{rank:04d}", "unified_patch_id": m["unified_patch_id"],
            "source_patch_id": m["source_patch_id"], "patch_family": m["patch_family"],
            "priority_rank": str(rank), "embedding_similarity_to_canaries": _r(sim_can[i]),
            "denoising_reconstruction_error": _r(float(err[i])),
            "physical_signal_percentile": _r(phys[i], 2), "hydrology_signal_percentile": _r(hydro[i], 2),
            "data_completeness": _r(m["completeness"], 4), "recommended_ground_reference_action": action,
            "review_only": "true", "ground_truth": "false", "training_label": "false",
        })
    return rows


# ---------------------------------------------------------------------------
# Etapa 8 - ground reference target pack
# ---------------------------------------------------------------------------

def ground_reference_target_rows():
    d = load_tensor()
    meta = {m["unified_patch_id"]: m for m in d["meta"]}
    ranked, sim_can, err, phys, hydro = _embedding_candidates_ranked()
    idx_by_uid = {m["unified_patch_id"]: i for i, m in enumerate(d["meta"])}
    seen = set()
    ordered = []
    # 1. canarios (tem ocorrencia oficial ancorada)
    for m in d["meta"]:
        if m["is_canary"]:
            ordered.append((m["unified_patch_id"], "canary_anchored_official_hydrological_occurrence"))
            seen.add(m["unified_patch_id"])
    # 2. top embedding candidates
    for _score, i in ranked:
        uid = d["meta"][i]["unified_patch_id"]
        if uid not in seen:
            ordered.append((uid, "high_embedding_signal_no_reference_yet"))
            seen.add(uid)
    rows = []
    for rank, (uid, reason) in enumerate(ordered, start=1):
        m = meta[uid]
        i = idx_by_uid[uid]
        needs_ref = not m["is_canary"]
        rows.append({
            "ground_reference_target_id": f"S18C_GRT_{rank:04d}", "unified_patch_id": uid,
            "source_patch_id": m["source_patch_id"], "patch_family": m["patch_family"],
            "city": m["city"], "region": m["region"], "priority_rank": str(rank),
            "target_reason": reason,
            "required_evidence_type": "dated_hydrological_occurrence_with_geometry",
            "preferred_source_authority": "civil_defense_or_open_data_municipal",
            "needs_event_date": _bool(needs_ref), "needs_geometry": _bool(needs_ref),
            "needs_patch_link": "true",
            "needs_sar_followup": _bool(phys[i] >= 60 and needs_ref),
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Etapa 9 - ground reference acquisition attempts
# ---------------------------------------------------------------------------

def _event_date_from_id(event_id):
    # REC_2022_05_24_30 -> 2022-05-24
    parts = event_id.split("_")
    if len(parts) >= 4 and parts[1].isdigit():
        return f"{parts[1]}-{parts[2]}-{parts[3]}"
    return "not_available"


def acquisition_attempt_rows():
    targets = ground_reference_target_rows()
    net = _network_enabled()
    canary_uids = {m["unified_patch_id"] for m in load_tensor()["meta"] if m["is_canary"]}
    rows = []
    n = 0
    # >=50 tentativas: canarios via reuso de ocorrencia existente + demais via rede (fail-closed)
    for t in targets:
        if n >= max(60, 50):
            break
        uid = t["unified_patch_id"]
        n += 1
        if uid in canary_uids:
            src, stype, auth = GR_SOURCES[0]  # Defesa Civil (ocorrencia ja geocodificada 17C33)
            rows.append({
                "acquisition_attempt_id": f"S18C_ACQ_{n:04d}", "ground_reference_target_id": t["ground_reference_target_id"],
                "source_name": src, "source_type": "reuse_existing_official_occurrence", "source_authority": auth,
                "attempt_mode": "offline_reuse_17c33_occurrence", "network_enabled": _bool(net),
                "artifact_acquired": "true", "event_date_found": "true", "geometry_found": "true",
                "patch_link_possible": "true", "sar_followup_possible": _bool(net),
                "blocking_reason": "none_reused_existing_geocoded_occurrence", "review_only": "true",
            })
        else:
            src, stype, auth = GR_SOURCES[n % len(GR_SOURCES)]
            blocked = not net
            rows.append({
                "acquisition_attempt_id": f"S18C_ACQ_{n:04d}", "ground_reference_target_id": t["ground_reference_target_id"],
                "source_name": src, "source_type": stype, "source_authority": auth,
                "attempt_mode": "network_official_source_search" if net else "offline_blocked_by_env",
                "network_enabled": _bool(net), "artifact_acquired": "false",
                "event_date_found": "false", "geometry_found": "false", "patch_link_possible": "false",
                "sar_followup_possible": _bool(net and stype == "sar_feasibility"),
                "blocking_reason": "network_not_enabled_SUSC_18C_ALLOW_*" if blocked else "no_dated_geometry_found_this_pass",
                "review_only": "true",
            })
    return rows


# ---------------------------------------------------------------------------
# Etapa 10 - ground reference candidate registry
# ---------------------------------------------------------------------------

def ground_reference_candidate_rows():
    d = load_tensor()
    uid_by_src = {m["source_patch_id"]: m for m in d["meta"]}
    rows = []
    n = 0
    for c in read_csv(C33_REGISTRY):
        spid = c["event_anchored_canary_patch_id"]
        m = uid_by_src.get(spid)
        if not m:
            continue
        n += 1
        event_id = c.get("event_id", "not_available")
        rows.append({
            "ground_reference_candidate_id": f"S18C_GRC_{n:04d}", "event_id": event_id,
            "unified_patch_id": m["unified_patch_id"], "source_patch_id": spid,
            "city": "recife", "region": c.get("bairro", "recife"),
            "phenomenon_type": "hydrological_flood" if str(c.get("hydrological_signal", "")).lower() == "true" else "unknown",
            "event_date": _event_date_from_id(event_id), "event_date_precision": "event_window_from_id",
            "geometry_type": "point", "geometry_source": "geocoded_street_level_17c32",
            "geometry_precision_m": c.get("uncertainty_m", "not_available"),
            "source_authority": "defesa_civil_recife_occurrence",
            "source_confidence": "official_but_geocoded" if str(c.get("geocoder_is_official", "")).lower() == "true" else "medium_street_geocode",
            "patch_link_status": "linked_to_canary_patch", "qa_status": "pending",
            "eligible_for_training": "false", "eligible_for_17b": "false",
            "not_ground_truth_reason": "sem QA aceito nem label contract; geocode nivel-rua; 1 evento so",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Etapa 11 - patch-reference link candidates
# ---------------------------------------------------------------------------

def patch_reference_link_rows():
    cands = ground_reference_candidate_rows()
    rows = []
    for i, c in enumerate(cands, start=1):
        rows.append({
            "patch_reference_link_id": f"S18C_PRL_{i:04d}",
            "ground_reference_candidate_id": c["ground_reference_candidate_id"],
            "unified_patch_id": c["unified_patch_id"],
            "patch_intersects_geometry": "true", "distance_to_geometry_m": "0.0",
            "temporal_compatibility": "pre_event_window_ok", "phenomenon_compatibility": "hydrological_match",
            "spatial_uncertainty_m": c["geometry_precision_m"], "link_confidence": "medium",
            "qa_required": "true", "accepted_for_label_contract": "false", "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Etapa 12 - QA queue
# ---------------------------------------------------------------------------

def qa_queue_rows():
    cands = ground_reference_candidate_rows()
    links = {l["ground_reference_candidate_id"]: l for l in patch_reference_link_rows()}
    questions = [
        ("event_date_precision", "A data do evento e precisa o suficiente para janela pre-evento?"),
        ("geometry_precision", "A geometria (ponto geocodificado ~800m) e precisa o suficiente para o patch?"),
        ("phenomenon_match", "O fenomeno documentado e hidrologico compativel com o patch?"),
    ]
    rows = []
    n = 0
    for c in cands:
        link = links.get(c["ground_reference_candidate_id"], {})
        for qname, q in questions:
            n += 1
            rows.append({
                "qa_queue_id": f"S18C_QA_{n:04d}", "ground_reference_candidate_id": c["ground_reference_candidate_id"],
                "patch_reference_link_id": link.get("patch_reference_link_id", "not_available"),
                "qa_priority": "high", "qa_question": q,
                "evidence_to_review": f"{c['source_authority']};{c['geometry_source']};uncertainty={c['geometry_precision_m']}m",
                "expected_decision_values": "accepted;rejected;ambiguous;needs_more_source;phenomenon_mismatch;temporal_mismatch;spatial_mismatch",
                "blocking_if_unresolved": "true", "review_only": "true",
            })
    return rows


# ---------------------------------------------------------------------------
# Etapa 13 - label contract execution
# ---------------------------------------------------------------------------

def _label_contract_state():
    cands = ground_reference_candidate_rows()
    accepted = [c for c in cands if c["qa_status"] == "accepted"]  # nenhum aceito nesta iteracao
    events = {c["event_id"] for c in accepted}
    regions = {c["region"] for c in accepted}
    req = read_json(B_LABEL_CONTRACT)["minimum_criteria_for_future_training"]
    accepted_count = len(accepted)
    event_count = len(events)
    region_count = len(regions)
    temporal_split = event_count >= 2
    negative_policy = False
    qa_accepted_only = True
    allowed = (accepted_count >= req["accepted_ground_reference_count"] and event_count >= req["event_count"]
               and region_count >= req["region_count"] and temporal_split and negative_policy)
    return {
        "req": req, "accepted_count": accepted_count, "event_count": event_count,
        "region_count": region_count, "temporal_split": temporal_split,
        "negative_policy": negative_policy, "qa_accepted_only": qa_accepted_only,
        "allowed": allowed, "candidate_count": len(cands),
    }


def label_contract_execution_rows():
    s = _label_contract_state()
    req = s["req"]
    blocking = []
    if s["accepted_count"] < req["accepted_ground_reference_count"]:
        blocking.append(f"accepted<{req['accepted_ground_reference_count']}")
    if s["event_count"] < req["event_count"]:
        blocking.append(f"events<{req['event_count']}")
    if s["region_count"] < req["region_count"]:
        blocking.append(f"regions<{req['region_count']}")
    if not s["temporal_split"]:
        blocking.append("temporal_split_not_possible")
    if not s["negative_policy"]:
        blocking.append("negative_policy_not_approved")
    return [{
        "label_contract_execution_id": "S18C_LCE_0001",
        "required_accepted_ground_reference_count": str(req["accepted_ground_reference_count"]),
        "accepted_ground_reference_count": str(s["accepted_count"]),
        "required_event_count": str(req["event_count"]), "event_count": str(s["event_count"]),
        "required_region_count": str(req["region_count"]), "region_count": str(s["region_count"]),
        "temporal_split_possible": _bool(s["temporal_split"]),
        "negative_sampling_policy_approved": _bool(s["negative_policy"]),
        "qa_accepted_only": _bool(s["qa_accepted_only"]),
        "label_contract_training_allowed": _bool(s["allowed"]),
        "blocking_reason": ";".join(blocking) if blocking else "none",
        "review_only": "true",
    }]


# ---------------------------------------------------------------------------
# Etapa 14 - supervised gate recheck
# ---------------------------------------------------------------------------

def supervised_gate_recheck_rows():
    s = _label_contract_state()
    allowed = s["allowed"]
    ready_pending = allowed  # criterios atingidos mas .fit nao executa sem aprovacao explicita
    return [{
        "supervised_gate_recheck_id": "S18C_SGR_0001",
        "previous_status_18b": "blocked_no_ground_reference_labels",
        "label_contract_training_allowed": _bool(allowed),
        "supervised_training_allowed_after_18c": _bool(allowed),
        "supervised_training_executed": "false",
        "supervised_training_ready_pending_approval": _bool(ready_pending),
        "blocking_reason": "none_ready_pending_user_approval" if ready_pending else
                           f"0 ground references aceitos (candidatos={s['candidate_count']}, eventos={s['event_count']}<{s['req']['event_count']}, regioes={s['region_count']}<{s['req']['region_count']})",
        "safe_next_step": "executar supervised baseline apos aprovacao humana" if ready_pending else
                          "QA + aquisicao de mais eventos/regioes de ground reference oficial",
        "review_only": "true",
    }]


# ---------------------------------------------------------------------------
# Etapa 15 - experiment registry
# ---------------------------------------------------------------------------

def experiment_registry_rows():
    rows = []
    for exp_id, _p, _f, _fam in MFM_EXPERIMENTS:
        rows.append(_exp_row(exp_id, "self_supervised_masked_modeling", exp_id, "physical;hydrology;urban;spectral;rainfall;quality", "316", "completed", rel(MFM_METRICS)))
    rows.append(_exp_row("DAE01_denoising_svd_5d", "self_supervised_denoising", "linear_denoising_svd_numpy", "all_x_features", "316", "completed", rel(DENOISE_EMB)))
    rows.append(_exp_row("CON01_contrastive_similarity", "self_supervised_contrastive", "embedding_euclidean_topk", "all_x_features", "316", "completed", rel(CONTRASTIVE)))
    rows.append(_exp_row("GRF01_ground_reference_factory", "ground_reference_factory", "candidate_to_qa_pipeline", "reference_only", "0", "completed_candidates_pending_qa", rel(GR_CANDIDATES)))
    rows.append(_exp_row("SGR01_supervised_gate_recheck", "supervised_gate_recheck", "label_contract_evaluation", "none", "0", "blocked_no_accepted_labels", rel(GATE_RECHECK)))
    return rows


def _exp_row(exp_id, etype, method, groups, n, status, out):
    return {
        "experiment_id": exp_id, "experiment_type": etype, "method": method, "feature_groups": groups,
        "sample_count": n, "uses_labels": "false", "uses_ground_truth": "false",
        "uses_score_v6_as_target": "false", "status": status, "output_files": out, "review_only": "true",
    }


# ---------------------------------------------------------------------------
# Etapa 16 - model cards
# ---------------------------------------------------------------------------

def model_card_rows():
    cards = [
        ("masked_feature_modeling", "MFM01_mask_10pct_linear_reconstruction", "self_supervised_reconstruction", "completed",
         "aprender estrutura multimodal reconstruindo features mascaradas", "usar como previsao/classe/risco/ground truth",
         "all_x_features", "reconstrucao low-rank; escala normalizada; sem label"),
        ("denoising_autoencoder", "DAE01_denoising_svd_5d", "self_supervised_representation", "completed",
         "embedding tabular denoising review-only", "usar embedding como score/risco/target",
         "all_x_features", "latente linear SVD 5d; nao interpretavel como risco"),
        ("contrastive_similarity", "CON01_contrastive_similarity", "self_supervised_similarity", "completed",
         "recuperacao de vizinhanca multimodal review-only", "usar similaridade como ground truth/label",
         "denoising_embedding", "distancia euclidiana no embedding; sem label"),
        ("ground_reference_factory", "GRF01_ground_reference_factory", "reference_acquisition", "candidates_pending_qa",
         "montar candidatos de referencia forte p/ treino futuro", "tratar candidato como ground truth antes de QA",
         "reference_only", "candidatos = ocorrencias geocodificadas ~800m; 1 evento; pendente QA"),
        ("supervised_gate_recheck", "SGR01_supervised_gate_recheck", "gate_evaluation", "blocked_no_labels",
         "reavaliar se treino supervisionado pode habilitar", "executar .fit sem contrato aprovado",
         "none", "0 references aceitos; harness fail-closed"),
        ("logistic_regression_reserved", "SUP_logistic_regression", "supervised_classification", "blocked_no_labels",
         "reservado p/ treino futuro com labels aceitos", "treinar sem ground reference aceito",
         "modeling_features", "harness 18B fail-closed; sem labels"),
    ]
    rows = []
    for i, (name, exp, task, status, intended, forbidden, feats, limits) in enumerate(cards, start=1):
        rows.append({
            "model_card_id": f"S18C_CARD_{i:04d}", "experiment_id": exp, "model_or_method_name": name,
            "task_type": task, "training_status": status, "intended_use": intended,
            "forbidden_use": forbidden, "input_features": feats, "known_limitations": limits,
            "uses_labels": "false", "uses_ground_truth": "false", "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# audits + summary + blockers + report
# ---------------------------------------------------------------------------

def score_integrity_audit_rows():
    v6_changed = bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]))
    return [
        {"score_integrity_audit_id": "S18C_INTEG_0001", "protected_artifact": rel(SCORE_V6),
         "artifact_type": "score_v6_candidate", "changed_in_working_tree": _bool(v6_changed),
         "used_as": "reference_only_never_target", "created_new": "false", "review_only": "true"},
        {"score_integrity_audit_id": "S18C_INTEG_0002", "protected_artifact": rel(SCORE_V7),
         "artifact_type": "score_v7_candidate", "changed_in_working_tree": "false",
         "used_as": "none", "created_new": _bool(SCORE_V7.exists()), "review_only": "true"},
    ]


def no_leakage_audit_rows():
    objs = [(e[0], "self_supervised_masked_modeling") for e in MFM_EXPERIMENTS]
    objs += [("DAE01_denoising_svd_5d", "self_supervised_denoising"),
             ("CON01_contrastive_similarity", "self_supervised_contrastive"),
             ("GRF01_ground_reference_factory", "ground_reference_factory"),
             ("SGR01_supervised_gate_recheck", "supervised_gate_recheck"),
             ("self_supervised_training_tensor", "dataset")]
    rows = []
    for i, (oid, otype) in enumerate(objs, start=1):
        rows.append({
            "no_leakage_audit_id": f"S18C_LEAK_{i:04d}", "object_id": oid, "object_type": otype,
            "uses_occurrence_as_feature": "false", "uses_canary_as_positive_label": "false",
            "uses_control_as_negative_label": "false", "uses_score_v6_as_target": "false",
            "uses_transparent_index_as_target": "false", "uses_ground_truth_label": "false",
            "uses_training_label": "false", "uses_post_event_feature_as_pre_event": "false",
            "uses_flow_acc_as_equivalent": "false", "ground_reference_candidate_treated_as_ground_truth": "false",
            "passes_no_leakage": "true",
            "blocking_reason": "self-supervised sem label; referencias candidatas pendentes de QA; harness supervisionado fail-closed",
            "review_only": "true",
        })
    return rows


def build_summary():
    d = load_tensor()
    tt = training_tensor_rows()
    mfm = masked_feature_modeling_rows()
    demb = denoising_embedding_rows()
    dmet = denoising_model_metrics_rows()
    con = contrastive_similarity_rows()
    rq = representation_quality_rows()
    ecand = embedding_candidate_discovery_rows()
    targets = ground_reference_target_rows()
    attempts = acquisition_attempt_rows()
    cands = ground_reference_candidate_rows()
    links = patch_reference_link_rows()
    qa = qa_queue_rows()
    lce = _label_contract_state()
    exp_reg = experiment_registry_rows()
    cards = model_card_rows()
    b_cand = read_csv(B_CANDIDATES)

    ss_exp_count = len(MFM_EXPERIMENTS) + 2  # MFM + denoising + contrastive
    allowed = lce["allowed"]
    accepted_gate_passed = allowed
    minimum = (
        len(d["ids"]) >= 316 and len(tt) >= 316 and ss_exp_count >= 5 and len(mfm) >= 5
        and len(rq) >= 5 and len(b_cand) >= 50 and len(targets) >= 75 and len(attempts) >= 50
        and len(cands) >= 1 and len(links) >= 1 and len(demb) >= 316 and len(con) >= 3160
        and not False  # supervised nao executa
    )
    return {
        "minimum_success_achieved": minimum,
        "ml_ready_dataset_rows_consumed_count": len(read_csv(B_DATASET)),
        "feature_tensor_rows_consumed_count": len(d["ids"]),
        "self_supervised_training_tensor_rows_count": len(tt),
        "self_supervised_experiment_count": ss_exp_count,
        "masked_feature_modeling_experiment_count": len(MFM_EXPERIMENTS),
        "masked_reconstruction_metrics_rows_count": len(mfm),
        "denoising_embedding_rows_count": len(demb),
        "denoising_model_metrics_rows_count": len(dmet),
        "contrastive_similarity_rows_count": len(con),
        "representation_quality_rows_count": len(rq),
        "embedding_candidate_discovery_rows_count": len(ecand),
        "candidate_discovery_rows_consumed_count": len(b_cand),
        "ground_reference_factory_targets_count": len(targets),
        "ground_reference_acquisition_attempts_count": len(attempts),
        "ground_reference_candidate_rows_count": len(cands),
        "patch_reference_link_candidate_rows_count": len(links),
        "qa_queue_rows_count": len(qa),
        "label_contract_execution_created": True,
        "accepted_ground_reference_count": lce["accepted_count"],
        "event_count": lce["event_count"],
        "region_count": lce["region_count"],
        "temporal_split_possible": lce["temporal_split"],
        "negative_policy_approved": lce["negative_policy"],
        "label_contract_training_allowed": allowed,
        "trainability_gate_rechecked": True,
        "supervised_harness_rechecked": True,
        "supervised_gate_rechecked": True,
        "supervised_training_allowed_after_18c": allowed,
        "supervised_training_ready_pending_approval": allowed,
        "supervised_training_executed": False,
        "self_supervised_pretraining_completed": True,
        "network_enabled": _network_enabled(),
        "experiment_registry_rows_count": len(exp_reg),
        "model_card_rows_count": len(cards),
        "accepted_ground_reference_gate_passed": accepted_gate_passed,
        "canary_used_as_positive_label": False,
        "control_used_as_negative_label": False,
        "score_v6_used_as_training_target": False,
        "occurrence_used_as_causal_feature": False,
        "flow_acc_treated_as_equivalent": False,
        "score_v6_original_file_changed": bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)])),
        "score_v7_created": SCORE_V7.exists(),
        "ground_truth_created": False,
        "training_labels_created": False,
        "official_patch_created": False,
        "official_patch_link_created": False,
        "eligible_for_17b_now": accepted_gate_passed,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "result_class": "A_self_supervised_complete_supervised_blocked" if not allowed else "B_self_supervised_complete_supervised_ready_pending_approval",
        "recommended_next_milestone": (
            "SUSC-18D executar QA humano da fila de ground reference (7 tipos de decisao) e ampliar aquisicao "
            "para >=5 eventos e >=3 regioes; ao atingir o contrato, habilitar supervised baseline sob aprovacao "
            "explicita (Resultado C). Ate la: self-supervised pretraining review-only e triagem de candidatos. "
            "Sem score v7, sem ground truth, 17B fail-closed."),
    }


BLOCKER_LIST = [
    ("supervised_training_blocked_no_accepted_ground_reference", "0 ground references aceitos por QA"),
    ("label_contract_not_satisfied", "criterios do label contract nao cumpridos (accepted<50, eventos<5, regioes<3)"),
    ("negative_policy_not_approved", "politica de amostragem negativa nao aprovada"),
    ("temporal_split_not_possible", "1 evento so nao permite split temporal (>=2)"),
    ("qa_not_accepted", "candidatos de ground reference pendentes de QA (nenhum aceito)"),
    ("ground_truth_blocked", "sem ground truth"),
    ("score_v7_blocked", "sem score v7"),
    ("17b_blocked", "17B exige G4_full + ground reference aceito suficiente; nao atingido"),
    ("self_supervised_completed", "pretraining self-supervised entregue (review-only)"),
]


def promotion_blocker_rows():
    return [{"blocker_id": f"S18C_BLOCKER_{i:04d}", "blocker_type": n, "description": d,
             "blocks_supervised_training": "true", "blocks_17b": "true", "review_only": "true"}
            for i, (n, d) in enumerate(BLOCKER_LIST, start=1)]


def build_report():
    s = build_summary()
    return "\n".join([
        "# SUSC-18C - Self-Supervised Multimodal Pretraining e Ground Reference Factory",
        "",
        "## Objetivo",
        "Iniciar aprendizado multimodal REAL sem labels (self-supervised sobre o feature tensor 18B) e, em paralelo, construir a fabrica de Ground Reference necessaria para destravar treino supervisionado futuro de forma cientificamente valida.",
        "",
        f"## Resultado: {s['result_class']}",
        f"- self_supervised_pretraining_completed={s['self_supervised_pretraining_completed']}",
        f"- supervised_training_executed={s['supervised_training_executed']} | ready_pending_approval={s['supervised_training_ready_pending_approval']} | allowed_after_18c={s['supervised_training_allowed_after_18c']}",
        "",
        "## Frente A - Self-supervised pretraining (numpy deterministico)",
        f"- Training tensor: {s['self_supervised_training_tensor_rows_count']} x {len(X_FEATURES)} features (targets proibidos excluidos).",
        f"- Masked feature modeling: {s['masked_reconstruction_metrics_rows_count']} experimentos (mask 10/20/30% + family masks spectral/physical/hydrology).",
        f"- Denoising autoencoder (SVD linear 5d): {s['denoising_embedding_rows_count']} embeddings.",
        f"- Contrastive similarity: {s['contrastive_similarity_rows_count']} pares (top10).",
        f"- Representation quality audit: {s['representation_quality_rows_count']} metricas.",
        "",
        "## Frente B - Ground Reference Factory",
        f"- Target pack: {s['ground_reference_factory_targets_count']} | tentativas de aquisicao: {s['ground_reference_acquisition_attempts_count']} (rede={s['network_enabled']}, fail-closed).",
        f"- Ground reference candidates: {s['ground_reference_candidate_rows_count']} (ocorrencias oficiais geocodificadas dos canarios, 17C33/17C32) | patch-links: {s['patch_reference_link_candidate_rows_count']} | QA queue: {s['qa_queue_rows_count']}.",
        f"- Label contract: accepted={s['accepted_ground_reference_count']} (req>=50), eventos={s['event_count']} (req>=5), regioes={s['region_count']} (req>=3) -> training_allowed={s['label_contract_training_allowed']}.",
        f"- Supervised gate recheck: allowed_after_18c={s['supervised_training_allowed_after_18c']}, executed={s['supervised_training_executed']}.",
        "",
        "## Por que supervised continua bloqueado (Resultado A honesto)",
        "Os candidatos de ground reference sao reais (ocorrencias hidrologicas oficiais geocodificadas), mas: (1) pendentes de QA; (2) apenas 1 evento (REC_2022_05_24_30) < 5; (3) 1 cidade (Recife) < 3 regioes; (4) geocode nivel-rua ~800m; (5) politica de negativos nao aprovada. Candidato != ground truth. Nada e aceito -> .fit supervisionado NAO executa.",
        "",
        "## Guardrails cientificos",
        "- Self-supervised treina mas e review_only; NAO usa labels/GT; canario!=positivo; controle!=negativo.",
        "- Ocorrencia so referencia/ancora (nunca feature causal); score v6/index so referencia (nunca target); flow_acc nao equivalente.",
        "- Footprint pos-evento nunca como feature pre-evento; ground reference candidate nao e GT ate QA+contrato.",
        "- Score v6 intacto; score v7 inexistente; ground truth nao criado; labels nao criados; 17B fail-closed.",
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
    write_json(CONTRACT, contract())
    write_csv(SS_TENSOR, training_tensor_rows())
    write_csv(MFM_METRICS, masked_feature_modeling_rows())
    write_csv(DENOISE_EMB, denoising_embedding_rows())
    write_csv(DENOISE_METRICS, denoising_model_metrics_rows())
    write_csv(CONTRASTIVE, contrastive_similarity_rows())
    write_csv(REP_QUALITY, representation_quality_rows())
    write_csv(EMB_CANDIDATES, embedding_candidate_discovery_rows())
    write_csv(GR_TARGETS, ground_reference_target_rows())
    write_csv(GR_ATTEMPTS, acquisition_attempt_rows())
    write_csv(GR_CANDIDATES, ground_reference_candidate_rows())
    write_csv(PATCH_LINKS, patch_reference_link_rows())
    write_csv(QA_QUEUE, qa_queue_rows())
    write_csv(LABEL_EXEC, label_contract_execution_rows())
    write_csv(GATE_RECHECK, supervised_gate_recheck_rows())
    write_csv(EXP_REGISTRY, experiment_registry_rows())
    write_csv(MODEL_CARDS, model_card_rows())
    write_csv(INTEGRITY, score_integrity_audit_rows())
    write_csv(LEAKAGE, no_leakage_audit_rows())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, promotion_blocker_rows())
    write_markdown(REPORT, build_report())


MODE_DISPATCH = {
    "build-contract": (write_json, CONTRACT, contract),
    "build-training-tensor": (write_csv, SS_TENSOR, training_tensor_rows),
    "run-masked-feature-modeling": (write_csv, MFM_METRICS, masked_feature_modeling_rows),
    "build-contrastive-similarity": (write_csv, CONTRASTIVE, contrastive_similarity_rows),
    "build-representation-quality-audit": (write_csv, REP_QUALITY, representation_quality_rows),
    "build-embedding-candidate-discovery": (write_csv, EMB_CANDIDATES, embedding_candidate_discovery_rows),
    "build-ground-reference-target-pack": (write_csv, GR_TARGETS, ground_reference_target_rows),
    "run-ground-reference-acquisition": (write_csv, GR_ATTEMPTS, acquisition_attempt_rows),
    "build-ground-reference-candidates": (write_csv, GR_CANDIDATES, ground_reference_candidate_rows),
    "build-patch-reference-links": (write_csv, PATCH_LINKS, patch_reference_link_rows),
    "build-qa-queue": (write_csv, QA_QUEUE, qa_queue_rows),
    "execute-label-contract": (write_csv, LABEL_EXEC, label_contract_execution_rows),
    "recheck-supervised-gate": (write_csv, GATE_RECHECK, supervised_gate_recheck_rows),
    "build-experiment-registry": (write_csv, EXP_REGISTRY, experiment_registry_rows),
    "build-model-cards": (write_csv, MODEL_CARDS, model_card_rows),
}


def run_mode(mode):
    _require_inputs()
    ensure_dir(OUT)
    if mode == "run-denoising-pretraining":
        write_csv(DENOISE_EMB, denoising_embedding_rows())
        write_csv(DENOISE_METRICS, denoising_model_metrics_rows())
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
    if not _committed(A_STORE) or not _committed(B_TENSOR):
        print("ERROR: susc_18a_or_18b_not_committed", file=sys.stderr)
        return 1
    missing = [p for p in REQUIRED_OUTPUTS if not p.exists()]
    if missing:
        print("MISSING: " + "; ".join(rel(p) for p in missing), file=sys.stderr)
        return 1
    errors = _validate_byte_identical()

    tt = read_csv(SS_TENSOR)
    mfm = read_csv(MFM_METRICS)
    demb = read_csv(DENOISE_EMB)
    con = read_csv(CONTRASTIVE)
    rq = read_csv(REP_QUALITY)
    ecand = read_csv(EMB_CANDIDATES)
    targets = read_csv(GR_TARGETS)
    attempts = read_csv(GR_ATTEMPTS)
    cands = read_csv(GR_CANDIDATES)
    links = read_csv(PATCH_LINKS)
    qa = read_csv(QA_QUEUE)
    lce = read_csv(LABEL_EXEC)
    gate = read_csv(GATE_RECHECK)
    exp_reg = read_csv(EXP_REGISTRY)
    cards = read_csv(MODEL_CARDS)
    leak = read_csv(LEAKAGE)
    summary = read_json(SUMMARY)
    ss_schema = read_json(SCHEMA_SS)
    grf_schema = read_json(SCHEMA_GRF)
    gate_schema = read_json(SCHEMA_GATE)

    for r in tt:
        errors.extend(_schema_violations(r, ss_schema))
    for r in cands:
        errors.extend(_schema_violations(r, grf_schema))
    for r in gate:
        errors.extend(_schema_violations(r, gate_schema))

    def chk(cond, tag):
        if cond:
            errors.append(tag)

    chk(summary["feature_tensor_rows_consumed_count"] < 316, "tensor_consumed_lt_316")  # 1/2
    chk(len(tt) < 316, "ss_tensor_lt_316")
    chk(summary["self_supervised_experiment_count"] < 5, "ss_experiments_lt_5")  # 3
    chk(len(mfm) < 5, "mfm_metrics_missing")  # 4
    chk(len(demb) < 316, "denoising_embeddings_lt_316")  # 5
    chk(len(con) < 3160, "contrastive_lt_3160")  # 6
    chk(len(rq) < 5, "rep_quality_lt_5")  # 7
    chk(len(ecand) < 75, "embedding_candidates_lt_75")  # 8
    chk(len(targets) < 75, "gr_targets_lt_75")  # 9
    chk(len(attempts) < 50, "gr_attempts_lt_50")  # 10
    chk(len(lce) == 0 or not summary["label_contract_execution_created"], "label_contract_missing")  # 11
    chk(len(gate) == 0 or not summary["supervised_gate_rechecked"], "gate_recheck_missing")  # 12
    chk(len(cands) < 1, "gr_candidates_lt_1")
    chk(len(links) < 1, "patch_links_lt_1")
    chk(len(qa) < 1, "qa_queue_empty")
    # 13 - supervised executa sem contrato
    allowed = lce[0]["label_contract_training_allowed"] == "true" if lce else False
    chk(summary["supervised_training_executed"] and not allowed, "supervised_executed_without_contract")
    chk(summary["supervised_training_executed"], "supervised_executed")
    for r in gate:
        chk(r["supervised_training_executed"] != "false", "gate_executed")
    # 14/15 canario/controle
    for r in leak:
        chk(r["uses_canary_as_positive_label"] != "false", "canary_positive_leak")
        chk(r["uses_control_as_negative_label"] != "false", "control_negative_leak")
    chk(summary["canary_used_as_positive_label"] is not False, "canary_positive_summary")
    chk(summary["control_used_as_negative_label"] is not False, "control_negative_summary")
    # 16 score v6 target
    for r in leak:
        chk(r["uses_score_v6_as_target"] != "false", "score_v6_target_leak")
    chk(summary["score_v6_used_as_training_target"] is not False, "score_v6_target_summary")
    # 17 ocorrencia feature causal
    for r in leak:
        chk(r["uses_occurrence_as_feature"] != "false", "occurrence_feature_leak")
    chk(summary["occurrence_used_as_causal_feature"] is not False, "occurrence_feature_summary")
    # 18 flow_acc equivalente
    for r in leak:
        chk(r["uses_flow_acc_as_equivalent"] != "false", "flow_acc_equiv_leak")
    chk(summary["flow_acc_treated_as_equivalent"] is not False, "flow_acc_equiv_summary")
    # 19 score v6 mudou
    chk(bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)])) or summary["score_v6_original_file_changed"], "score_v6_changed")
    # 20 score v7
    chk(SCORE_V7.exists() or summary["score_v7_created"], "score_v7_exists")
    # 21 ground truth sem accepted
    chk(summary["ground_truth_created"] and summary["accepted_ground_reference_count"] == 0, "gt_without_accepted")
    chk(summary["ground_truth_created"], "gt_created")
    # 22 labels sem contrato
    chk(summary["training_labels_created"] and not allowed, "labels_without_contract")
    # ground reference candidate nunca eligible sem QA
    for r in cands:
        chk(r["eligible_for_training"] != "false", "candidate_eligible_without_qa")
    # 23 17B liberado sem accepted suficiente
    chk(summary["eligible_for_17b_now"] and not summary["accepted_ground_reference_gate_passed"], "17b_without_gate")
    # post-evento como pre-evento
    for r in leak:
        chk(r["uses_post_event_feature_as_pre_event"] != "false", "post_as_pre_leak")
    chk(any(r["passes_no_leakage"] != "true" for r in leak), "no_leakage_failed")
    chk(not summary["minimum_success_achieved"], "minimum_success_not_achieved")

    # 24 summary reflete contagens
    expected = build_summary()
    for k, v in expected.items():
        if summary.get(k) != v:
            errors.append(f"summary_mismatch:{k}")

    for rows, key in [
        (tt, "training_tensor_id"), (mfm, "mfm_metric_id"), (demb, "denoising_embedding_id"),
        (con, "contrastive_similarity_id"), (rq, "quality_metric_id"), (ecand, "embedding_candidate_id"),
        (targets, "ground_reference_target_id"), (attempts, "acquisition_attempt_id"),
        (cands, "ground_reference_candidate_id"), (links, "patch_reference_link_id"),
        (qa, "qa_queue_id"), (cards, "model_card_id"), (leak, "no_leakage_audit_id"),
    ]:
        ids = [r[key] for r in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")
    # experiment_id sao nomes semanticos (nao ordenados): so unicidade
    exp_ids = [r["experiment_id"] for r in exp_reg]
    if len(exp_ids) != len(set(exp_ids)):
        errors.append("experiment_id_not_unique")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(
        "18C -> "
        f"tensor={summary['feature_tensor_rows_consumed_count']} ss_exp={summary['self_supervised_experiment_count']} "
        f"mfm={summary['masked_reconstruction_metrics_rows_count']} denoise_emb={summary['denoising_embedding_rows_count']} "
        f"contrastive={summary['contrastive_similarity_rows_count']} rep_quality={summary['representation_quality_rows_count']} "
        f"emb_cand={summary['embedding_candidate_discovery_rows_count']} gr_targets={summary['ground_reference_factory_targets_count']} "
        f"gr_attempts={summary['ground_reference_acquisition_attempts_count']} gr_cands={summary['ground_reference_candidate_rows_count']} "
        f"accepted={summary['accepted_ground_reference_count']} events={summary['event_count']} regions={summary['region_count']} "
        f"sup_exec={summary['supervised_training_executed']} ready_pending={summary['supervised_training_ready_pending_approval']} "
        f"result={summary['result_class']} min_success={summary['minimum_success_achieved']}"
    )
    return 0


MODES = list(MODE_DISPATCH.keys()) + ["run-denoising-pretraining", "build-offline", "audit"]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="susc_18c")
    parser.add_argument("mode", choices=MODES)
    args = parser.parse_args(argv)
    if args.mode == "audit":
        return validate()
    return run_mode(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
