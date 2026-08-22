"""REV-P v1r7 (SUSC-21) -- Refinamento de EVIDENCIA nos 23 patches de Recife"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

import numpy as np

from revp_v1qg_v1qm_smoke_embedding_common import (
    DATASETS, DOCS, SCHEMAS,
    _p, parse_embedding_from_row, read_csv, write_csv, write_doc, write_schema,
)

IN_EMB = _p("REVP_V1R7_IN_EMB", DATASETS / "dino_recife_sedec_all_embeddings_v1r3.csv")
OUT_SCORES = _p("REVP_V1R7_OUT_SCORES", DATASETS / "dino_recife_evidence_refinement_scores_v1r7.csv")
OUT_QUEUE = _p("REVP_V1R7_OUT_QUEUE", DATASETS / "dino_recife_evidence_refinement_review_queue_v1r7.csv")
OUT_QA = _p("REVP_V1R7_OUT_QA", DATASETS / "dino_recife_evidence_refinement_qa_v1r7.csv")
OUT_BOUNDARY = _p("REVP_V1R7_OUT_BOUNDARY", DATASETS / "dino_evidence_refinement_boundary_matrix_v1r7.csv")
SCH_SCORES = _p("REVP_V1R7_SCH_SCORES", SCHEMAS / "dino_recife_evidence_refinement_scores_v1r7_schema.csv")
DOC = _p("REVP_V1R7_DOC", DOCS / "revp_v1r7_dino_evidence_refinement_recife.md")

# Private PROJETO inputs: never hardcoded. Fail-closed when unset.
ENV_IN_V12 = "REVP_V1R7_IN_V12"
ENV_SENTINEL_DIR = "REVP_V1R7_SENTINEL_DIR"
# v1r5 used the same two private inputs; accept its env names as fallback.
ENV_IN_V12_FALLBACK = "REVP_V1R5_IN_V12"
ENV_SENTINEL_DIR_FALLBACK = "REVP_V1R5_SENTINEL_DIR"

# Pre-registered defaults. Overridable ONLY to reproduce the declared
# sensitivity run (min support 1); the primary route never varies them.
PHI_H = float(os.environ.get("REVP_V1R7_PHI_H", "0.75"))
PHI_L = float(os.environ.get("REVP_V1R7_PHI_L", "0.25"))
MIN_SUPPORT = int(os.environ.get("REVP_V1R7_MIN_SUPPORT", "2"))
N_PERM = 20000
RANDOM_SEED = 0

# Physical/causal features of v12, used ONLY to test whether the DINO ranking
# is redundant with the causal base. Never combined with the DINO score.
PHYS_COLS = ["elevation_m", "slope_deg", "hand_m_dinf", "twi_dinf",
             "rain_decay_index_api_chirps", "rain_peak_residual_orthogonalized"]

SCORE_FIELDS = [
    "patch_id", "provenance", "n_v12_points", "n_pos", "n_neg", "frac_positive",
    "seed_role", "mean_cos_to_positive_seeds", "mean_cos_to_negative_seeds",
    "refinement_score", "refinement_rank_among_targets", "nearest_positive_seed",
    "nearest_positive_seed_cos", "candidate_class",
    "refinement_can_create_label", "refinement_can_train_classifier",
]
QUEUE_FIELDS = [
    "review_rank", "patch_id", "candidate_class", "refinement_score",
    "n_v12_points", "n_pos", "n_neg", "frac_positive", "nearest_positive_seed",
    "nearest_positive_seed_cos", "review_status", "promotion_allowed",
    "evidence_role", "blocking_note",
]
QA_FIELDS = ["qa_key", "qa_value"]
BOUNDARY_FIELDS = [
    "boundary_id", "refinement_can_create_label", "refinement_can_train_classifier",
    "refinement_can_validate_event", "refinement_can_enter_firth_model",
    "refinement_can_prioritize_review", "refinement_role", "seed_source",
    "similarity_space", "unit_of_analysis", "cluster_n", "pseudoreplication_checked",
    "supersedes", "blocker", "notes",
]


# Inputs

def _patch_geometry_wgs84(sentinel_dir: Path) -> dict[str, tuple[float, float, float, float]]:
    import rasterio
    from rasterio.warp import transform_bounds
    out: dict[str, tuple[float, float, float, float]] = {}
    for p in sorted(glob.glob(str(sentinel_dir / "patch_recife_*.tif"))):
        with rasterio.open(p) as ds:
            b = ds.bounds
            bounds = transform_bounds(ds.crs, "EPSG:4326", b.left, b.bottom, b.right, b.top)
        pid = Path(p).stem.split("patch_recife_")[-1].upper()
        out[f"REC_{pid}"] = bounds
    return out


def _join_points_to_patches(in_v12: Path, bboxes: dict[str, tuple[float, float, float, float]],
                            patch_ids: set[str]) -> list[dict[str, Any]]:
    joined: list[dict[str, Any]] = []
    for r in read_csv(in_v12):
        try:
            lat, lon, label = float(r["lat"]), float(r["lon"]), int(r["label"])
        except (KeyError, ValueError):
            continue
        matched = None
        for pid, (xmin, ymin, xmax, ymax) in bboxes.items():
            if pid in patch_ids and xmin <= lon <= xmax and ymin <= lat <= ymax:
                matched = pid
                break
        if matched is None:
            continue
        row: dict[str, Any] = {"point_id": r.get("point_id", ""), "label": label,
                               "patch_id": matched, "lat": lat, "lon": lon}
        for c in PHYS_COLS:
            try:
                row[c] = float(r[c])
            except (KeyError, ValueError, TypeError):
                row[c] = float("nan")
        joined.append(row)
    return joined


# Statistics helpers (no new dependency beyond numpy/scipy already in use)

def _spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    rho, p, _ = _spearman_n(a, b)
    return (rho, p)


def _spearman_n(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    from scipy import stats
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 4:
        return (float("nan"), float("nan"), int(mask.sum()))
    rho, p = stats.spearmanr(a[mask], b[mask])
    return (float(rho), float(p), int(mask.sum()))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(h)))


def _seed_separation(sim: np.ndarray, idx_h: list[int], idx_l: list[int]) -> float:
    def _mean_pairs(ii: list[int], jj: list[int], same: bool) -> float:
        vals = [sim[i, j] for i in ii for j in jj if not (same and i == j)]
        return float(np.mean(vals)) if vals else float("nan")
    return _mean_pairs(idx_h, idx_h, True) - _mean_pairs(idx_h, idx_l, False)


# Main

def _fail_closed(reason: str, detail: str) -> None:
    write_csv(OUT_SCORES, [], SCORE_FIELDS)
    write_csv(OUT_QUEUE, [], QUEUE_FIELDS)
    write_csv(OUT_QA, [{"qa_key": "status", "qa_value": reason},
                       {"qa_key": "detail", "qa_value": detail}], QA_FIELDS)
    print(f"[v1r7] FAIL_CLOSED: {reason} -- {detail}")


def run() -> None:
    in_v12_raw = os.environ.get(ENV_IN_V12, "") or os.environ.get(ENV_IN_V12_FALLBACK, "")
    sentinel_raw = os.environ.get(ENV_SENTINEL_DIR, "") or os.environ.get(ENV_SENTINEL_DIR_FALLBACK, "")
    if not in_v12_raw or not sentinel_raw:
        _fail_closed("FAIL_CLOSED_MISSING_PRIVATE_PATH_ENV",
                     f"set {ENV_IN_V12} and {ENV_SENTINEL_DIR} (private PROJETO paths)")
        return

    emb_rows = read_csv(IN_EMB)
    vec_by_patch: dict[str, np.ndarray] = {}
    prov_by_patch: dict[str, str] = {}
    for r in emb_rows:
        v = parse_embedding_from_row(r)
        if v is None:
            continue
        pid = r["patch_id"]
        vec_by_patch[pid] = np.asarray(v, dtype=float)
        prov_by_patch[pid] = "SEDEC_NEG_EVIDENCE_PATCH" if "_NEG_" in pid else "OFFICIAL_300_GRID_PATCH"

    patch_ids = sorted(vec_by_patch)
    if len(patch_ids) < 6:
        _fail_closed("FAIL_CLOSED_INSUFFICIENT_PATCHES", f"n_patches={len(patch_ids)}")
        return

    bboxes = _patch_geometry_wgs84(Path(sentinel_raw))
    joined = _join_points_to_patches(Path(in_v12_raw), bboxes, set(patch_ids))

    agg: dict[str, dict[str, Any]] = {
        pid: {"n": 0, "n_pos": 0, "phys": {c: [] for c in PHYS_COLS}} for pid in patch_ids}
    for j in joined:
        a = agg[j["patch_id"]]
        a["n"] += 1
        a["n_pos"] += int(j["label"] == 1)
        for c in PHYS_COLS:
            if np.isfinite(j[c]):
                a["phys"][c].append(j[c])

    # ---- Seeds (pre-registered thresholds) --------------------------------
    def _assign(min_support: int) -> dict[str, str]:
        roles: dict[str, str] = {}
        for pid in patch_ids:
            n, npos = agg[pid]["n"], agg[pid]["n_pos"]
            frac = (npos / n) if n else float("nan")
            if n >= min_support and n > 0 and frac >= PHI_H:
                roles[pid] = "POSITIVE_SEED"
            elif n >= min_support and n > 0 and frac <= PHI_L:
                roles[pid] = "NEGATIVE_SEED"
            else:
                roles[pid] = "REFINEMENT_TARGET"
        return roles

    roles = _assign(MIN_SUPPORT)
    roles_sens = _assign(1)

    idx = {pid: i for i, pid in enumerate(patch_ids)}
    x = np.vstack([vec_by_patch[p] for p in patch_ids])
    x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    sim = x @ x.T  # cosine, vectors already L2-normalized upstream

    seeds_h = [p for p in patch_ids if roles[p] == "POSITIVE_SEED"]
    seeds_l = [p for p in patch_ids if roles[p] == "NEGATIVE_SEED"]
    targets = [p for p in patch_ids if roles[p] == "REFINEMENT_TARGET"]

    if len(seeds_h) < 2 or len(seeds_l) < 2:
        _fail_closed("FAIL_CLOSED_INSUFFICIENT_SEEDS",
                     f"positive_seeds={len(seeds_h)} negative_seeds={len(seeds_l)} "
                     f"(pre-registered phi_H={PHI_H} phi_L={PHI_L} min_support={MIN_SUPPORT})")
        return

    ih = [idx[p] for p in seeds_h]
    il = [idx[p] for p in seeds_l]

    # H2O-Net's "adaptive" selection picks the densest seed class. Recorded
    # explicitly: with balanced seed sets it is a no-op and the signed score
    # uses both sides.
    if len(ih) > len(il) * 2:
        adaptive_side = "POSITIVE_SEEDS_DENSER"
    elif len(il) > len(ih) * 2:
        adaptive_side = "NEGATIVE_SEEDS_DENSER"
    else:
        adaptive_side = "BALANCED_BOTH_SIDES_USED_NO_OP"

    def _mean_cos(i: int, group: list[int]) -> float:
        vals = [sim[i, g] for g in group if g != i]
        return float(np.mean(vals)) if vals else float("nan")

    score_rows: list[dict[str, Any]] = []
    for pid in patch_ids:
        i = idx[pid]
        n, npos = agg[pid]["n"], agg[pid]["n_pos"]
        frac = (npos / n) if n else float("nan")
        mh, ml = _mean_cos(i, ih), _mean_cos(i, il)
        best = max(((sim[i, idx[s]], s) for s in seeds_h if s != pid), default=(float("nan"), ""))
        if n == 0:
            cand = "NO_SEDEC_POINT_IN_PATCH"
        elif npos == 0:
            cand = "NO_FLOOD_POSITIVE_IN_SEDEC_RECORD"
        elif roles[pid] == "REFINEMENT_TARGET":
            cand = "UNRESOLVED_MIXED_SEDEC_RECORD"
        else:
            cand = "SEED_NOT_A_CANDIDATE"
        score_rows.append({
            "patch_id": pid, "provenance": prov_by_patch[pid], "n_v12_points": n,
            "n_pos": npos, "n_neg": n - npos,
            "frac_positive": round(frac, 4) if n else "",
            "seed_role": roles[pid],
            "mean_cos_to_positive_seeds": round(mh, 6),
            "mean_cos_to_negative_seeds": round(ml, 6),
            "refinement_score": round(mh - ml, 6),
            "refinement_rank_among_targets": "",
            "nearest_positive_seed": best[1],
            "nearest_positive_seed_cos": round(float(best[0]), 6),
            "candidate_class": cand,
            "refinement_can_create_label": "false",
            "refinement_can_train_classifier": "false",
        })

    by_pid = {r["patch_id"]: r for r in score_rows}
    ranked = sorted(targets, key=lambda p: -by_pid[p]["refinement_score"])
    for k, pid in enumerate(ranked, start=1):
        by_pid[pid]["refinement_rank_among_targets"] = k

    # ---- QA ---------------------------------------------------------------
    rng = np.random.default_rng(RANDOM_SEED)
    obs_sep = _seed_separation(sim, ih, il)
    pool = ih + il
    n_h = len(ih)
    perm = np.empty(N_PERM)
    for b in range(N_PERM):
        sh = rng.permutation(pool)
        perm[b] = _seed_separation(sim, list(sh[:n_h]), list(sh[n_h:]))
    perm_p = float((np.sum(perm >= obs_sep) + 1) / (N_PERM + 1))

    # With this few seeds the permutation set is small enough to enumerate
    # exactly, which also pins down the smallest p this design can ever reach
    # (a power statement, reported whether the result is null or not).
    import itertools
    exact_vals = [_seed_separation(sim, list(c), [k for k in pool if k not in c])
                  for c in itertools.combinations(pool, n_h)] if len(pool) <= 20 else []
    if exact_vals:
        ev = np.asarray(exact_vals, dtype=float)
        exact_p = float(np.sum(ev >= obs_sep) / len(ev))
        exact_n = len(ev)
        exact_min_p = 1.0 / len(ev)
        exact_rank = int(np.sum(ev >= obs_sep))
    else:
        exact_p = exact_min_p = float("nan")
        exact_n = exact_rank = 0

    # Confound 1: geographic adjacency vs embedding similarity (pairwise).
    cent = {}
    for pid in patch_ids:
        xmin, ymin, xmax, ymax = bboxes[pid]
        cent[pid] = ((ymin + ymax) / 2.0, (xmin + xmax) / 2.0)
    pd_km, pd_cos = [], []
    for a_i in range(len(patch_ids)):
        for b_i in range(a_i + 1, len(patch_ids)):
            pa, pb = patch_ids[a_i], patch_ids[b_i]
            pd_km.append(_haversine_km(*cent[pa], *cent[pb]))
            pd_cos.append(sim[a_i, b_i])
    rho_geo, p_geo = _spearman(np.asarray(pd_km), np.asarray(pd_cos))

    # Confound 2/3: score vs provenance and vs number of points.
    tgt_scores = np.asarray([by_pid[p]["refinement_score"] for p in targets], dtype=float)
    tgt_npts = np.asarray([agg[p]["n"] for p in targets], dtype=float)
    rho_npts, p_npts = _spearman(tgt_scores, tgt_npts)
    is_neg_prov = np.asarray([1.0 if "_NEG_" in p else 0.0 for p in targets])
    rho_prov, p_prov = _spearman(tgt_scores, is_neg_prov)

    # Confound 4: score vs frac_positive of the target itself (does the score
    # recover the very quantity the seeds were built from?).
    tgt_frac = np.asarray([agg[p]["n_pos"] / agg[p]["n"] if agg[p]["n"] else np.nan
                           for p in targets], dtype=float)
    rho_frac, p_frac, n_frac = _spearman_n(tgt_scores, tgt_frac)

    # Same test over ALL patches (seeds included) -- the strongest single
    # descriptive check of whether the embedding tracks documented evidence.
    all_scores = np.asarray([by_pid[p]["refinement_score"] for p in patch_ids], dtype=float)
    all_frac = np.asarray([agg[p]["n_pos"] / agg[p]["n"] if agg[p]["n"] else np.nan
                           for p in patch_ids], dtype=float)
    rho_all, p_all, n_all = _spearman_n(all_scores, all_frac)

    # Redundancy with the causal base: is the DINO score just re-encoding terrain/rain?
    phys_rho: dict[str, tuple[float, float, int]] = {}
    for c in PHYS_COLS:
        vals = np.asarray([np.mean(agg[p]["phys"][c]) if agg[p]["phys"][c] else np.nan
                           for p in patch_ids], dtype=float)
        r_, p_, n_ = _spearman_n(all_scores, vals)
        phys_rho[c] = (r_, p_, n_)

    # Leave-one-seed-out stability of the target ranking.
    loo_rhos = []
    for drop in seeds_h + seeds_l:
        ih2 = [idx[p] for p in seeds_h if p != drop]
        il2 = [idx[p] for p in seeds_l if p != drop]
        if len(ih2) < 2 or len(il2) < 2:
            continue
        s2 = np.asarray([_mean_cos(idx[p], ih2) - _mean_cos(idx[p], il2) for p in targets])
        r, _ = _spearman(tgt_scores, s2)
        if np.isfinite(r):
            loo_rhos.append(r)
    loo_min = float(np.min(loo_rhos)) if loo_rhos else float("nan")
    loo_mean = float(np.mean(loo_rhos)) if loo_rhos else float("nan")

    # Pseudoreplication accounting (the failure mode that closed v1r5/v1r6).
    n_points_in_targets = int(sum(agg[p]["n"] for p in targets))
    top3 = ranked[:3]
    top3_points = int(sum(agg[p]["n"] for p in top3))
    conc = (top3_points / n_points_in_targets) if n_points_in_targets else float("nan")

    useful = (perm_p < 0.05) and (p_geo >= 0.05) and (p_prov >= 0.05) and (p_npts >= 0.05)
    verdict = ("REFINEMENT_SIGNAL_PRESENT_REVIEW_QUEUE_USABLE" if useful else
               "REFINEMENT_SIGNAL_NOT_ESTABLISHED_QUEUE_IS_UNORDERED_NULL_RESULT")

    qa = [
        ("preregistered_phi_H", str(PHI_H)),
        ("preregistered_phi_L", str(PHI_L)),
        ("preregistered_min_support_points", str(MIN_SUPPORT)),
        ("unit_of_analysis", "PATCH"),
        ("n_patches_total", str(len(patch_ids))),
        ("n_v12_points_joined", str(len(joined))),
        ("n_positive_seed_patches", str(len(seeds_h))),
        ("n_negative_seed_patches", str(len(seeds_l))),
        ("n_refinement_target_patches", str(len(targets))),
        ("positive_seed_patches", ";".join(seeds_h)),
        ("negative_seed_patches", ";".join(seeds_l)),
        ("adaptive_seed_side_h2onet_analogue", adaptive_side),
        ("similarity_space", "COSINE_768D_DINOV2_WITH_REGISTERS_L2_NORMALIZED"),
        ("cos_min_offdiag", f"{np.min(sim[~np.eye(len(patch_ids), dtype=bool)]):.6f}"),
        ("cos_max_offdiag", f"{np.max(sim[~np.eye(len(patch_ids), dtype=bool)]):.6f}"),
        ("cos_mean_offdiag", f"{np.mean(sim[~np.eye(len(patch_ids), dtype=bool)]):.6f}"),
        ("refinement_score_min", f"{np.nanmin(tgt_scores):.6f}"),
        ("refinement_score_max", f"{np.nanmax(tgt_scores):.6f}"),
        ("refinement_score_spread_targets", f"{np.nanmax(tgt_scores) - np.nanmin(tgt_scores):.6f}"),
        ("qa_seed_separation_observed", f"{obs_sep:.6f}"),
        ("qa_seed_separation_permutation_p", f"{perm_p:.4f}"),
        ("qa_seed_separation_permutation_n", str(N_PERM)),
        ("qa_seed_separation_exact_p", f"{exact_p:.4f}"),
        ("qa_seed_separation_exact_n_partitions", str(exact_n)),
        ("qa_seed_separation_exact_rank_of_observed",
         f"{exact_rank} of {exact_n} (1 = most separated split)"),
        ("qa_design_min_achievable_p", f"{exact_min_p:.4f}"),
        ("qa_confound_geographic_adjacency_spearman",
         f"rho={rho_geo:.4f} p={p_geo:.4f} n_pairs={len(pd_km)}"),
        ("qa_confound_patch_provenance_spearman", f"rho={rho_prov:.4f} p={p_prov:.4f}"),
        ("qa_confound_n_points_per_patch_spearman", f"rho={rho_npts:.4f} p={p_npts:.4f}"),
        ("qa_score_vs_frac_positive_targets_spearman",
         f"rho={rho_frac:.4f} p={p_frac:.4f} n_used={n_frac} of {len(targets)} targets"),
        ("qa_score_vs_frac_positive_all_patches_spearman",
         f"rho={rho_all:.4f} p={p_all:.4f} n_used={n_all} of {len(patch_ids)} patches"),
        ("qa_leave_one_seed_out_rank_stability", f"mean_rho={loo_mean:.4f} min_rho={loo_min:.4f}"),
        ("qa_pseudoreplication_cluster_n", str(len(patch_ids))),
        ("qa_pseudoreplication_points_behind_targets", str(n_points_in_targets)),
        ("qa_pseudoreplication_note",
         f"every quantity here is per-patch; the {n_points_in_targets} v12 points inside the "
         f"target patches are NOT independent observations of the DINO score -- effective n "
         f"for any DINO statement is {len(patch_ids)} patches, the same limit that closed v1r5/v1r6"),
        ("qa_top3_concentration_share_of_target_points", f"{conc:.4f}"),
        ("sensitivity_min_support_1_positive_seeds",
         str(sum(1 for p in patch_ids if roles_sens[p] == "POSITIVE_SEED"))),
        ("sensitivity_min_support_1_negative_seeds",
         str(sum(1 for p in patch_ids if roles_sens[p] == "NEGATIVE_SEED"))),
        ("strict_candidates_without_any_sedec_record",
         str(sum(1 for p in patch_ids if agg[p]["n"] == 0))),
        ("candidates_without_flood_positive_in_sedec",
         str(sum(1 for p in patch_ids if agg[p]["n"] > 0 and agg[p]["n_pos"] == 0))),
        ("labels_created", "0"),
        ("classifier_trained", "false"),
        ("new_embeddings_extracted", "0"),
        ("new_imagery_downloaded", "0"),
        ("preregistered_utility_criterion",
         "perm_p<0.05 AND geographic/provenance/n_points confounds all p>=0.05"),
        ("verdict", verdict),
    ]
    for c, (r, p, nn) in phys_rho.items():
        qa.append((f"qa_redundancy_with_causal_feature__{c}",
                   f"rho={r:.4f} p={p:.4f} n_used={nn}"))

    # ---- Review queue -----------------------------------------------------
    ordered = ranked if useful else sorted(targets)
    queue_rows = []
    for k, pid in enumerate(ordered, start=1):
        r = by_pid[pid]
        queue_rows.append({
            "review_rank": k if useful else "",
            "patch_id": pid, "candidate_class": r["candidate_class"],
            "refinement_score": r["refinement_score"],
            "n_v12_points": r["n_v12_points"], "n_pos": r["n_pos"], "n_neg": r["n_neg"],
            "frac_positive": r["frac_positive"],
            "nearest_positive_seed": r["nearest_positive_seed"],
            "nearest_positive_seed_cos": r["nearest_positive_seed_cos"],
            "review_status": "PENDING_HUMAN_REVIEW",
            "promotion_allowed": "false",
            "evidence_role": "REVIEW_QUEUE_PRIORITIZATION_ONLY",
            "blocking_note": ("ordering is descriptive only; DINO is auxiliary evidence, "
                              "never causal" if useful else
                              "ordering SUPPRESSED: refinement signal failed the pre-registered "
                              "utility criterion; rows listed in patch_id order, not by score"),
        })

    write_csv(OUT_SCORES, score_rows, SCORE_FIELDS)
    write_schema(SCH_SCORES, SCORE_FIELDS, "v1r7_dino_recife_evidence_refinement_scores")
    write_csv(OUT_QUEUE, queue_rows, QUEUE_FIELDS)
    write_csv(OUT_QA, [{"qa_key": k, "qa_value": v} for k, v in qa], QA_FIELDS)
    write_csv(OUT_BOUNDARY, [{
        "boundary_id": "DINO_EVIDENCE_REFINEMENT_BOUNDARY_V1R7",
        "refinement_can_create_label": "false",
        "refinement_can_train_classifier": "false",
        "refinement_can_validate_event": "false",
        "refinement_can_enter_firth_model": "false",
        "refinement_can_prioritize_review": "true",
        "refinement_role": "REVIEW_QUEUE_PRIORITIZATION_ONLY",
        "seed_source": "SEDEC_CAUSALLY_VALIDATED_POINTS_V12_NOT_MNDWI",
        "similarity_space": "COSINE_DINOV2_WITH_REGISTERS_768D",
        "unit_of_analysis": "PATCH",
        "cluster_n": str(len(patch_ids)),
        "pseudoreplication_checked": "true",
        "supersedes": "none (extends DINO_TRAINING_BOUNDARY_V1NN with the refinement role)",
        "blocker": ("REFINEMENT_SIGNAL_NOT_ESTABLISHED" if not useful else
                    "HUMAN_REVIEW_AND_SUPERVISOR_DECISION_REQUIRED_BEFORE_ANY_USE"),
        "notes": ("Adapted from H2O-Net (arXiv:2010.05309) sec 3.4: seed thresholds and signed "
                  "seed-distance kept; pixel-space distance replaced by embedding cosine; the "
                  "refiner CNN and pseudo-mask generation were DELIBERATELY DROPPED because they "
                  "create labels. No candidate here is a positive, a label, or a model feature."),
    }], BOUNDARY_FIELDS)

    write_doc(DOC, "v1r7 (SUSC-21) -- Refinamento de evidencia DINO em Recife, adaptado do H2O-Net", [
        "## Papel do DINO nesta etapa",
        "O teste A/B de v1r5/v1r6 permanece fechado: DINO nao e feature do "
        "modelo causal (Wald conjunto cluster-robusto p=0.1752 sobre 23 "
        "patches). Esta etapa nao reabre aquela decisao; ela muda o papel do "
        "DINO para evidencia a ser refinada, cujo unico produto permitido e "
        "ordenacao de fila de revisao humana.",
        "## Adaptacao do H2O-Net (o que foi mantido e o que foi descartado)",
        "Mantido: a estrutura de sementes de alta confianca com limiar alto "
        "(phi_H) e baixo (phi_L) sobre um indicador continuo, o conjunto "
        "explicitamente nao-resolvido, e o score assinado (proximidade ao "
        "positivo menos proximidade ao negativo), que e o que o mapa de "
        "distancia adaptativo normalizado codifica. "
        "Adaptado: no H2O-Net a distancia e euclidiana em COORDENADA DE PIXEL "
        "dentro de uma imagem, valida por contiguidade espacial da agua; aqui "
        "a unidade e o patch inteiro e contiguidade geografica seria "
        "confundidor, entao a distancia passou para o espaco de embedding "
        "(cosseno 768D) e a adjacencia geografica virou teste obrigatorio de "
        "QA. A semente tambem e mais forte: pontos SEDEC do v12 em vez de "
        "MNDWI limiarizado. "
        "Descartado: o refiner CNN e a pseudo-mascara que supervisiona a rede "
        "de segmentacao -- esse passo GERA label, proibido no projeto. O "
        "pipeline para no ranqueamento.",
        "## Pre-registro",
        f"phi_H={PHI_H}, phi_L={PHI_L}, suporte minimo={MIN_SUPPORT} pontos por "
        f"patch, unidade = patch, criterio de utilidade fixado antes de rodar: "
        f"permutacao p<0.05 e nenhum confundidor (adjacencia geografica, "
        f"proveniencia do patch, n de pontos) significativo. Sensibilidade com "
        f"suporte minimo 1 reportada junto.",
        "## Resultado",
        f"{len(seeds_h)} patches semente positiva, {len(seeds_l)} semente "
        f"negativa, {len(targets)} alvos de refinamento sobre {len(patch_ids)} "
        f"patches. Separacao semente-a-semente observada={obs_sep:.4f} "
        f"(permutacao p={perm_p:.4f}, {N_PERM} sorteios; enumeracao exata das "
        f"{exact_n} particoes: p={exact_p:.4f}, posicao {exact_rank}, menor p "
        f"alcancavel por este desenho={exact_min_p:.4f}). Score x "
        f"frac_positive em todos os patches: rho={rho_all:.4f} p={p_all:.4f}. "
        f"Confundidores: adjacencia geografica rho={rho_geo:.4f} p={p_geo:.4f}; "
        f"proveniencia rho={rho_prov:.4f} p={p_prov:.4f}; n de pontos "
        f"rho={rho_npts:.4f} p={p_npts:.4f}. Veredito: {verdict}.",
        "## Limitacoes",
        f"n efetivo = {len(patch_ids)} patches, nao os {len(joined)} pontos "
        f"v12 que caem dentro deles -- a mesma pseudorreplicacao que fechou "
        f"v1r5/v1r6 continua sendo o teto desta analise. Nenhum candidato aqui "
        f"e positivo, label ou feature; a promocao de qualquer linha depende de "
        f"revisao humana e decisao explicita de orientacao. Recife apenas; "
        f"nenhum embedding novo, nenhuma imagem nova.",
    ])

    print(f"[v1r7] patches={len(patch_ids)} seeds_h={len(seeds_h)} seeds_l={len(seeds_l)} "
          f"targets={len(targets)} perm_p={perm_p:.4f} rho_all={rho_all:.4f} "
          f"geo_rho={rho_geo:.4f}(p={p_geo:.4f}) verdict={verdict}")


if __name__ == "__main__":
    run()
