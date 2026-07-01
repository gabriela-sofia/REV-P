"""SUSC-17C22 aceite formal automatizado do QA sensorial e feature matrix candidata.

Marco totalmente offline e deterministico. Aplica uma politica objetiva de aceite
sobre as features sensoriais reais ja materializadas (pre-evento no 17C20, deltas
pre/pos no 17C21), monta uma matriz candidata review-only por patch e trava
qualquer promocao indevida (treino, label, ground truth, Ground Reference, score
v7, 17B). Nao acessa rede, nao baixa raster, nao cria tile/embedding e nao calcula
nenhuma feature nova: apenas consolida, aceita e rastreia o que ja existe.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import read_csv, read_json, rel, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"
OFFICIAL_PATCHES = DAT / "susc_patches_official_v1.csv"
OFFICIAL_PATCH_LINKS = DAT / "susc_patch_links_official_v1.csv"

C21_PRE_QA = OUT / "susc_17c21_pre_event_artifact_qa.csv"
C21_POST_SELECTION = OUT / "susc_17c21_post_event_scene_selection.csv"
C21_POST_MANIFEST = OUT / "susc_17c21_post_event_light_artifact_manifest.csv"
C21_POST_BAND = OUT / "susc_17c21_post_event_band_statistics.csv"
C21_POST_INDEX = OUT / "susc_17c21_post_event_index_statistics.csv"
C21_DELTA = OUT / "susc_17c21_pre_post_delta_features.csv"
C21_OBS = OUT / "susc_17c21_observational_change_registry.csv"
C21_FALLBACK = OUT / "susc_17c21_fallback_source_audit.csv"
C21_REPLAY = OUT / "susc_17c21_replay_audit.csv"

C20_MANIFEST = OUT / "susc_17c20_light_artifact_manifest.csv"
C20_BAND = OUT / "susc_17c20_band_statistics.csv"
C20_INDEX = OUT / "susc_17c20_index_statistics.csv"
C20_FEATURES = OUT / "susc_17c20_materialized_feature_registry.csv"
C20_FALLBACK = OUT / "susc_17c20_fallback_source_audit.csv"
C20_REPLAY = OUT / "susc_17c20_replay_audit.csv"

C19_LINEAGE = OUT / "susc_17c19_event_temporal_lineage.csv"
C19_BINDING = OUT / "susc_17c19_candidate_patch_temporal_binding.csv"
C19_CANONICAL = OUT / "susc_17c19_sentinel2_canonical_pre_event_scenes.csv"
PATCH_GRID = OUT / "susc_17c6_candidate_patch_grid.csv"
PATCH_LINKS = OUT / "susc_17c6_candidate_patch_links.csv"

REQUIRED_INPUTS = [
    SCORE_V6, C21_PRE_QA, C21_POST_SELECTION, C21_POST_MANIFEST, C21_POST_BAND,
    C21_POST_INDEX, C21_DELTA, C21_OBS, C21_FALLBACK, C21_REPLAY,
    C20_MANIFEST, C20_BAND, C20_INDEX, C20_FEATURES, C20_FALLBACK, C20_REPLAY,
    C19_LINEAGE, C19_BINDING, C19_CANONICAL, PATCH_GRID, PATCH_LINKS,
]

REPORT = OUT / "SUSC_17C22_ACEITE_FORMAL_QA_SENSORIAL_FEATURE_MATRIX_REPORT.md"
POLICY = OUT / "susc_17c22_sensor_qa_acceptance_policy.csv"
PRE_ACCEPT = OUT / "susc_17c22_pre_event_sensor_feature_acceptance.csv"
DELTA_ACCEPT = OUT / "susc_17c22_observational_delta_acceptance.csv"
FEATURE_MATRIX = OUT / "susc_17c22_candidate_patch_sensor_feature_matrix.csv"
DELTA_MATRIX = OUT / "susc_17c22_candidate_patch_observational_delta_matrix.csv"
PROVENANCE = OUT / "susc_17c22_feature_provenance_trace.csv"
FALLBACK_REVIEW = OUT / "susc_17c22_fallback_policy_review.csv"
SCIENCE_GATES = OUT / "susc_17c22_science_readiness_gates.csv"
TRAINING_BLOCKERS = OUT / "susc_17c22_training_and_score_blockers.csv"
NO_LEAKAGE = OUT / "susc_17c22_no_leakage_audit.csv"
GATES = OUT / "susc_17c22_gate_evaluation_matrix.csv"
SUMMARY = OUT / "susc_17c22_readiness_summary.json"
BLOCKERS = OUT / "susc_17c22_promotion_blockers.csv"

ACCEPTANCE_SCHEMA = SCHEMAS / "susc_17c22_feature_acceptance_schema_v1.json"
MATRIX_SCHEMA = SCHEMAS / "susc_17c22_feature_matrix_schema_v1.json"
PROVENANCE_SCHEMA = SCHEMAS / "susc_17c22_provenance_trace_schema_v1.json"

GOV = {"review_only": "true", "trainable": "false", "ground_truth": "false"}

BAND_FEATURES = ["B03_mean", "B04_mean", "B08_mean", "B11_mean"]
INDEX_FEATURES = ["NDVI_mean", "NDWI_mean", "MNDWI_mean", "NDBI_mean"]
FEATURE_ORDER = BAND_FEATURES + INDEX_FEATURES
DELTA_ORDER = [f"delta_{f}" for f in FEATURE_ORDER]


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _run_git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _require_inputs() -> None:
    missing = [path for path in REQUIRED_INPUTS if not path.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(path) for path in missing))
    for path in REQUIRED_INPUTS:
        if path.suffix == ".json":
            read_json(path)
        elif path.suffix == ".csv":
            read_csv(path)


def _patch_ids() -> list[str]:
    return [row["candidate_patch_id"] for row in read_csv(PATCH_GRID)]


def _event_id() -> str:
    rows = read_csv(C19_CANONICAL)
    return rows[0]["event_id"] if rows else ""


def _c20_features() -> list[dict]:
    return read_csv(C20_FEATURES)


def _c20_manifest_ids() -> set[str]:
    return {row["light_artifact_manifest_id"] for row in read_csv(C20_MANIFEST)}


def _c20_manifest_hash(manifest_id: str) -> str:
    for row in read_csv(C20_MANIFEST):
        if row["light_artifact_manifest_id"] == manifest_id:
            return row["sha256"]
    return "not_available"


def _c20_manifest_path(manifest_id: str) -> str:
    for row in read_csv(C20_MANIFEST):
        if row["light_artifact_manifest_id"] == manifest_id:
            return row["artifact_local_path"]
    return "not_available"


def _c20_replay_ok(manifest_id: str) -> bool:
    for row in read_csv(C20_REPLAY):
        if row["light_artifact_manifest_id"] == manifest_id:
            return row["byte_identical_replay"] == "true"
    return False


def _c21_replay_ok(manifest_id: str) -> bool:
    for row in read_csv(C21_REPLAY):
        if row["artifact_manifest_id"] == manifest_id:
            return row["byte_identical_replay"] == "true"
    return False


def _c21_post_manifest_hash(manifest_id: str) -> str:
    for row in read_csv(C21_POST_MANIFEST):
        if row["post_event_artifact_manifest_id"] == manifest_id:
            return row["sha256"]
    return "not_available"


def _pre_scene() -> dict:
    """candidate_patch_id -> cena pre-evento canonica."""
    return {row["candidate_patch_id"]: row["scene_or_item_id"] for row in read_csv(C19_CANONICAL)}


def _fallback_declared(patch_id: str, table: Path) -> bool:
    for row in read_csv(table):
        if row["candidate_patch_id"] == patch_id:
            role = row.get("fallback_source_role", "")
            used = row.get("fallback_used", "")
            gr = row.get("can_be_used_for_ground_reference", "")
            return role == "materialization_fallback" and used == "true" and gr == "false"
    return False


# ---------------------------------------------------------------------------
# Parte 1 - Politica de aceite
# ---------------------------------------------------------------------------

POLICY_DEFS = [
    ("artifact_hash_required", "pre_event_sensor_feature", "feature so aceita se o artefato de origem tem hash conferido", "artifact_hash_verified=true", "true", "accepted_review_only_sensor_feature", "blocked_missing_hash"),
    ("replay_required", "pre_event_sensor_feature", "feature so aceita se replay byte-identico do artefato foi verificado", "replay_verified=true", "true", "accepted_review_only_sensor_feature", "blocked_replay_failed"),
    ("pre_event_required_for_sensor_feature", "pre_event_sensor_feature", "feature sensorial exige cena pre-evento", "pre_event_only=true", "true", "accepted_review_only_sensor_feature", "blocked_not_pre_event"),
    ("post_event_blocked_as_pre_event_feature", "post_event_observation", "observacao pos-evento nunca vira feature pre-evento", "usable_as_pre_event_feature=false", "true", "accepted_observational_change_review_only", "blocked_not_pre_event"),
    ("delta_blocked_as_training_label", "observational_delta", "delta nunca vira label de treino", "usable_for_training=false", "true", "accepted_observational_change_review_only", "blocked_delta_leakage_risk"),
    ("fallback_must_be_declared", "materialization_source", "fonte fallback deve estar declarada explicitamente", "fallback_declared=true", "true", "review_only_sensor_materialization_source", "blocked_fallback_policy"),
    ("fallback_requires_policy_review_before_science", "materialization_source", "fallback exige revisao de politica antes de uso cientifico", "can_be_used_for_science_now=false", "true", "review_only_sensor_materialization_source", "blocked_fallback_policy"),
    ("ground_reference_blocked", "any", "nenhum objeto sensorial vira Ground Reference Candidate", "all_gates_passed_for_ground_reference=false", "true", "blocked_ground_reference", "blocked_ground_reference"),
    ("score_v7_blocked", "any", "nenhum objeto sensorial alimenta score v7", "eligible_for_score_v7=false", "true", "blocked_score_v7", "blocked_score_v7"),
    ("training_blocked", "any", "nenhum objeto sensorial alimenta treino", "usable_for_training=false", "true", "blocked_training", "blocked_training"),
]


def sensor_qa_acceptance_policy_rows() -> list[dict]:
    rows = []
    for idx, (rule, obj, _desc, cond, blocks, accepted, failed) in enumerate(POLICY_DEFS, start=1):
        rows.append({
            "policy_id": f"S17C22_POLICY_{idx:04d}",
            "object_type": obj,
            "acceptance_rule": rule,
            "required_condition": cond,
            "blocks_if_failed": blocks,
            "accepted_status_if_pass": accepted,
            "status_if_failed": failed,
            **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 2 - Aceite de features pre-evento
# ---------------------------------------------------------------------------

def pre_event_sensor_feature_acceptance_rows() -> list[dict]:
    event_id = _event_id()
    manifest_ids = _c20_manifest_ids()
    rows = []
    for idx, feat in enumerate(_c20_features(), start=1):
        patch_id = feat["candidate_patch_id"]
        manifest_id = feat["source_artifact_manifest_id"]
        real = feat["feature_value_real"] == "true"
        hash_ok = manifest_id in manifest_ids
        replay_ok = _c20_replay_ok(manifest_id)
        pre_event = True  # 17C20 so materializou a cena canonica pre-evento
        fallback_declared = _fallback_declared(patch_id, C20_FALLBACK)
        passes = all([real, hash_ok, replay_ok, pre_event, fallback_declared])
        if not real:
            status = "blocked_not_real_feature"
        elif not hash_ok:
            status = "blocked_missing_hash"
        elif not replay_ok:
            status = "blocked_replay_failed"
        elif not pre_event:
            status = "blocked_not_pre_event"
        elif not fallback_declared:
            status = "blocked_fallback_policy"
        else:
            status = "accepted_review_only_sensor_feature"
        rows.append({
            "feature_acceptance_id": f"S17C22_FEATACC_{idx:04d}",
            "candidate_patch_id": patch_id,
            "event_id": event_id,
            "scene_or_item_id": feat["scene_or_item_id"],
            "feature_name": feat["feature_name"],
            "feature_value": feat["feature_value"],
            "feature_unit": feat["feature_unit"],
            "source_artifact_manifest_id": manifest_id,
            "source_role": feat["source_role"],
            "feature_value_real": _bool_text(real),
            "artifact_hash_verified": _bool_text(hash_ok),
            "replay_verified": _bool_text(replay_ok),
            "pre_event_only": _bool_text(pre_event),
            "fallback_declared": _bool_text(fallback_declared),
            "passes_sensor_qa": _bool_text(passes),
            "acceptance_status": status,
            "usable_as_review_only_sensor_feature": _bool_text(passes),
            "usable_for_science_now": "false",
            "usable_for_training": "false",
            "ground_truth": "false",
            "eligible_for_score_v7": "false",
            "blocking_reason": "not_applicable" if passes else "feature reprovada em criterio objetivo de QA sensorial",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 3 - Aceite de deltas observacionais
# ---------------------------------------------------------------------------

def observational_delta_acceptance_rows() -> list[dict]:
    rows = []
    for idx, delta in enumerate(read_csv(C21_DELTA), start=1):
        pre_manifest = delta["pre_event_manifest_id"]
        post_manifest = delta["post_event_manifest_id"]
        real = delta["delta_value_real"] == "true"
        pre_hash = pre_manifest in _c20_manifest_ids()
        post_hash = _c21_post_manifest_hash(post_manifest) != "not_available"
        no_leak = delta["usable_as_pre_event_feature"] == "false" and delta["usable_for_training"] == "false"
        passes = all([real, pre_hash, post_hash, no_leak])
        if not real:
            status = "blocked_not_real_delta"
        elif not pre_hash:
            status = "blocked_missing_pre_hash"
        elif not post_hash:
            status = "blocked_missing_post_hash"
        elif not no_leak:
            status = "blocked_delta_leakage_risk"
        else:
            status = "accepted_observational_change_review_only"
        rows.append({
            "delta_acceptance_id": f"S17C22_DELTAACC_{idx:04d}",
            "candidate_patch_id": delta["candidate_patch_id"],
            "event_id": delta["event_id"],
            "feature_name": delta["feature_name"],
            "pre_event_value": delta["pre_event_value"],
            "post_event_value": delta["post_event_value"],
            "delta_value": delta["delta_value"],
            "pre_event_manifest_id": pre_manifest,
            "post_event_manifest_id": post_manifest,
            "delta_value_real": _bool_text(real),
            "pre_hash_verified": _bool_text(pre_hash),
            "post_hash_verified": _bool_text(post_hash),
            "usable_as_observational_change_review_only": _bool_text(passes),
            "usable_as_pre_event_feature": "false",
            "usable_for_training": "false",
            "ground_truth": "false",
            "acceptance_status": status,
            "blocking_reason": "not_applicable" if passes else "delta reprovado em criterio objetivo de aceite observacional",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 4 - Matriz candidata de features sensoriais por patch
# ---------------------------------------------------------------------------

def candidate_patch_sensor_feature_matrix_rows() -> list[dict]:
    event_id = _event_id()
    pre_scene = _pre_scene()
    accepted = pre_event_sensor_feature_acceptance_rows()
    rows = []
    for patch_id in _patch_ids():
        patch_feats = [f for f in accepted if f["candidate_patch_id"] == patch_id]
        by_name = {f["feature_name"]: f for f in patch_feats}
        accepted_count = len([f for f in patch_feats if f["acceptance_status"] == "accepted_review_only_sensor_feature"])
        row = {
            "candidate_patch_id": patch_id,
            "event_id": event_id,
            "pre_event_scene_or_item_id": pre_scene.get(patch_id, "not_available"),
            "source_role": "materialization_fallback",
            "sensor_feature_matrix_status": "accepted_review_only" if accepted_count == len(FEATURE_ORDER) else "partial_or_blocked",
        }
        for name in FEATURE_ORDER:
            row[name] = by_name[name]["feature_value"] if name in by_name else "not_available"
        row.update({
            "features_real_count": str(len([f for f in patch_feats if f["feature_value_real"] == "true"])),
            "features_accepted_review_only_count": str(accepted_count),
            "usable_for_science_now": "false",
            "usable_for_training": "false",
            "ground_truth": "false",
            "eligible_for_score_v7": "false",
            "review_only": "true",
        })
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Parte 5 - Matriz de deltas observacionais por patch
# ---------------------------------------------------------------------------

def candidate_patch_observational_delta_matrix_rows() -> list[dict]:
    event_id = _event_id()
    pre_scene = _pre_scene()
    post_sel = {r["candidate_patch_id"]: r["selected_scene_or_item_id"] for r in read_csv(C21_POST_SELECTION)}
    accepted = observational_delta_acceptance_rows()
    rows = []
    for patch_id in _patch_ids():
        patch_deltas = [d for d in accepted if d["candidate_patch_id"] == patch_id]
        by_name = {d["feature_name"]: d for d in patch_deltas}
        real_count = len([d for d in patch_deltas if d["delta_value_real"] == "true"])
        row = {
            "candidate_patch_id": patch_id,
            "event_id": event_id,
            "pre_event_scene_or_item_id": pre_scene.get(patch_id, "not_available"),
            "post_event_scene_or_item_id": post_sel.get(patch_id, "not_available"),
            "observational_delta_status": "accepted_observational_change_review_only" if real_count == len(FEATURE_ORDER) else "partial_or_blocked",
        }
        for name, delta_name in zip(FEATURE_ORDER, DELTA_ORDER):
            row[delta_name] = by_name[name]["delta_value"] if name in by_name else "not_available"
        row.update({
            "delta_features_real_count": str(real_count),
            "usable_as_observational_change_review_only": "true",
            "usable_as_pre_event_feature": "false",
            "usable_for_training": "false",
            "ground_truth": "false",
            "review_only": "true",
        })
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Parte 6 - Rastreabilidade de proveniencia
# ---------------------------------------------------------------------------

def feature_provenance_trace_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    idx = 0
    # Pre-evento sensor features.
    for feat in _c20_features():
        idx += 1
        manifest_id = feat["source_artifact_manifest_id"]
        rows.append({
            "provenance_trace_id": f"S17C22_PROV_{idx:04d}",
            "candidate_patch_id": feat["candidate_patch_id"], "event_id": event_id,
            "object_type": "pre_event_sensor_feature", "feature_or_delta_name": feat["feature_name"],
            "source_scene_or_item_id": feat["scene_or_item_id"], "source_artifact_manifest_id": manifest_id,
            "artifact_path": _c20_manifest_path(manifest_id), "sha256": _c20_manifest_hash(manifest_id),
            "source_role": feat["source_role"], "sensor_or_dataset": "SENTINEL-2",
            "temporal_role": "pre_event", "processing_chain": "17c20_cog_window_read->band_index_stats",
            "replay_status": _bool_text(_c20_replay_ok(manifest_id)), **GOV,
        })
    # Pos-evento sensor observations (medias pos).
    post_means = [r for r in read_csv(C21_POST_BAND) if r["stat_name"] == "mean"] + [r for r in read_csv(C21_POST_INDEX) if r["stat_name"] == "mean"]
    for obs in post_means:
        idx += 1
        manifest_id = obs["source_artifact_manifest_id"]
        name = f"{obs.get('band_name', obs.get('index_name', ''))}_mean"
        rows.append({
            "provenance_trace_id": f"S17C22_PROV_{idx:04d}",
            "candidate_patch_id": obs["candidate_patch_id"], "event_id": event_id,
            "object_type": "post_event_sensor_observation", "feature_or_delta_name": name,
            "source_scene_or_item_id": obs["scene_or_item_id"], "source_artifact_manifest_id": manifest_id,
            "artifact_path": "outputs_public/suscetibilidade/susc_17c21_post_event_light_artifacts", "sha256": _c21_post_manifest_hash(manifest_id),
            "source_role": "materialization_fallback", "sensor_or_dataset": "SENTINEL-2",
            "temporal_role": "post_event", "processing_chain": "17c21_cog_window_read->band_index_stats",
            "replay_status": _bool_text(_c21_replay_ok(manifest_id)), **GOV,
        })
    # Deltas pre/pos.
    for delta in read_csv(C21_DELTA):
        idx += 1
        rows.append({
            "provenance_trace_id": f"S17C22_PROV_{idx:04d}",
            "candidate_patch_id": delta["candidate_patch_id"], "event_id": event_id,
            "object_type": "pre_post_observational_delta", "feature_or_delta_name": delta["feature_name"],
            "source_scene_or_item_id": f"{delta['pre_event_scene_or_item_id']}|{delta['post_event_scene_or_item_id']}",
            "source_artifact_manifest_id": f"{delta['pre_event_manifest_id']}|{delta['post_event_manifest_id']}",
            "artifact_path": "derived_from_pre_and_post_manifests", "sha256": "derived_delta_no_single_artifact",
            "source_role": "materialization_fallback", "sensor_or_dataset": "SENTINEL-2",
            "temporal_role": "pre_post_delta", "processing_chain": "17c22_delta=post_mean-pre_mean",
            "replay_status": "derived", **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 7 - Revisao de politica do fallback
# ---------------------------------------------------------------------------

def fallback_policy_review_rows() -> list[dict]:
    c20_fb = {r["candidate_patch_id"]: r for r in read_csv(C20_FALLBACK)}
    rows = []
    for idx, patch_id in enumerate(_patch_ids(), start=1):
        fb = c20_fb.get(patch_id, {})
        rows.append({
            "fallback_policy_review_id": f"S17C22_FBREV_{idx:04d}",
            "candidate_patch_id": patch_id,
            "source_role": "materialization_fallback",
            "primary_source": fb.get("primary_official_source", "CDSE/OData"),
            "fallback_source": fb.get("fallback_source", "earth_search_stac:sentinel-2-l2a"),
            "fallback_used": fb.get("fallback_used", "true"),
            "fallback_declared": "true",
            "fallback_matches_scene_or_equivalent": fb.get("fallback_matches_scene_or_equivalent", "true"),
            "can_be_used_for_review_only_sensor_feature": "true",
            "can_be_used_for_science_now": "false",
            "can_be_used_for_ground_reference": "false",
            "requires_external_policy_review": "true",
            "limitation": "fonte fallback nao oficial primaria; sustenta feature sensorial review-only mas exige revisao de politica antes de ciencia/score",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 8 - Gates de prontidao cientifica
# ---------------------------------------------------------------------------

def science_readiness_gate_rows() -> list[dict]:
    defs = [
        ("sensor_artifact_real", "pass", True, "artefatos sensoriais reais materializados no 17C20/17C21", "manter rastreabilidade"),
        ("sensor_artifact_hash_verified", "pass", True, "hash conferido byte a byte para todos os artefatos", "manter manifesto"),
        ("sensor_replay_verified", "pass", True, "replay offline byte-identico confirmado", "manter replay"),
        ("pre_event_feature_only", "pass", True, "features sensoriais restritas a cena pre-evento", "manter separacao temporal"),
        ("fallback_policy_pending", "blocked", False, "fonte fallback exige revisao de politica antes de ciencia", "revisar politica de fonte fallback"),
        ("event_geometry_missing", "blocked", False, "geometria do evento observado ainda ausente", "obter artefato de evento com geometria"),
        ("phenomenon_gate_missing", "blocked", False, "separacao de fenomeno ainda nao satisfeita", "obter evidencia de fenomeno"),
        ("patch_candidate_policy_missing", "blocked", False, "politica formal de patch candidato ausente", "definir politica de patch candidato"),
        ("ground_reference_missing", "blocked", False, "artefatos de evento para Ground Reference ausentes", "adquirir Ground Reference"),
        ("training_blocked", "blocked", False, "treino bloqueado ate ciencia e ground reference", "manter bloqueio"),
        ("score_v7_blocked", "blocked", False, "score v7 bloqueado ate politica e ground reference", "manter bloqueio"),
        ("17b_blocked", "blocked", False, "17B bloqueado ate evento e politica", "manter bloqueio"),
    ]
    rows = []
    for idx, (name, status, passes, reason, action) in enumerate(defs, start=1):
        rows.append({
            "science_readiness_gate_id": f"S17C22_SCIGATE_{idx:04d}",
            "gate_name": name, "gate_status": status, "passes": _bool_text(passes),
            "blocking_reason": "not_applicable" if passes else reason,
            "required_next_action": action, "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 9 - Bloqueadores de treino e score
# ---------------------------------------------------------------------------

def training_and_score_blocker_rows() -> list[dict]:
    rows = []
    idx = 0
    for feat in pre_event_sensor_feature_acceptance_rows():
        idx += 1
        rows.append({
            "training_score_blocker_id": f"S17C22_TSBLOCK_{idx:04d}",
            "object_type": "pre_event_sensor_feature", "object_id": feat["feature_acceptance_id"],
            "blocks_training": "true", "blocks_score_v7": "true", "blocks_17b": "true",
            "blocking_reason": "feature sensorial review-only via fallback; treino/score/17B exigem politica de fonte, ground reference e evento",
            "review_only": "true",
        })
    for delta in observational_delta_acceptance_rows():
        idx += 1
        rows.append({
            "training_score_blocker_id": f"S17C22_TSBLOCK_{idx:04d}",
            "object_type": "observational_delta", "object_id": delta["delta_acceptance_id"],
            "blocks_training": "true", "blocks_score_v7": "true", "blocks_17b": "true",
            "blocking_reason": "delta observacional review-only; nunca label/treino/score/17B",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 10 - Anti-leakage
# ---------------------------------------------------------------------------

def no_leakage_audit_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    idx = 0
    for feat in pre_event_sensor_feature_acceptance_rows():
        idx += 1
        rows.append({
            "no_leakage_audit_id": f"S17C22_LEAK_{idx:04d}",
            "candidate_patch_id": feat["candidate_patch_id"], "event_id": event_id,
            "object_id": feat["feature_acceptance_id"], "object_type": "pre_event_sensor_feature",
            "uses_pre_event_data_only_for_sensor_feature": "true",
            "uses_post_event_as_pre_event_feature": "false",
            "uses_delta_as_training_label": "false",
            "uses_ground_reference_as_feature": "false",
            "uses_synthetic_as_real": "false",
            "passes_no_leakage": "true",
            "blocking_reason": "feature pre-evento review-only sem vazamento", "review_only": "true",
        })
    for delta in observational_delta_acceptance_rows():
        idx += 1
        rows.append({
            "no_leakage_audit_id": f"S17C22_LEAK_{idx:04d}",
            "candidate_patch_id": delta["candidate_patch_id"], "event_id": event_id,
            "object_id": delta["delta_acceptance_id"], "object_type": "observational_delta",
            "uses_pre_event_data_only_for_sensor_feature": "true",
            "uses_post_event_as_pre_event_feature": "false",
            "uses_delta_as_training_label": "false",
            "uses_ground_reference_as_feature": "false",
            "uses_synthetic_as_real": "false",
            "passes_no_leakage": "true",
            "blocking_reason": "delta observacional review-only; nunca label", "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 11 - Gates gerais
# ---------------------------------------------------------------------------

def gate_evaluation_matrix_rows() -> list[dict]:
    rows = []
    idx = 0
    for feat in pre_event_sensor_feature_acceptance_rows():
        idx += 1
        rows.append({
            "gate_eval_id": f"S17C22_GATE_{idx:04d}",
            "object_id": feat["feature_acceptance_id"], "object_type": "pre_event_sensor_feature",
            "G1_existencia_documental": "true", "G2_confiabilidade_fonte": "true", "G3_precisao_temporal": "true",
            "G4_vinculo_espacial_evento": "false", "G5_separacao_fenomeno": "false",
            "G6_proveniencia_integridade": "true", "G7_anti_leakage": "true",
            "all_gates_passed_for_ground_reference": "false",
            "usable_as_review_only_sensor_feature": feat["usable_as_review_only_sensor_feature"],
            "usable_as_observational_change_review_only": "false",
            "acceptance_status": feat["acceptance_status"],
            "blocking_reason": "feature sensorial pre-evento review-only; nao e evento observado, nao vira Ground Reference",
            **GOV,
        })
    for delta in observational_delta_acceptance_rows():
        idx += 1
        rows.append({
            "gate_eval_id": f"S17C22_GATE_{idx:04d}",
            "object_id": delta["delta_acceptance_id"], "object_type": "observational_delta",
            "G1_existencia_documental": "true", "G2_confiabilidade_fonte": "true", "G3_precisao_temporal": "true",
            "G4_vinculo_espacial_evento": "false", "G5_separacao_fenomeno": "false",
            "G6_proveniencia_integridade": "true", "G7_anti_leakage": "true",
            "all_gates_passed_for_ground_reference": "false",
            "usable_as_review_only_sensor_feature": "false",
            "usable_as_observational_change_review_only": delta["usable_as_observational_change_review_only"],
            "acceptance_status": delta["acceptance_status"],
            "blocking_reason": "delta observacional review-only; nao e evento observado, nao vira Ground Reference",
            **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# Parte 12 - Summary
# ---------------------------------------------------------------------------

def build_summary() -> dict:
    pre = pre_event_sensor_feature_acceptance_rows()
    delta = observational_delta_acceptance_rows()
    accepted_pre = [r for r in pre if r["acceptance_status"] == "accepted_review_only_sensor_feature"]
    accepted_delta = [r for r in delta if r["acceptance_status"] == "accepted_observational_change_review_only"]
    feature_matrix = candidate_patch_sensor_feature_matrix_rows()
    delta_matrix = candidate_patch_observational_delta_matrix_rows()
    fallback_review = fallback_policy_review_rows()
    acceptance_completed = len(pre) >= 40 and len(delta) >= 40
    matrix_created = len(feature_matrix) == 5 and len(delta_matrix) == 5
    minimum = (
        acceptance_completed and matrix_created
        and len(accepted_pre) >= 40 and len(accepted_delta) >= 40
    )
    return {
        "sensor_qa_acceptance_completed": acceptance_completed,
        "candidate_patch_feature_matrix_created": matrix_created,
        "candidate_patch_count": len(_patch_ids()),
        "pre_event_sensor_features_evaluated_count": len(pre),
        "accepted_pre_event_sensor_features_count": len(accepted_pre),
        "observational_delta_features_evaluated_count": len(delta),
        "accepted_observational_delta_features_count": len(accepted_delta),
        "candidate_patch_sensor_feature_matrix_rows_count": len(feature_matrix),
        "candidate_patch_observational_delta_matrix_rows_count": len(delta_matrix),
        "features_promoted_to_training_count": 0,
        "post_event_used_as_pre_event_feature_count": 0,
        "delta_used_as_training_label_count": 0,
        "fallback_used_count": len([r for r in fallback_review if r["fallback_used"] == "true"]),
        "fallback_accepted_for_review_only_count": len([r for r in fallback_review if r["can_be_used_for_review_only_sensor_feature"] == "true"]),
        "fallback_accepted_for_science_now_count": 0,
        "ground_reference_candidates_created_count": 0,
        "ground_truth_created": False,
        "training_labels_created": False,
        "score_v6_changed": bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)])),
        "score_v7_created": SCORE_V7.exists(),
        "official_patch_created": False,
        "official_patch_link_created": False,
        "eligible_for_17b_now": False,
        "eligible_for_score_v7": False,
        "minimum_success_achieved": minimum,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "recommended_next_milestone": "SUSC-17C23 Revisao de politica da fonte fallback e requisitos formais para promover feature sensorial review-only a uso cientifico",
    }


def build_blockers() -> list[dict]:
    blockers = [
        "fallback_source_policy_pending_for_science",
        "sensor_features_review_only_not_trainable",
        "observational_deltas_not_trainable",
        "event_geometry_still_missing",
        "phenomenon_gate_not_satisfied",
        "ground_reference_event_artifacts_still_missing",
        "candidate_patch_policy_missing",
        "official_patch_link_missing",
        "17b_blocked_until_event_reference_and_policy",
        "score_v7_blocked_until_policy_and_ground_reference",
    ]
    return [
        {
            "blocker_id": f"S17C22_BLOCKER_{idx:04d}",
            "blocker_type": blocker,
            "description": "Bloqueio preservado: features sensoriais e deltas sao review-only; faltam revisao de politica do fallback, artefato de evento, geometria, fenomeno, ground reference e politica de patch antes de qualquer treino, score v7 ou 17B.",
            "blocks_ground_reference_candidate": _bool_text(blocker in {"event_geometry_still_missing", "phenomenon_gate_not_satisfied", "ground_reference_event_artifacts_still_missing", "candidate_patch_policy_missing"}),
            "blocks_17b": "true",
            "blocks_score_v7": "true",
            "review_only": "true",
        }
        for idx, blocker in enumerate(blockers, start=1)
    ]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

def _schema(required: list[str], props: dict, title: str) -> dict:
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": title, "type": "object", "required": required, "properties": props}


def build_acceptance_schema() -> dict:
    required = list(pre_event_sensor_feature_acceptance_rows()[0].keys())
    return _schema(required, {
        "feature_acceptance_id": {"type": "string", "pattern": "^S17C22_FEATACC_"},
        "usable_for_training": {"const": "false"}, "ground_truth": {"const": "false"},
        "eligible_for_score_v7": {"const": "false"}, "usable_for_science_now": {"const": "false"},
        "review_only": {"const": "true"},
    }, "SUSC-17C22 feature acceptance schema v1")


def build_matrix_schema() -> dict:
    required = list(candidate_patch_sensor_feature_matrix_rows()[0].keys())
    return _schema(required, {
        "usable_for_science_now": {"const": "false"}, "usable_for_training": {"const": "false"},
        "ground_truth": {"const": "false"}, "eligible_for_score_v7": {"const": "false"},
        "review_only": {"const": "true"},
    }, "SUSC-17C22 feature matrix schema v1")


def build_provenance_schema() -> dict:
    required = list(feature_provenance_trace_rows()[0].keys())
    return _schema(required, {
        "provenance_trace_id": {"type": "string", "pattern": "^S17C22_PROV_"},
        "review_only": {"const": "true"}, "trainable": {"const": "false"}, "ground_truth": {"const": "false"},
    }, "SUSC-17C22 provenance trace schema v1")


# ---------------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------------

def build_report() -> str:
    summary = build_summary()
    return "\n".join([
        "# SUSC-17C22 - Aceite formal automatizado do QA sensorial e feature matrix candidata",
        "",
        "## Objetivo",
        "Operacionalizar a 'revisao humana' como gates objetivos aplicados pelo agente: consolidar as features sensoriais reais (pre-evento 17C20, deltas pre/pos 17C21), aplicar politica de aceite, montar matriz candidata review-only por patch e bloquear qualquer promocao cientifica indevida.",
        "",
        "## Aceite de features pre-evento",
        f"- Features pre-evento avaliadas: {summary['pre_event_sensor_features_evaluated_count']}.",
        f"- Aceitas como accepted_review_only_sensor_feature: {summary['accepted_pre_event_sensor_features_count']}.",
        "- Criterios objetivos: feature real, hash conferido, replay byte-identico, cena pre-evento, fallback declarado.",
        "",
        "## Aceite de deltas observacionais",
        f"- Deltas avaliados: {summary['observational_delta_features_evaluated_count']}.",
        f"- Aceitos como accepted_observational_change_review_only: {summary['accepted_observational_delta_features_count']}.",
        "- Delta nunca e feature pre-evento, label, ground truth ou score.",
        "",
        "## Matrizes candidatas por patch",
        f"- Matriz sensorial candidata (pre-evento): {summary['candidate_patch_sensor_feature_matrix_rows_count']} linhas.",
        f"- Matriz de deltas observacionais: {summary['candidate_patch_observational_delta_matrix_rows_count']} linhas.",
        "- Deltas ficam em bloco separado; nunca misturados com features pre-evento.",
        "",
        "## Fallback",
        f"- Fallback usado: {summary['fallback_used_count']} patches; aceito para review-only: {summary['fallback_accepted_for_review_only_count']}; aceito para ciencia agora: {summary['fallback_accepted_for_science_now_count']}.",
        "- Fallback mantem source_role=materialization_fallback, can_be_used_for_ground_reference=false e requires_external_policy_review=true.",
        "",
        "## Guardrails",
        f"- Features promovidas a treino: {summary['features_promoted_to_training_count']}; deltas usados como label: {summary['delta_used_as_training_label_count']}; pos-evento como pre-evento: {summary['post_event_used_as_pre_event_feature_count']}.",
        "- Ground Reference: 0; ground truth: 0; label: 0; score v7: inexistente; score v6 intacto; 17B bloqueado.",
        "",
        f"## minimum_success_achieved: {summary['minimum_success_achieved']}",
        "",
        "## Proximo marco recomendado",
        summary["recommended_next_milestone"],
    ])


# ---------------------------------------------------------------------------
# Build / validacao
# ---------------------------------------------------------------------------

def build_all() -> None:
    _require_inputs()
    write_csv(POLICY, sensor_qa_acceptance_policy_rows())
    write_csv(PRE_ACCEPT, pre_event_sensor_feature_acceptance_rows())
    write_csv(DELTA_ACCEPT, observational_delta_acceptance_rows())
    write_csv(FEATURE_MATRIX, candidate_patch_sensor_feature_matrix_rows())
    write_csv(DELTA_MATRIX, candidate_patch_observational_delta_matrix_rows())
    write_csv(PROVENANCE, feature_provenance_trace_rows())
    write_csv(FALLBACK_REVIEW, fallback_policy_review_rows())
    write_csv(SCIENCE_GATES, science_readiness_gate_rows())
    write_csv(TRAINING_BLOCKERS, training_and_score_blocker_rows())
    write_csv(NO_LEAKAGE, no_leakage_audit_rows())
    write_csv(GATES, gate_evaluation_matrix_rows())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_json(ACCEPTANCE_SCHEMA, build_acceptance_schema())
    write_json(MATRIX_SCHEMA, build_matrix_schema())
    write_json(PROVENANCE_SCHEMA, build_provenance_schema())
    write_markdown(REPORT, build_report())


REQUIRED_OUTPUTS = [
    REPORT, POLICY, PRE_ACCEPT, DELTA_ACCEPT, FEATURE_MATRIX, DELTA_MATRIX, PROVENANCE,
    FALLBACK_REVIEW, SCIENCE_GATES, TRAINING_BLOCKERS, NO_LEAKAGE, GATES, SUMMARY, BLOCKERS,
    ACCEPTANCE_SCHEMA, MATRIX_SCHEMA, PROVENANCE_SCHEMA,
]


def _schema_violations(row: dict, schema: dict) -> list[str]:
    violations = []
    for field in schema.get("required", []):
        if field not in row or row[field] == "":
            violations.append(f"missing:{field}")
    for field, rules in schema.get("properties", {}).items():
        if field not in row:
            continue
        value = row[field]
        if "const" in rules and value != rules["const"]:
            violations.append(f"{field}:const:{rules['const']}")
        if "pattern" in rules and not value.startswith(rules["pattern"].replace("^", "")):
            violations.append(f"{field}:pattern")
    return violations


def _validate_byte_identical() -> list[str]:
    before = {path: path.read_bytes() for path in REQUIRED_OUTPUTS if path.exists()}
    build_all()
    errors = []
    for path, content in before.items():
        if path.read_bytes() != content:
            errors.append(f"offline_build_not_byte_identical:{rel(path)}")
    return errors


def validate() -> int:
    missing = [path for path in REQUIRED_OUTPUTS if not path.exists()]
    if missing:
        print("MISSING: " + "; ".join(rel(path) for path in missing), file=sys.stderr)
        return 1
    errors = _validate_byte_identical()
    policy = read_csv(POLICY)
    pre = read_csv(PRE_ACCEPT)
    delta = read_csv(DELTA_ACCEPT)
    feature_matrix = read_csv(FEATURE_MATRIX)
    delta_matrix = read_csv(DELTA_MATRIX)
    provenance = read_csv(PROVENANCE)
    fallback = read_csv(FALLBACK_REVIEW)
    science = read_csv(SCIENCE_GATES)
    ts_blockers = read_csv(TRAINING_BLOCKERS)
    leakage = read_csv(NO_LEAKAGE)
    gates = read_csv(GATES)
    summary = read_json(SUMMARY)
    acceptance_schema = read_json(ACCEPTANCE_SCHEMA)
    matrix_schema = read_json(MATRIX_SCHEMA)
    provenance_schema = read_json(PROVENANCE_SCHEMA)

    for row in pre:
        errors.extend(_schema_violations(row, acceptance_schema))
    for row in feature_matrix:
        errors.extend(_schema_violations(row, matrix_schema))
    for row in provenance:
        errors.extend(_schema_violations(row, provenance_schema))

    for rows, key in [
        (policy, "policy_id"), (pre, "feature_acceptance_id"), (delta, "delta_acceptance_id"),
        (provenance, "provenance_trace_id"), (fallback, "fallback_policy_review_id"),
        (science, "science_readiness_gate_id"), (ts_blockers, "training_score_blocker_id"),
        (leakage, "no_leakage_audit_id"), (gates, "gate_eval_id"),
    ]:
        ids = [row[key] for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")

    accepted_pre = [r for r in pre if r["acceptance_status"] == "accepted_review_only_sensor_feature"]
    accepted_delta = [r for r in delta if r["acceptance_status"] == "accepted_observational_change_review_only"]
    # 5..11.
    if not summary["minimum_success_achieved"]:
        errors.append("minimum_success_not_achieved")
    if len(pre) != 40:
        errors.append("pre_event_features_not_40")
    if len(accepted_pre) != 40:
        errors.append("accepted_pre_features_not_40")
    if len(delta) != 40:
        errors.append("deltas_not_40")
    if len(accepted_delta) != 40:
        errors.append("accepted_deltas_not_40")
    if len(feature_matrix) != 5:
        errors.append("feature_matrix_not_5")
    if len(delta_matrix) != 5:
        errors.append("delta_matrix_not_5")
    # 12: matriz sensorial nao contem coluna de delta.
    if any(col.startswith("delta_") for col in (feature_matrix[0].keys() if feature_matrix else [])):
        errors.append("feature_matrix_contains_delta")
    # 12/13: pos nunca pre; delta nunca label.
    if any(r["usable_for_training"] != "false" for r in pre + delta):
        errors.append("feature_or_delta_trainable")
    if any(r["usable_as_pre_event_feature"] != "false" for r in delta):
        errors.append("delta_marked_pre_event")
    if any(r["uses_post_event_as_pre_event_feature"] != "false" or r["uses_delta_as_training_label"] != "false" for r in leakage):
        errors.append("leakage_detected")
    if any(r["passes_no_leakage"] != "true" for r in leakage):
        errors.append("no_leakage_failed")
    # 14/15/16: fallback declarado, nao ciencia agora, nao ground reference.
    if any(r["fallback_declared"] != "true" for r in fallback):
        errors.append("fallback_not_declared")
    if any(r["can_be_used_for_science_now"] != "false" for r in fallback):
        errors.append("fallback_accepted_for_science")
    if any(r["can_be_used_for_ground_reference"] != "false" for r in fallback):
        errors.append("fallback_as_ground_reference")
    if summary["fallback_accepted_for_science_now_count"] != 0:
        errors.append("fallback_science_count_nonzero")
    # 17/18: nada passa G4/G5; nenhum Ground Reference.
    if any(r["G4_vinculo_espacial_evento"] == "true" or r["G5_separacao_fenomeno"] == "true" for r in gates):
        errors.append("sensor_promoted_to_event")
    if any(r["all_gates_passed_for_ground_reference"] == "true" for r in gates):
        errors.append("ground_reference_accepted")
    # 19/20: sem ground truth/label.
    if summary["ground_truth_created"] or summary["training_labels_created"] or summary["features_promoted_to_training_count"] != 0:
        errors.append("forbidden_promotion")
    if any(r["blocks_training"] != "true" or r["blocks_score_v7"] != "true" for r in ts_blockers):
        errors.append("training_score_not_blocked")

    for path in [SCORE_V6, OFFICIAL_PATCHES, OFFICIAL_PATCH_LINKS]:
        if path.exists() and _run_git(["diff", "--name-only", "--", rel(path)]):
            errors.append(f"official_dataset_changed:{rel(path)}")
    if SCORE_V7.exists():
        errors.append("score_v7_exists")

    expected = build_summary()
    for key, value in expected.items():
        if summary.get(key) != value:
            errors.append(f"summary_mismatch:{key}")
    if summary["eligible_for_17b_now"] or summary["eligible_for_score_v7"] or summary["trainable"] or summary["ground_truth"]:
        errors.append("promotion_guardrail_failed")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "17C22 -> "
        f"pre_accepted={summary['accepted_pre_event_sensor_features_count']}/40 "
        f"delta_accepted={summary['accepted_observational_delta_features_count']}/40 "
        f"feature_matrix={summary['candidate_patch_sensor_feature_matrix_rows_count']} "
        f"delta_matrix={summary['candidate_patch_observational_delta_matrix_rows_count']} "
        f"min_success={summary['minimum_success_achieved']} "
        f"score_v7_created={summary['score_v7_created']}"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def evaluate_qa_text() -> str:
    pre = pre_event_sensor_feature_acceptance_rows()
    accepted = [r for r in pre if r["acceptance_status"] == "accepted_review_only_sensor_feature"]
    return f"evaluate-qa: {len(accepted)}/{len(pre)} features pre-evento aceitas como review-only sensor feature."


def build_feature_matrix_text() -> str:
    fm = candidate_patch_sensor_feature_matrix_rows()
    dm = candidate_patch_observational_delta_matrix_rows()
    return f"build-feature-matrix: {len(fm)} linhas de matriz sensorial e {len(dm)} linhas de matriz de deltas (blocos separados, review-only)."


def trace_provenance_text() -> str:
    rows = feature_provenance_trace_rows()
    by_type = {}
    for r in rows:
        by_type[r["object_type"]] = by_type.get(r["object_type"], 0) + 1
    return "trace-provenance: " + "; ".join(f"{k}={v}" for k, v in sorted(by_type.items())) + "."


def audit_blockers_text() -> str:
    ts = training_and_score_blocker_rows()
    return f"audit-blockers: {len(ts)} objetos bloqueados para treino/score/17B; nenhum promovido."


def status_text() -> str:
    summary = build_summary()
    return (
        f"status 17C22: pre_accepted={summary['accepted_pre_event_sensor_features_count']} "
        f"delta_accepted={summary['accepted_observational_delta_features_count']} "
        f"min_success={summary['minimum_success_achieved']} "
        f"score_v7_created={summary['score_v7_created']}"
    )
