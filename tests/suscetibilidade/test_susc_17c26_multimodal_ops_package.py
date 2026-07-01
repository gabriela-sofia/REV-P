"""Testes do SUSC-17C26 - pacote multimodal operacional, evidence graph e fila G4/G5."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"
DOSSIER_DIR = OUT / "susc_17c26_patch_dossiers"
PACKET_DIR = OUT / "susc_17c26_evidence_packets"
CHECKLIST_DIR = OUT / "susc_17c26_review_checklists"
QUERY_DIR = OUT / "susc_17c26_source_query_packages"

DOSSIER_SECTIONS = [
    "# Dossiê multimodal", "## Identificação", "## Evento e janela temporal", "## AOI e patch candidato",
    "## Sentinel-2 multitemporal", "## CHIRPS antecedente", "## Delta observacional pré/pós",
    "## Artefatos e hashes", "## Interpretação permitida", "## Interpretação proibida",
    "## Lacunas G4/G5", "## Campos necessários para Ground Reference", "## Próximas buscas objetivas",
    "## Status final",
]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str, *args: str):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        cwd=ROOT, text=True, capture_output=True, timeout=600, check=False,
    )


def _patch_ids():
    return [r["candidate_patch_id"] for r in read_csv(OUT / "susc_17c6_candidate_patch_grid.csv")]


def test_build_offline_byte_identico_e_valida():
    result = run_script("build_susc_17c26_multimodal_ops_package.py")
    assert result.returncode == 0, result.stderr + result.stdout
    tracked = [
        OUT / "susc_17c26_readiness_summary.json", OUT / "susc_17c26_evidence_graph_nodes.csv",
        OUT / "susc_17c26_evidence_graph_edges.csv", OUT / "susc_17c26_ground_reference_target_queue.csv",
    ] + [DOSSIER_DIR / f"{p.lower()}_dossie_multimodal.md" for p in _patch_ids()]
    first = {path: path.read_bytes() for path in tracked}
    result = run_script("build_susc_17c26_multimodal_ops_package.py")
    assert result.returncode == 0, result.stderr + result.stdout
    assert {path: path.read_bytes() for path in tracked} == first
    result = run_script("validate_susc_17c26_multimodal_ops_package.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_5_dossies_com_secoes_obrigatorias():
    assert len(_patch_ids()) == 5
    for patch_id in _patch_ids():
        path = DOSSIER_DIR / f"{patch_id.lower()}_dossie_multimodal.md"
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        for section in DOSSIER_SECTIONS:
            assert section in text, (patch_id, section)
        assert "scientific_review_only_not_ground_reference" in text


def test_5_evidence_packets_completos():
    registry = read_csv(OUT / "susc_17c26_patch_evidence_packet_registry.csv")
    assert len(registry) == 5
    assert {r["packet_complete"] for r in registry} == {"true"}
    for patch_id in _patch_ids():
        path = PACKET_DIR / f"{patch_id.lower()}_evidence_packet.json"
        assert path.exists()
        packet = json.loads(path.read_text(encoding="utf-8"))
        assert packet["g4_status"] is False and packet["g5_status"] is False
        assert packet["can_be_ground_reference_now"] is False
        assert packet["can_be_training_label"] is False
        assert packet["review_only"] is True


def test_evidence_graph_tem_nos_e_arestas():
    nodes = read_csv(OUT / "susc_17c26_evidence_graph_nodes.csv")
    edges = read_csv(OUT / "susc_17c26_evidence_graph_edges.csv")
    assert len(nodes) > 0 and len(edges) > 0
    node_ids = {n["node_id"] for n in nodes}
    for e in edges:
        assert e["source_node_id"] in node_ids and e["target_node_id"] in node_ids
    types = {n["node_type"] for n in nodes}
    assert {"candidate_patch", "event", "sentinel2_feature", "chirps_feature", "artifact", "source", "gate", "ground_reference_target"} <= types


def test_prioridades_targets_queries_checklists():
    priority = read_csv(OUT / "susc_17c26_patch_review_priority.csv")
    assert len(priority) == 5
    assert {r["not_risk_score"] for r in priority} == {"true"}
    assert {r["not_training_label"] for r in priority} == {"true"}
    targets = read_csv(OUT / "susc_17c26_ground_reference_target_queue.csv")
    assert len(targets) >= 15
    for patch_id in _patch_ids():
        assert len([t for t in targets if t["candidate_patch_id"] == patch_id]) >= 3
    queries = read_csv(OUT / "susc_17c26_source_query_packages.csv")
    assert len(queries) >= 15
    checklists = read_csv(OUT / "susc_17c26_review_checklist_registry.csv")
    assert len(checklists) == 5
    for patch_id in _patch_ids():
        assert (CHECKLIST_DIR / f"{patch_id.lower()}_checklist_revisao_g4_g5.md").exists()


def test_master_index_e_reproducibility_manifest():
    assert (OUT / "susc_17c26_master_index.md").exists()
    manifest = json.loads((OUT / "susc_17c26_reproducibility_manifest.json").read_text(encoding="utf-8"))
    assert manifest["not_training"] is True and manifest["not_ground_truth"] is True
    assert manifest["not_score_v7"] is True and manifest["not_17b"] is True
    assert manifest["main_output_hashes"]


def test_g4_g5_false_e_nada_vira_ground_reference_ou_evento():
    gaps = read_csv(OUT / "susc_17c26_g4_g5_gap_matrix.csv")
    gates = read_csv(OUT / "susc_17c26_gate_evaluation_matrix.csv")
    assert {r["G4_vinculo_espacial_evento"] for r in gaps} == {"false"}
    assert {r["G5_separacao_fenomeno"] for r in gaps} == {"false"}
    assert {r["G4_vinculo_espacial_evento"] for r in gates} == {"false"}
    assert {r["all_gates_passed_for_ground_reference"] for r in gates} == {"false"}
    targets = read_csv(OUT / "susc_17c26_ground_reference_target_queue.csv")
    assert {r["can_be_ground_reference_now"] for r in targets} == {"false"}
    leakage = read_csv(OUT / "susc_17c26_no_leakage_audit.csv")
    assert {r["uses_chirps_as_event_reference"] for r in leakage} == {"false"}
    assert {r["uses_sensor_as_ground_reference"] for r in leakage} == {"false"}
    assert {r["uses_delta_as_training_label"] for r in leakage} == {"false"}
    assert {r["passes_no_leakage"] for r in leakage} == {"true"}


def test_hashes_verificados_e_score_guardrails():
    recheck = read_csv(OUT / "susc_17c26_artifact_hash_recheck.csv")
    assert len(recheck) > 0
    assert {r["sha256_matches"] for r in recheck} == {"true"}
    summary = json.loads((OUT / "susc_17c26_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["minimum_success_achieved"] is True
    assert summary["all_hashes_verified"] is True
    assert summary["G4_true_count"] == 0 and summary["G5_true_count"] == 0
    assert summary["features_promoted_to_training_count"] == 0
    assert summary["sensor_used_as_ground_reference_count"] == 0
    assert summary["chirps_used_as_event_reference_count"] == 0
    assert summary["ground_reference_candidates_created_count"] == 0
    assert summary["ground_truth_created"] is False
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["eligible_for_17b_now"] is False
    assert summary["official_patch_created"] is False
    assert not (DAT / "susc_score_v7_candidate_by_patch_v1.csv").exists()
    for target in [
        DAT / "susc_score_v6_candidate_by_patch_v1.csv",
        DAT / "susc_patches_official_v1.csv",
        DAT / "susc_patch_links_official_v1.csv",
    ]:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--", str(target.relative_to(ROOT))],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        assert result.stdout.strip() == ""
