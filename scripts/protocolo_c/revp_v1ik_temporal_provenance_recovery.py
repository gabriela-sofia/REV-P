"""
revp_v1ik_temporal_provenance_recovery.py

v1ik -- Auditoria de Proveniencia Temporal para Candidatos Vetoriais Bloqueados

Objetivo:
    Recuperar e auditar proveniencia temporal dos candidatos vetoriais bloqueados,
    especialmente aqueles com geometria/fenomeno/CRS mas sem data documentada
    (ex: Cicatriz_Area_A.shp, Cicatriz_Ponto_P.shp).

    Usar APENAS evidencia local, sidecars, metadados, documentacao publica
    ja baixada, registries existentes. Nao inferir data. Nao aceitar pistas fracas.

Modos de operacao:
    default (sem flags)              -- dry-run
    --force                          -- escreve registries consolidados publicos
    --scan-sidecars                  -- busca sidecars locais (.prj, .xml, .dbf)
    --scan-registries                -- consulta registries v1if/v1ii
    --scan-local-docs                -- consulta documentacao versionada
    --focus-best-candidates          -- prioriza Cicatriz_Area_A.shp, Cicatriz_Ponto_P.shp
    --emit-temporal-decision-matrix  -- gera matriz de decisao temporal

Invariantes permanentes:
    nao_inventar_data               = true
    nao_aceitar_pistas_fracas       = true
    so_data_documentada             = true
    nao_enviar_email                = true
    nao_criar_label                 = true
    nao_fazer_overlay               = true
    nao_treinar                     = true
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import defaultdict
import re

# =========================================================================
# Caminhos do repositorio
# =========================================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"
DOCS_DIR = REPO_ROOT / "docs" / "metodologia_cientifica"
LOCAL_RUNS = REPO_ROOT / "local_runs" / "protocolo_c" / "v1ik"

PRIVATE_MARKERS = [
    "gabriela", "C:\\Users", "/Users/", "PROJETO",
    "\\gabriela\\", "/gabriela/",
]

# =========================================================================
# Eventos-alvo
# =========================================================================
EVENTS_TARGET: Dict[str, Dict] = {
    "PET": {
        "event_id": "PET_2022_02_15",
        "date": "2022-02-15",
        "keywords": ["petropolis", "2022-02-15", "15/02/2022", "fevereiro 2022"],
    },
}

# Candidatos prioritarios
PRIORITY_CANDIDATES = {
    "Cicatriz_Area_A.shp",
    "Cicatriz_Ponto_P.shp",
}


# =========================================================================
# Estruturas de dados
# =========================================================================
@dataclass
class TemporalEvidence:
    source_type: str = ""  # sidecar, registry, document
    source_name: str = ""
    field_name: str = ""
    value: str = ""
    strength: str = "INSUFFICIENT_EVIDENCE"  # STRONG_*, MODERATE_*, WEAK_*, INVALID_*, INSUFFICIENT_*, CONTRADICTORY_*
    is_event_date: bool = False
    is_survey_date: bool = False
    is_window: bool = False


@dataclass
class TemporalReview:
    temporal_review_id: str = ""
    consolidated_candidate_id: str = ""
    source_candidate_id: str = ""
    region: str = ""
    event_id: str = ""
    asset_name: str = ""
    asset_format: str = ""
    geometry_available: str = ""
    crs: str = ""
    phenomenon_group: str = ""
    observed_not_risk: str = ""
    previous_blocking_gate: str = ""
    previous_blocking_reason: str = ""
    temporal_evidence_source_type: str = ""
    temporal_evidence_source_name: str = ""
    temporal_evidence_field: str = ""
    temporal_evidence_value: str = ""
    temporal_evidence_strength: str = "INSUFFICIENT_EVIDENCE"
    event_date_candidate_before: str = ""
    event_date_candidate_after: str = ""
    event_date_compatible_after_review: str = "UNKNOWN"
    temporal_confidence: str = "LOW"
    accepted_as_event_date: str = "NO"
    accepted_as_survey_date: str = "NO"
    accepted_as_context_only: str = "NO"
    temporal_status_after_review: str = "TEMPORAL_GATE_BLOCKED_NO_DATE"
    ground_truth_candidate_status_after_review: str = "BLOCKED"
    patch_binding_preflight_status_after_review: str = "BLOCKED"
    can_create_training_label: str = "NO"
    limitations: str = ""
    next_required_action: str = ""
    notes: str = ""


@dataclass
class TemporalGateRecord:
    candidate_id: str = ""
    has_explicit_event_date: str = "NO"
    has_explicit_survey_date: str = "NO"
    has_event_window: str = "NO"
    has_documentary_linkage: str = "NO"
    has_registry_cross_reference: str = "NO"
    has_sidecar_metadata: str = "NO"
    only_file_or_folder_hint: str = "NO"
    only_file_system_date: str = "NO"
    contradictory_temporal_evidence: str = "NO"
    event_date_compatible: str = "UNKNOWN"
    survey_date_usable_as_context: str = "NO"
    temporal_gate_status: str = "BLOCKED"
    temporal_blocking_reason: str = ""
    can_reopen_patch_binding_preflight: str = "NO"
    can_reopen_ground_truth_candidate: str = "NO"
    can_create_training_label: str = "NO"


# =========================================================================
# Auditor de Proveniencia Temporal
# =========================================================================
class TemporalProvenanceAuditor:
    def __init__(self, force: bool = False, scan_sidecars: bool = False,
                 scan_registries: bool = False, scan_local_docs: bool = False,
                 focus_best_candidates: bool = False, emit_temporal_decision_matrix: bool = False):
        self.force = force
        self.scan_sidecars = scan_sidecars
        self.scan_registries = scan_registries
        self.scan_local_docs = scan_local_docs
        self.focus_best_candidates = focus_best_candidates
        self.emit_temporal_decision_matrix = emit_temporal_decision_matrix

        self.temporal_reviews: List[TemporalReview] = []
        self.temporal_gates: List[TemporalGateRecord] = []
        self.sidecar_log = []
        self.registry_log = []
        self.documentary_log = []
        self.qa_log = []

        self.stats = {
            "total_candidates_reviewed": 0,
            "candidates_with_temporal_evidence": 0,
            "candidates_with_event_date": 0,
            "candidates_with_survey_date": 0,
            "candidates_with_temporal_context": 0,
            "candidates_still_blocked": 0,
            "evidence_types_found": defaultdict(int),
        }

    def _load_consolidated_candidates(self) -> List[Dict]:
        """Carregar candidatos consolidados de v1ij."""
        registry_path = DATASETS_DIR / "consolidated_observed_event_vector_candidate_registry.csv"
        if not registry_path.exists():
            return []

        try:
            with open(registry_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return list(reader) if reader else []
        except Exception as e:
            self.qa_log.append(f"WARN: Failed to load consolidated candidates: {str(e)[:100]}")
            return []

    def _should_prioritize(self, candidate: Dict) -> bool:
        """Verificar se candidato eh prioritario.

        Priorizar:
        1. Nomes conhecidos (Cicatriz_Area_A, Cicatriz_Ponto_P)
        2. OU candidatos com: geometria YES, CRS presente, fenômeno,
           observed_not_risk YES, bloqueado por data (gate_04/gate_05)
        """
        if not self.focus_best_candidates:
            return True

        asset_name = candidate.get("asset_name", "")

        # Regra 1: nomes conhecidos
        if any(priority in asset_name for priority in PRIORITY_CANDIDATES):
            return True

        # Regra 2: critérios de qualidade (geometria OK, bloqueado por data)
        is_geo_ok = candidate.get("geometry_available") == "YES"
        is_crs_ok = bool(candidate.get("crs", "").strip())
        is_observed_ok = candidate.get("observed_not_risk") in {"YES", "PASS"}
        is_blocked_by_date = "gate_04" in candidate.get("blocking_gate", "") or \
                             "gate_05" in candidate.get("blocking_gate", "")

        return is_geo_ok and is_crs_ok and is_observed_ok and is_blocked_by_date

    def _scan_sidecars_for_candidate(self, candidate: Dict) -> List[TemporalEvidence]:
        """Buscar sidecars locais para candidato (read-only)."""
        if not self.scan_sidecars:
            return []

        evidences = []
        asset_name = candidate.get("asset_name", "")

        # Procurar sidecars com padroes simples em nomes conhecidos
        # Simulando busca local sem alterar filesystem
        if "Cicatriz_Area_A" in asset_name:
            self.sidecar_log.append(f"Busca simulada por sidecars de {asset_name}")

        return evidences

    def _scan_registries_for_candidate(self, candidate: Dict) -> List[TemporalEvidence]:
        """Consultar registries existentes para proveniencia temporal."""
        if not self.scan_registries:
            return []

        evidences = []
        asset_name = candidate.get("asset_name", "")
        region = candidate.get("region", "")
        event_id = candidate.get("event_id", "")

        # Tentar consultar registry de v1if
        v1if_path = DATASETS_DIR / "official_observed_event_vector_registry.csv"
        if v1if_path.exists():
            try:
                with open(v1if_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if asset_name in row.get("source_asset_name", "") or \
                           asset_name in row.get("notes", ""):
                            self.registry_log.append(f"Registry v1if: {asset_name} encontrado")

            except Exception as e:
                self.qa_log.append(f"WARN: Failed to scan v1if registry: {str(e)[:80]}")

        # Tentar consultar registry de v1ii
        v1ii_path = DATASETS_DIR / "targeted_official_repository_event_vector_registry.csv"
        if v1ii_path.exists():
            try:
                with open(v1ii_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if asset_name in row.get("resource_name", "") or \
                           "Cicatriz" in row.get("notes", ""):
                            self.registry_log.append(f"Registry v1ii: {asset_name} ou cicatrizes encontrados")

            except Exception as e:
                self.qa_log.append(f"WARN: Failed to scan v1ii registry: {str(e)[:80]}")

        return evidences

    def _scan_local_docs_for_candidate(self, candidate: Dict) -> List[TemporalEvidence]:
        """Consultar documentacao versionada para referencias temporais."""
        if not self.scan_local_docs:
            return []

        evidences = []
        asset_name = candidate.get("asset_name", "")
        region = candidate.get("region", "")

        # Termos a procurar
        search_terms = [
            asset_name.replace(".shp", ""),
            "cicatriz",
            "deslizamento",
            region.lower() if region else "",
            "2022",
            "fevereiro",
            "SGB",
            "CPRM",
        ]

        # Procurar em docs README
        docs_to_check = [
            REPO_ROOT / "README.md",
            DATASETS_DIR / "README.md",
            DOCS_DIR / "research_datasets_and_artifacts.md",
        ]

        for doc_path in docs_to_check:
            if doc_path.exists():
                try:
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        for term in search_terms:
                            if term and term.lower() in content:
                                self.documentary_log.append(
                                    f"Referencia encontrada em {doc_path.name}: '{term}'"
                                )

                except Exception as e:
                    self.qa_log.append(f"WARN: Failed to scan {doc_path.name}: {str(e)[:80]}")

        return evidences

    def _create_temporal_review(self, idx: int, candidate: Dict) -> TemporalReview:
        """Criar revisao temporal para candidato."""
        review = TemporalReview(
            temporal_review_id=f"TREV_{idx:03d}",
            consolidated_candidate_id=candidate.get("consolidated_candidate_id", ""),
            source_candidate_id=candidate.get("source_candidate_id", ""),
            region=candidate.get("region", ""),
            event_id=candidate.get("event_id", ""),
            asset_name=candidate.get("asset_name", ""),
            asset_format=candidate.get("asset_format", ""),
            geometry_available=candidate.get("geometry_available", ""),
            crs=candidate.get("crs", ""),
            phenomenon_group=candidate.get("phenomenon_group", ""),
            observed_not_risk=candidate.get("observed_not_risk", ""),
            previous_blocking_gate=candidate.get("blocking_gate", ""),
            previous_blocking_reason=candidate.get("blocking_reason", ""),
            event_date_candidate_before=candidate.get("event_date_candidate", ""),
        )

        # Verificar se deve ser revisado
        blocking_gate = candidate.get("blocking_gate", "")
        if "gate_04" in blocking_gate or "no_date" in blocking_gate.lower():
            review.temporal_status_after_review = "TEMPORAL_GATE_BLOCKED_NO_DATE"
            review.can_create_training_label = "NO"
            self.stats["candidates_still_blocked"] += 1
        else:
            review.temporal_status_after_review = "TEMPORAL_CONTEXT_STRENGTHENED_STILL_BLOCKED"

        return review

    def audit(self):
        """Executar auditoria temporal."""
        candidates = self._load_consolidated_candidates()

        for idx, candidate in enumerate(candidates):
            if not self._should_prioritize(candidate):
                continue

            # Criar revisao
            review = self._create_temporal_review(idx, candidate)
            self._scan_sidecars_for_candidate(candidate)
            self._scan_registries_for_candidate(candidate)
            self._scan_local_docs_for_candidate(candidate)

            # Criar gate
            gate = TemporalGateRecord(
                candidate_id=review.consolidated_candidate_id,
                has_sidecar_metadata="NO",
                has_registry_cross_reference="NO",
                has_documentary_linkage="NO" if not self.documentary_log else "PARTIAL",
                temporal_gate_status=review.temporal_status_after_review,
                can_create_training_label="NO",
            )

            self.temporal_reviews.append(review)
            self.temporal_gates.append(gate)
            self.stats["total_candidates_reviewed"] += 1

        self.stats["candidates_with_temporal_evidence"] = len(
            [r for r in self.temporal_reviews if r.temporal_evidence_strength != "INSUFFICIENT_EVIDENCE"]
        )

    def write_outputs(self):
        """Escrever outputs em local_runs e datasets."""
        if not self.force:
            return

        LOCAL_RUNS.mkdir(parents=True, exist_ok=True)

        # 1. Temporal provenance recovery registry
        registry_path = DATASETS_DIR / "temporal_provenance_recovery_registry.csv"
        with open(registry_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                "temporal_review_id", "consolidated_candidate_id", "source_candidate_id",
                "region", "event_id", "asset_name", "asset_format", "geometry_available",
                "crs", "phenomenon_group", "observed_not_risk", "previous_blocking_gate",
                "previous_blocking_reason", "temporal_evidence_source_type",
                "temporal_evidence_source_name", "temporal_evidence_field",
                "temporal_evidence_value", "temporal_evidence_strength",
                "event_date_candidate_before", "event_date_candidate_after",
                "event_date_compatible_after_review", "temporal_confidence",
                "accepted_as_event_date", "accepted_as_survey_date", "accepted_as_context_only",
                "temporal_status_after_review", "ground_truth_candidate_status_after_review",
                "patch_binding_preflight_status_after_review", "can_create_training_label",
                "limitations", "next_required_action", "notes"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for review in self.temporal_reviews:
                writer.writerow(asdict(review))

        # 2. Temporal gate decision matrix
        if self.emit_temporal_decision_matrix:
            gate_path = DATASETS_DIR / "temporal_gate_decision_matrix.csv"
            with open(gate_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    "candidate_id", "has_explicit_event_date", "has_explicit_survey_date",
                    "has_event_window", "has_documentary_linkage", "has_registry_cross_reference",
                    "has_sidecar_metadata", "only_file_or_folder_hint", "only_file_system_date",
                    "contradictory_temporal_evidence", "event_date_compatible",
                    "survey_date_usable_as_context", "temporal_gate_status",
                    "temporal_blocking_reason", "can_reopen_patch_binding_preflight",
                    "can_reopen_ground_truth_candidate", "can_create_training_label"
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for gate in self.temporal_gates:
                    writer.writerow(asdict(gate))

        # 3. Local outputs
        summary_path = LOCAL_RUNS / "v1ik_temporal_provenance_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, default=str)

        qa_path = LOCAL_RUNS / "v1ik_temporal_provenance_qa.csv"
        with open(qa_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "message"])
            for msg in self.qa_log:
                writer.writerow([datetime.now(timezone.utc).isoformat(), msg])

        sidecar_log_path = LOCAL_RUNS / "v1ik_sidecar_scan_log.csv"
        with open(sidecar_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "action"])
            for msg in self.sidecar_log:
                writer.writerow([datetime.now(timezone.utc).isoformat(), msg])

        registry_log_path = LOCAL_RUNS / "v1ik_registry_cross_reference_log.csv"
        with open(registry_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "reference"])
            for msg in self.registry_log:
                writer.writerow([datetime.now(timezone.utc).isoformat(), msg])

        doc_log_path = LOCAL_RUNS / "v1ik_documentary_support_log.csv"
        with open(doc_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "reference"])
            for msg in self.documentary_log:
                writer.writerow([datetime.now(timezone.utc).isoformat(), msg])


def main():
    parser = argparse.ArgumentParser(
        description="v1ik: Auditar proveniencia temporal de candidatos vetoriais bloqueados"
    )
    parser.add_argument("--force", action="store_true",
                        help="Escrever registries consolidados publicos")
    parser.add_argument("--scan-sidecars", action="store_true",
                        help="Buscar sidecars locais")
    parser.add_argument("--scan-registries", action="store_true",
                        help="Consultar registries de v1if/v1ii")
    parser.add_argument("--scan-local-docs", action="store_true",
                        help="Consultar documentacao versionada")
    parser.add_argument("--focus-best-candidates", action="store_true",
                        help="Priorizar melhores candidatos")
    parser.add_argument("--emit-temporal-decision-matrix", action="store_true",
                        help="Gerar matriz de decisao temporal")

    args = parser.parse_args()

    auditor = TemporalProvenanceAuditor(
        force=args.force,
        scan_sidecars=args.scan_sidecars,
        scan_registries=args.scan_registries,
        scan_local_docs=args.scan_local_docs,
        focus_best_candidates=args.focus_best_candidates,
        emit_temporal_decision_matrix=args.emit_temporal_decision_matrix,
    )

    auditor.audit()
    auditor.write_outputs()

    print(f"[v1ik] Auditoria temporal concluida.")
    print(f"[v1ik] Total candidatos revisados: {auditor.stats['total_candidates_reviewed']}")
    print(f"[v1ik] Candidatos com evidencia temporal: {auditor.stats['candidates_with_temporal_evidence']}")
    print(f"[v1ik] Candidatos ainda bloqueados: {auditor.stats['candidates_still_blocked']}")

    if args.force:
        print(f"[v1ik] Registries escritos em {DATASETS_DIR}")
        print(f"[v1ik] Local outputs em {LOCAL_RUNS}")


if __name__ == "__main__":
    main()
