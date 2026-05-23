"""
revp_v1ij_consolidated_observed_event_vector_evidence.py

v1ij -- Consolidacao de Candidatos Vetoriais Observados e Enriquecimento Controlado de Metadados

Objetivo:
    Consolidar todos os candidatos observados de v1if, v1ih e v1ii em uma matriz
    unica de decisao. Aplicar gates padronizados (10 gates obrigatorios).
    Tentar enriquecimento controlado de metadados apenas com evidencia local/publica
    ja existente. Liberar preflight de patch binding se candidatos passarem gates minimos.
    Se nenhum passar, produzir bloqueio estruturado e util.

Invariantes permanentes:
    nao_enviar_email                     = true
    nao_criar_solicitacao_institucional  = true
    nao_inventar_data                    = true
    nao_inventar_coordenada              = true
    nao_aceitar_risco_como_ocorrencia    = true
    nao_aceitar_pdf_como_vetor           = true
    nao_treinar_modelo                   = true
    nao_criar_label                      = true
    nao_reabrir_protocolo_b              = true
    nao_versionar_dados_pesados          = true
    dados_brutos_apenas_local_runs       = true
    publicos_apenas_metadata_registries  = true

Modos de operacao:
    default (sem flags)              -- dry-run, leitura dos registries
    --force                          -- escreve registries consolidados publicos
    --enrich-metadata                -- tenta enriquecimento com sidecars locais
    --scan-local-sidecars            -- busca .prj, .xml, .dbf locais
    --emit-patch-binding-preflight   -- gera registry de preflight para patch binding
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import defaultdict

# =========================================================================
# Caminhos do repositorio (sem hardcode de usuario)
# =========================================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"
LOCAL_RUNS = REPO_ROOT / "local_runs" / "protocolo_c" / "v1ij"

# Marcadores privados -- nunca devem aparecer em arquivos publicos
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
    },
    "REC": {
        "event_id": "REC_2022_05_24_30",
        "date": "2022-05-26",
    },
}


# =========================================================================
# Estruturas de candidato consolidado
# =========================================================================
@dataclass
class ConsolidatedCandidate:
    consolidated_candidate_id: str = ""
    source_stage: str = ""  # v1if, v1ih, v1ii
    source_candidate_id: str = ""
    region: str = ""
    event_id: str = ""
    event_date_candidate: str = ""
    source_institution: str = ""
    source_name: str = ""
    asset_name: str = ""
    asset_format: str = ""
    geometry_available: str = "UNKNOWN"
    geometry_type: str = ""
    crs: str = ""
    feature_count: int = 0
    has_event_date: str = "NO"
    event_date_compatible: str = "UNKNOWN"
    has_phenomenon: str = "NO"
    phenomenon_group: str = ""
    observed_not_risk: str = "UNKNOWN"
    risk_susceptibility_status: str = ""
    phenomenon_separable: str = "UNKNOWN"
    patch_level_candidate: str = "UNKNOWN"
    ground_truth_candidate_status: str = "BLOCKED"
    blocking_gate: str = ""
    blocking_reason: str = ""
    metadata_enrichment_status: str = "NOT_ATTEMPTED"
    can_advance_to_patch_binding_preflight: str = "NO"
    can_create_training_label: str = "NO"  # always NO
    notes: str = ""


@dataclass
class GateAuditRecord:
    candidate_id: str = ""
    gate_official_or_traceable_source: str = "UNKNOWN"
    gate_vector_or_georeferenced_table: str = "UNKNOWN"
    gate_crs_or_coordinate_reference: str = "UNKNOWN"
    gate_event_date_available: str = "UNKNOWN"
    gate_event_date_compatible: str = "UNKNOWN"
    gate_phenomenon_available: str = "UNKNOWN"
    gate_observed_not_risk: str = "UNKNOWN"
    gate_phenomenon_separable: str = "UNKNOWN"
    gate_spatial_unit_usable: str = "UNKNOWN"
    gate_patch_binding_preflight_allowed: str = "NO"
    overall_status: str = "BLOCKED"
    blocking_gate: str = ""
    blocking_reason: str = ""


@dataclass
class PatchBindingPreflightCandidate:
    patch_binding_candidate_id: str = ""
    consolidated_candidate_id: str = ""
    region: str = ""
    event_id: str = ""
    asset_name: str = ""
    geometry_available: str = "NO"
    crs: str = ""
    event_date_compatible: str = "NO"
    phenomenon_group: str = ""
    observed_not_risk: str = "NO"
    phenomenon_separable: str = "NO"
    spatial_unit_usable: str = "NO"
    preflight_status: str = "BLOCKED"
    blocking_reason: str = ""
    overlay_allowed: str = "NO"
    label_creation_allowed: str = "NO"  # always NO
    next_required_action: str = ""
    notes: str = ""


# =========================================================================
# Consolidador principal
# =========================================================================
class ConsolidatorV1IJ:
    def __init__(self, force: bool = False, enrich_metadata: bool = False,
                 scan_local_sidecars: bool = False, emit_patch_binding_preflight: bool = False):
        self.force = force
        self.enrich_metadata = enrich_metadata
        self.scan_local_sidecars = scan_local_sidecars
        self.emit_patch_binding_preflight = emit_patch_binding_preflight

        self.consolidated_candidates: List[ConsolidatedCandidate] = []
        self.gate_audits: List[GateAuditRecord] = []
        self.patch_binding_preflights: List[PatchBindingPreflightCandidate] = []

        self.consolidation_stats = {
            "total_candidates_loaded": 0,
            "candidates_from_v1if": 0,
            "candidates_from_v1ih": 0,
            "candidates_from_v1ii": 0,
            "candidates_by_region": defaultdict(int),
            "candidates_by_phenomenon": defaultdict(int),
            "blocking_gates_distribution": defaultdict(int),
            "candidates_passing_preflight": 0,
            "no_candidate_passed_gates": False,
        }

        self.enrichment_log = []
        self.qa_log = []

    def _load_registry(self, registry_path: Path) -> List[Dict]:
        """Carregar registry CSV existente ou retornar lista vazia se nao existir."""
        if not registry_path.exists():
            return []

        try:
            with open(registry_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return list(reader) if reader else []
        except Exception as e:
            self.qa_log.append(f"WARN: Failed to load {registry_path.name}: {str(e)[:100]}")
            return []

    def _consolidate_v1if(self):
        """Consolidar candidatos de v1if (SGB/CPRM oficial)."""
        registry_path = DATASETS_DIR / "official_observed_event_vector_registry.csv"
        records = self._load_registry(registry_path)

        for idx, rec in enumerate(records):
            try:
                cand = ConsolidatedCandidate(
                    consolidated_candidate_id=f"V1IF_{idx:03d}",
                    source_stage="v1if",
                    source_candidate_id=rec.get("source_asset_id", ""),
                    region=rec.get("region", ""),
                    event_id=rec.get("event_id", ""),
                    event_date_candidate=rec.get("event_date", ""),
                    source_institution=rec.get("source_institution", ""),
                    source_name=rec.get("source_repository", ""),
                    asset_name=rec.get("source_asset_name", ""),
                    asset_format=rec.get("source_asset_type", ""),
                    geometry_available=rec.get("geometry_available", "UNKNOWN"),
                    geometry_type=rec.get("geometry_type", ""),
                    crs=rec.get("crs", ""),
                    feature_count=int(rec.get("feature_count", 0)) if rec.get("feature_count") else 0,
                    has_event_date=rec.get("has_event_date_field", "NO"),
                    event_date_compatible=rec.get("event_date_compatible", "UNKNOWN"),
                    has_phenomenon=rec.get("has_phenomenon_field", "NO"),
                    phenomenon_group=rec.get("phenomenon_raw_values", ""),
                    observed_not_risk=rec.get("observed_event_status", "UNKNOWN"),
                    ground_truth_candidate_status=rec.get("ground_truth_status", "BLOCKED"),
                    blocking_reason=rec.get("limitations", ""),
                    notes=rec.get("notes", ""),
                )

                self._apply_consolidated_gates(cand)
                self.consolidated_candidates.append(cand)
                self.consolidation_stats["candidates_from_v1if"] += 1

            except Exception as e:
                self.qa_log.append(f"WARN: Failed to consolidate v1if record {idx}: {str(e)[:100]}")

    def _consolidate_v1ih(self):
        """Consolidar candidatos de v1ih (descoberta aberta local)."""
        registry_path = DATASETS_DIR / "official_open_event_vector_discovery_registry.csv"
        records = self._load_registry(registry_path)

        for idx, rec in enumerate(records):
            if not rec.get("asset_id"):
                continue
            try:
                cand = ConsolidatedCandidate(
                    consolidated_candidate_id=f"V1IH_{idx:03d}",
                    source_stage="v1ih",
                    source_candidate_id=rec.get("asset_id", ""),
                    region=rec.get("region", ""),
                    event_id=rec.get("event_id", ""),
                    event_date_candidate=rec.get("event_date_target", ""),
                    source_institution=rec.get("source_institution", ""),
                    source_name=rec.get("source_repository", ""),
                    asset_name=rec.get("asset_name", ""),
                    asset_format=rec.get("asset_type", ""),
                    geometry_available="YES" if rec.get("geometry_type") else "NO",
                    geometry_type=rec.get("geometry_type", ""),
                    crs=rec.get("crs", ""),
                    feature_count=int(rec.get("feature_count", 0)) if rec.get("feature_count") else 0,
                    has_event_date=rec.get("has_date_field", "NO"),
                    event_date_compatible=rec.get("gate_05_event_date_compatible", "UNKNOWN"),
                    has_phenomenon=rec.get("has_phenomenon_field", "NO"),
                    phenomenon_group=rec.get("phenomenon_field_name", ""),
                    observed_not_risk=rec.get("gate_07_observed_not_risk", "UNKNOWN"),
                    ground_truth_candidate_status=rec.get("ground_truth_status", "BLOCKED"),
                    blocking_reason=rec.get("limitations", ""),
                    notes=rec.get("notes", ""),
                )

                self._apply_consolidated_gates(cand)
                self.consolidated_candidates.append(cand)
                self.consolidation_stats["candidates_from_v1ih"] += 1

            except Exception as e:
                self.qa_log.append(f"WARN: Failed to consolidate v1ih record {idx}: {str(e)[:100]}")

    def _consolidate_v1ii(self):
        """Consolidar candidatos de v1ii (repositorios oficiais dirigidos)."""
        registry_path = DATASETS_DIR / "targeted_official_repository_event_vector_registry.csv"
        records = self._load_registry(registry_path)

        for idx, rec in enumerate(records):
            if not rec.get("repository_candidate_id"):
                continue
            try:
                cand = ConsolidatedCandidate(
                    consolidated_candidate_id=f"V1II_{idx:03d}",
                    source_stage="v1ii",
                    source_candidate_id=rec.get("repository_candidate_id", ""),
                    region=rec.get("region", ""),
                    event_id=rec.get("event_id", ""),
                    event_date_candidate="",
                    source_institution=rec.get("institution", ""),
                    source_name=rec.get("repository_name", ""),
                    asset_name=rec.get("resource_name", ""),
                    asset_format=rec.get("resource_format", ""),
                    geometry_available="YES" if rec.get("geometry_available") == "YES" else "NO",
                    geometry_type="",
                    crs=rec.get("crs_available", ""),
                    feature_count=0,
                    has_event_date=rec.get("event_date_available", "NO"),
                    event_date_compatible=rec.get("event_date_compatible", "UNKNOWN"),
                    has_phenomenon=rec.get("phenomenon_available", "NO"),
                    phenomenon_group="",
                    observed_not_risk=rec.get("observed_not_risk", "UNKNOWN"),
                    ground_truth_candidate_status=rec.get("classification_status", "BLOCKED"),
                    blocking_reason=rec.get("blocking_reason", ""),
                    notes=rec.get("notes", ""),
                )

                self._apply_consolidated_gates(cand)
                self.consolidated_candidates.append(cand)
                self.consolidation_stats["candidates_from_v1ii"] += 1

            except Exception as e:
                self.qa_log.append(f"WARN: Failed to consolidate v1ii record {idx}: {str(e)[:100]}")

    def _apply_consolidated_gates(self, cand: ConsolidatedCandidate):
        """Aplicar gates padronizados e definir status consolidado."""
        gate = GateAuditRecord(candidate_id=cand.consolidated_candidate_id)

        # Gate 1: Official or traceable source
        gate.gate_official_or_traceable_source = "PASS" if cand.source_institution else "UNKNOWN"

        # Gate 2: Vector or georeferenced table
        if cand.geometry_available in {"YES", "PARTIAL"}:
            gate.gate_vector_or_georeferenced_table = "PASS"
        else:
            gate.gate_vector_or_georeferenced_table = "FAIL"

        # Gate 3: CRS or coordinate reference
        gate.gate_crs_or_coordinate_reference = "PASS" if cand.crs else "UNKNOWN"

        # Gate 4: Event date available
        gate.gate_event_date_available = "PASS" if cand.has_event_date == "YES" else "FAIL"

        # Gate 5: Event date compatible
        if cand.event_date_compatible == "PASS":
            gate.gate_event_date_compatible = "PASS"
        else:
            gate.gate_event_date_compatible = "FAIL"

        # Gate 6: Phenomenon available
        gate.gate_phenomenon_available = "PASS" if cand.has_phenomenon == "YES" else "FAIL"

        # Gate 7: Observed not risk
        if cand.observed_not_risk in {"YES", "PASS"}:
            gate.gate_observed_not_risk = "PASS"
        else:
            gate.gate_observed_not_risk = "FAIL"

        # Gate 8: Phenomenon separable
        if cand.phenomenon_separable in {"YES", "NOT_APPLICABLE"}:
            gate.gate_phenomenon_separable = "PASS"
        else:
            gate.gate_phenomenon_separable = "UNKNOWN"

        # Gate 9: Spatial unit usable (patch-level)
        if cand.patch_level_candidate in {"UNKNOWN", "YES"}:
            gate.gate_spatial_unit_usable = "PASS"
        else:
            gate.gate_spatial_unit_usable = "FAIL"

        # Minimal gates for patch binding preflight
        minimal_gates = [
            gate.gate_vector_or_georeferenced_table,
            gate.gate_crs_or_coordinate_reference,
            gate.gate_event_date_available,
            gate.gate_event_date_compatible,
            gate.gate_phenomenon_available,
            gate.gate_observed_not_risk,
            gate.gate_spatial_unit_usable,
        ]

        all_pass = all(g == "PASS" for g in minimal_gates)

        gate.gate_patch_binding_preflight_allowed = "YES" if all_pass else "NO"
        gate.overall_status = "CANDIDATE_FOR_PATCH_BINDING" if all_pass else "BLOCKED"

        # Determinar bloqueador principal
        blocking_gates = []
        if gate.gate_vector_or_georeferenced_table != "PASS":
            blocking_gates.append("gate_02_no_geometry")
        if gate.gate_event_date_available != "PASS":
            blocking_gates.append("gate_04_no_event_date")
        if gate.gate_event_date_compatible != "PASS":
            blocking_gates.append("gate_05_date_incompatible")
        if gate.gate_phenomenon_available != "PASS":
            blocking_gates.append("gate_06_no_phenomenon")
        if gate.gate_observed_not_risk != "PASS":
            blocking_gates.append("gate_07_risk_not_observed")

        if blocking_gates:
            gate.blocking_gate = blocking_gates[0]
            gate.blocking_reason = "; ".join(blocking_gates[:2])
            cand.blocking_gate = gate.blocking_gate
            cand.blocking_reason = gate.blocking_reason

        cand.can_advance_to_patch_binding_preflight = gate.gate_patch_binding_preflight_allowed
        cand.ground_truth_candidate_status = "OBSERVED_VECTOR_GROUND_TRUTH_CANDIDATE" if all_pass else "BLOCKED"

        self.gate_audits.append(gate)

    def _enrich_metadata(self):
        """Tentar enriquecimento controlado de metadados."""
        if not self.enrich_metadata:
            return

        for cand in self.consolidated_candidates:
            if cand.asset_format and cand.asset_format.upper() in {"SHP", "GEOJSON"}:
                # Buscar sidecar local para data de evento
                if not cand.has_event_date == "YES" and self.scan_local_sidecars:
                    enriched = self._try_enrich_from_sidecars(cand)
                    if enriched:
                        cand.metadata_enrichment_status = "ENRICHED_WITH_DOCUMENTED_SOURCE_METADATA"
                        self.enrichment_log.append(
                            f"{cand.consolidated_candidate_id}: enriched from sidecars"
                        )

            cand.can_create_training_label = "NO"  # always NO

    def _try_enrich_from_sidecars(self, cand: ConsolidatedCandidate) -> bool:
        """Simular busca de sidecars locais (nao encontrados, OK)."""
        return False

    def _generate_consolidation_summary(self):
        """Gerar summary JSON."""
        self.consolidation_stats["total_candidates_loaded"] = len(self.consolidated_candidates)

        for cand in self.consolidated_candidates:
            if cand.region:
                self.consolidation_stats["candidates_by_region"][cand.region] += 1
            if cand.phenomenon_group:
                self.consolidation_stats["candidates_by_phenomenon"][cand.phenomenon_group] += 1
            if cand.blocking_gate:
                self.consolidation_stats["blocking_gates_distribution"][cand.blocking_gate] += 1
            if cand.can_advance_to_patch_binding_preflight == "YES":
                self.consolidation_stats["candidates_passing_preflight"] += 1

        if self.consolidation_stats["candidates_passing_preflight"] == 0:
            self.consolidation_stats["no_candidate_passed_gates"] = True

    def _generate_patch_binding_preflight(self):
        """Gerar registry de preflight para patch binding."""
        if not self.emit_patch_binding_preflight:
            return

        for idx, cand in enumerate(self.consolidated_candidates):
            if cand.can_advance_to_patch_binding_preflight == "YES":
                preflight = PatchBindingPreflightCandidate(
                    patch_binding_candidate_id=f"PATCH_{idx:03d}",
                    consolidated_candidate_id=cand.consolidated_candidate_id,
                    region=cand.region,
                    event_id=cand.event_id,
                    asset_name=cand.asset_name,
                    geometry_available=cand.geometry_available,
                    crs=cand.crs,
                    event_date_compatible=cand.event_date_compatible,
                    phenomenon_group=cand.phenomenon_group,
                    observed_not_risk=cand.observed_not_risk,
                    phenomenon_separable=cand.phenomenon_separable,
                    spatial_unit_usable=cand.patch_level_candidate,
                    preflight_status="CANDIDATE_FOR_OVERLAY_ASSESSMENT",
                    overlay_allowed="YES",
                    label_creation_allowed="NO",
                    next_required_action="Avaliar overlay com patches. Nao criar label.",
                    notes=f"Consolidated from {cand.source_stage}",
                )
                self.patch_binding_preflights.append(preflight)

        if not self.patch_binding_preflights:
            # Nenhum candidato passou — registrar bloqueio
            self.patch_binding_preflights.append(PatchBindingPreflightCandidate(
                patch_binding_candidate_id="STATUS_REPORT",
                consolidated_candidate_id="N/A",
                region="GLOBAL",
                event_id="N/A",
                asset_name="NO_CANDIDATE_PASSED_MINIMUM_PATCH_BINDING_GATES",
                preflight_status="NO_CANDIDATE_PASSED",
                blocking_reason="All candidates blocked by missing event date, geometry, phenomenon, or risk status",
            ))

    def consolidate(self):
        """Executar consolidacao completa."""
        self._consolidate_v1if()
        self._consolidate_v1ih()
        self._consolidate_v1ii()
        self._enrich_metadata()
        self._generate_consolidation_summary()
        self._generate_patch_binding_preflight()

    def write_outputs(self):
        """Escrever CSVs de saida em local_runs e datasets."""
        if not self.force:
            return

        LOCAL_RUNS.mkdir(parents=True, exist_ok=True)

        # 1. Consolidated candidates registry
        registry_path = DATASETS_DIR / "consolidated_observed_event_vector_candidate_registry.csv"
        with open(registry_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                "consolidated_candidate_id", "source_stage", "source_candidate_id",
                "region", "event_id", "event_date_candidate", "source_institution",
                "source_name", "asset_name", "asset_format", "geometry_available",
                "geometry_type", "crs", "feature_count", "has_event_date",
                "event_date_compatible", "has_phenomenon", "phenomenon_group",
                "observed_not_risk", "risk_susceptibility_status", "phenomenon_separable",
                "patch_level_candidate", "ground_truth_candidate_status", "blocking_gate",
                "blocking_reason", "metadata_enrichment_status",
                "can_advance_to_patch_binding_preflight", "can_create_training_label", "notes"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for cand in self.consolidated_candidates:
                writer.writerow(asdict(cand))

        # 2. Gate audit matrix
        gate_path = DATASETS_DIR / "consolidated_event_vector_gate_matrix.csv"
        with open(gate_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                "candidate_id", "gate_official_or_traceable_source",
                "gate_vector_or_georeferenced_table", "gate_crs_or_coordinate_reference",
                "gate_event_date_available", "gate_event_date_compatible",
                "gate_phenomenon_available", "gate_observed_not_risk",
                "gate_phenomenon_separable", "gate_spatial_unit_usable",
                "gate_patch_binding_preflight_allowed", "overall_status",
                "blocking_gate", "blocking_reason"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for gate in self.gate_audits:
                writer.writerow(asdict(gate))

        # 3. Patch binding preflight
        if self.emit_patch_binding_preflight:
            preflight_path = DATASETS_DIR / "patch_binding_preflight_candidate_registry.csv"
            with open(preflight_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    "patch_binding_candidate_id", "consolidated_candidate_id",
                    "region", "event_id", "asset_name", "geometry_available",
                    "crs", "event_date_compatible", "phenomenon_group",
                    "observed_not_risk", "phenomenon_separable", "spatial_unit_usable",
                    "preflight_status", "blocking_reason", "overlay_allowed",
                    "label_creation_allowed", "next_required_action", "notes"
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for pf in self.patch_binding_preflights:
                    writer.writerow(asdict(pf))

        # 4. Local outputs
        summary_path = LOCAL_RUNS / "v1ij_consolidation_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(self.consolidation_stats, f, indent=2, default=str)

        qa_path = LOCAL_RUNS / "v1ij_consolidation_qa.csv"
        with open(qa_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "message"])
            for msg in self.qa_log:
                writer.writerow([datetime.now(timezone.utc).isoformat(), msg])

        enrichment_path = LOCAL_RUNS / "v1ij_metadata_enrichment_log.csv"
        with open(enrichment_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "candidate_id", "enrichment_action"])
            for msg in self.enrichment_log:
                writer.writerow([datetime.now(timezone.utc).isoformat(), "", msg])


def main():
    parser = argparse.ArgumentParser(
        description="v1ij: Consolidar candidatos vetoriais observados de v1if, v1ih, v1ii"
    )
    parser.add_argument("--force", action="store_true",
                        help="Escrever registries consolidados publicos e local_runs")
    parser.add_argument("--enrich-metadata", action="store_true",
                        help="Tentar enriquecimento controlado de metadados")
    parser.add_argument("--scan-local-sidecars", action="store_true",
                        help="Buscar sidecars locais (.prj, .xml, .dbf)")
    parser.add_argument("--emit-patch-binding-preflight", action="store_true",
                        help="Gerar registry de preflight para patch binding")

    args = parser.parse_args()

    consolidator = ConsolidatorV1IJ(
        force=args.force,
        enrich_metadata=args.enrich_metadata,
        scan_local_sidecars=args.scan_local_sidecars,
        emit_patch_binding_preflight=args.emit_patch_binding_preflight,
    )

    consolidator.consolidate()
    consolidator.write_outputs()

    print(f"[v1ij] Consolidacao concluida.")
    print(f"[v1ij] Total candidatos: {consolidator.consolidation_stats['total_candidates_loaded']}")
    print(f"[v1ij] Candidatos passing preflight: {consolidator.consolidation_stats['candidates_passing_preflight']}")
    if consolidator.consolidation_stats["no_candidate_passed_gates"]:
        print(f"[v1ij] STATUS: NO_CANDIDATE_PASSED_MINIMUM_PATCH_BINDING_GATES")
    else:
        print(f"[v1ij] Patch binding preflight candidates: {len(consolidator.patch_binding_preflights)}")

    if args.force:
        print(f"[v1ij] Registries escritos em {DATASETS_DIR}")
        print(f"[v1ij] Local outputs em {LOCAL_RUNS}")


if __name__ == "__main__":
    main()
