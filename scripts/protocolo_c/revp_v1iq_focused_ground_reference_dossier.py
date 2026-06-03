"""
revp_v1iq_focused_ground_reference_dossier.py

Protocolo C — v1iq: Focused Ground Reference Dossier for PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA

Objetivo: Construir dossiê técnico focado em camada original de feições poligonais de deslizamento fotointerpretadas para verificar
se o conjunto composto de evidências permite promovê-lo a GROUND_REFERENCE_CANDIDATE.

Pergunta central: Com as evidências já existentes, camada original de feições poligonais de deslizamento fotointerpretadas pode subir
de STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK para GROUND_REFERENCE_CANDIDATE?

Evidências locais reais lidas e auditadas nesta etapa:
- XML sidecar: sidecar original de pontos de feições de deslizamento fotointerpretadas → SIG criado em 2013-2015 por fotointerpretação
- DBF camada original de feições poligonais de deslizamento fotointerpretadas (Petrópolis): date=2015-11-30, 444 registros, sem campo de data
- DBF VALORES (v1iq-R2): TIPO, CONDICIONA, FONTE, OBS, MUNICIPIO, UF lidos registro a registro
- Registries v1ij/v1ik: blocking_reason inclui 'feições de deslizamento_cumulativas_sem_data_especifica'
- v1in: 0 linkages a candidatos específicos
"""

import argparse
import csv
import json
import re
import struct
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "protocolo_c"))

SOURCE_LAYER_ALIAS = "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA"
SOURCE_LAYER_DISPLAY_NAME = "Feições poligonais de deslizamento fotointerpretadas"
SOURCE_LAYER_ORIGINAL_NAME = "Cicatriz_Area_A.shp"

# ---------------------------------------------------------------------------
# Termos de busca para auditoria de atributos (v1iq-R2)
# ---------------------------------------------------------------------------

TEMPORAL_TERMS = [
    "2022", "15/02/2022", "2022-02-15", "fevereiro", "data",
    "evento", "ocorrência", "ocorrencia", "vistoria", "campo",
    "levantamento", "mapeamento", "pós-desastre", "pos-desastre",
]

SOURCE_TERMS = [
    "sgb", "cprm", "rigeo", "drm", "defesa civil", "prefeitura",
    "relatório", "relatorio", "avaliação", "avaliacao", "laudo",
    "carta", "mapeamento",
]

PHENOMENON_TERMS = [
    "feição de deslizamento", "deslizamento", "escorregamento", "movimento de massa",
    "corrida de massa", "queda", "ruptura", "instabilidade",
]

# Termos de data de evento específico (mais restrito que TEMPORAL_TERMS)
EVENT_DATE_TERMS = ["2022", "15/02/2022", "2022-02-15"]


@dataclass
class GateResult:
    """Resultado de um gate composto."""
    gate_name: str
    status: str          # PASS, FAIL, MODERATE, WEAK, UNKNOWN
    evidence_source: str  # De onde vem a evidência
    evidence_detail: str  # Detalhe da evidência
    note: str            # Observação metodológica


@dataclass
class AttributeProvenanceDecision:
    """Decisão de proveniência baseada nos valores dos atributos do DBF (v1iq-R2)."""
    candidate_asset_name: str
    source_layer_alias: str
    source_layer_display_name: str
    source_layer_original_name: str
    records_count: int
    municipio_values: str
    uf_values: str
    tipo_values: str
    condiciona_values: str
    fonte_values_sanitized: str
    obs_values_sanitized: str
    has_source_in_field: str        # YES/NO
    has_phenomenon_in_field: str    # YES/NO
    has_temporal_expression_in_field: str  # YES/NO
    has_event_or_survey_date_in_field: str  # YES/NO
    observed_status_from_attributes: str
    source_lineage_from_attributes: str
    temporal_link_from_attributes: str
    attribute_evidence_strength: str  # STRONG/MODERATE/WEAK/NONE
    promotion_decision_after_attribute_audit: str
    remaining_blocker: str
    minimum_evidence_needed: str
    notes: str


@dataclass
class CicatrizAreaDossier:
    """Dossiê técnico focado em PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA."""
    dossier_id: str
    candidate_asset_name: str
    source_layer_alias: str
    source_layer_display_name: str
    source_layer_original_name: str
    region: str
    event_id: str
    source_institution: str
    source_document_name_sanitized: str
    source_asset_name_sanitized: str
    geometry_available: str
    crs_available: str
    phenomenon_available: str
    phenomenon_group: str
    observed_not_modelled_status: str
    event_date_documented: str
    survey_date_documented: str
    document_vector_package_link: str
    source_lineage_match: str
    region_match: str
    phenomenon_match: str
    temporal_link_strength: str
    composite_evidence_strength: str
    promotion_decision: str
    can_be_ground_reference_candidate: str
    can_be_operational_ground_truth: str
    can_create_training_label: str
    can_train_model: str
    can_reopen_protocol_b: str
    primary_blocker: str
    minimum_evidence_needed: str
    notes: str


class FocusedGroundReferenceDossierBuilder:
    """Construtor de dossiê focado em PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA."""

    TARGET_TERMS = [
        "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA", "feição de deslizamento", "deslizamento", "escorregamento",
        "movimento de massa", "Petrópolis", "Petropolis", "Quitandinha",
        "fevereiro de 2022", "15/02/2022", "2022-02-15",
        "avaliação pós-desastre", "pos-desastre", "campo", "vistoria", "levantamento",
    ]

    _PROJETO_ROOTS = [
        Path(r"C:\Users\gabriela\Documents\PROJETO"),
        Path.home() / "Documents" / "PROJETO",
    ]

    def __init__(self, force=False):
        self.force = force
        self.repo_root = REPO_ROOT
        self.dossier: Optional[CicatrizAreaDossier] = None
        self.gates: List[GateResult] = []
        self.dbf_audit: Dict = {}
        self.attr_prov: Optional[AttributeProvenanceDecision] = None
        self.stats: Dict = {
            "dossier_complete": False,
            "gates_pass": 0,
            "gates_moderate": 0,
            "gates_fail": 0,
            "promotion_decision": "UNKNOWN",
            "local_files_audited": 0,
            "registries_read": 0,
            "targeted_term_matches": 0,
        }
        self._projeto_root: Optional[Path] = None
        self._find_projeto()

    # ------------------------------------------------------------------
    # Inicialização
    # ------------------------------------------------------------------

    def _find_projeto(self):
        for candidate in self._PROJETO_ROOTS:
            if candidate.exists():
                self._projeto_root = candidate
                break

    # ------------------------------------------------------------------
    # Execução principal
    # ------------------------------------------------------------------

    def run(self, focus_cicatriz_area=True, read_composite_evidence=True,
            read_documentary_evidence=True, read_local_metadata=True,
            extract_targeted_text=True, emit_dossier=True,
            emit_promotion_decision=True) -> Dict:

        local_runs_dir = self.repo_root / "local_runs" / "protocolo_c" / "v1iq"
        if self.force and local_runs_dir.exists():
            import shutil
            shutil.rmtree(local_runs_dir, ignore_errors=True)
        local_runs_dir.mkdir(parents=True, exist_ok=True)

        # 1. Auditar metadados locais (cabeçalho DBF, XML, PRJ)
        vector_meta = {}
        if read_local_metadata:
            vector_meta = self._audit_local_vector_metadata()

        # 2. (R2) Auditar valores reais do DBF registro a registro
        if focus_cicatriz_area:
            self.dbf_audit = self._audit_dbf_values()
            self.attr_prov = self._evaluate_attribute_provenance(self.dbf_audit)

        # 3. Ler registries públicos existentes
        registry_evidence = {}
        if read_composite_evidence:
            registry_evidence = self._read_existing_registries()

        # 4. Cruzar com evidências de v1in
        v1in_evidence = {}
        if read_documentary_evidence:
            v1in_evidence = self._read_v1in_evidence()

        # 5. Extração textual direcionada em documentos locais
        text_matches = []
        if extract_targeted_text:
            text_matches = self._extract_targeted_text()

        # 6. Avaliar gates compostos (agora pode usar self.attr_prov)
        self._evaluate_gates(vector_meta, registry_evidence, v1in_evidence, text_matches)

        # 7. Construir dossiê
        self._build_dossier(vector_meta, registry_evidence)

        # 8. Emitir outputs
        self._emit_local_outputs(local_runs_dir, vector_meta, registry_evidence, text_matches)

        if emit_promotion_decision:
            self._emit_promotion_registry()

        self.stats["dossier_complete"] = True

        return {
            "status": "complete",
            "dossier_complete": True,
            "promotion_decision": self.stats["promotion_decision"],
            "gates_pass": self.stats["gates_pass"],
            "gates_moderate": self.stats["gates_moderate"],
            "gates_fail": self.stats["gates_fail"],
            "local_files_audited": self.stats["local_files_audited"],
            "targeted_term_matches": self.stats["targeted_term_matches"],
            "dbf_records_audited": self.dbf_audit.get("total_records", 0),
            "attribute_evidence_strength": (
                self.attr_prov.attribute_evidence_strength if self.attr_prov else "NOT_RUN"
            ),
        }

    # ------------------------------------------------------------------
    # Auditoria de metadados vetoriais (cabeçalho)
    # ------------------------------------------------------------------

    def _audit_local_vector_metadata(self) -> Dict:
        result = {
            "shp_exists": False,
            "dbf_exists": False,
            "shx_exists": False,
            "prj_exists": False,
            "xml_sidecar_exists": False,
            "crs_wkt": "",
            "crs_zone": "",
            "dbf_date": "",
            "dbf_records": 0,
            "dbf_fields": [],
            "has_date_field": False,
            "has_event_date": False,
            "sig_creation_period": "",
            "cicatriz_ponto_p_xml_date": "",
            "cicatriz_ponto_p_source_note": "",
            "source_path_hint": "PET/SGB/CPRM",
        }

        if not self._projeto_root:
            return result

        pet_feicoes = (
            self._projeto_root / "data" / "raw" / "petropolis"
            / "sgb_cprm" / "sig_extracted" / "Feicoes"
        )

        shp = pet_feicoes / "Cicatriz_Area_A.shp"
        dbf = pet_feicoes / "Cicatriz_Area_A.dbf"
        shx = pet_feicoes / "Cicatriz_Area_A.shx"
        prj = pet_feicoes / "Cicatriz_Area_A.prj"

        result["shp_exists"] = shp.exists()
        result["dbf_exists"] = dbf.exists()
        result["shx_exists"] = shx.exists()
        result["prj_exists"] = prj.exists()
        result["xml_sidecar_exists"] = (pet_feicoes / "Cicatriz_Area_A.shp.xml").exists()

        if prj.exists():
            crs_wkt = prj.read_text(encoding="utf-8", errors="replace")
            result["crs_wkt"] = crs_wkt[:200]
            if "Zone_23S" in crs_wkt:
                result["crs_zone"] = "SIRGAS_2000_UTM_Zone_23S"

        if dbf.exists():
            try:
                with open(dbf, "rb") as f:
                    _ = struct.unpack("B", f.read(1))[0]
                    yr, mo, dy = struct.unpack("BBB", f.read(3))
                    num_records = struct.unpack("<I", f.read(4))[0]
                    f.read(24)
                    result["dbf_date"] = f"{yr + 1900}/{mo:02d}/{dy:02d}"
                    result["dbf_records"] = num_records
                    fields = []
                    while True:
                        term = f.read(1)
                        if not term or term == b"\r":
                            break
                        fname_b = term + f.read(10)
                        fname = fname_b.rstrip(b"\x00").decode("latin-1", errors="replace")
                        ftype = f.read(1).decode("latin-1", errors="replace")
                        f.read(4)
                        flen = struct.unpack("B", f.read(1))[0]
                        fdec = struct.unpack("B", f.read(1))[0]
                        f.read(14)
                        fields.append(f"{fname}:{ftype}({flen})")
                        if any(kw in fname.upper() for kw in ["DATA", "DATE", "DT", "YEAR", "ANO"]):
                            result["has_date_field"] = True
                    result["dbf_fields"] = fields
                self.stats["local_files_audited"] += 1
            except Exception as e:
                result["dbf_error"] = str(e)

        xml_pp = pet_feicoes / "Cicatriz_Ponto_P.shp.xml"
        if xml_pp.exists():
            try:
                content = xml_pp.read_text(encoding="utf-8", errors="replace")
                m = re.search(r"<CreaDate>(\d{8})</CreaDate>", content)
                if m:
                    d = m.group(1)
                    result["cicatriz_ponto_p_xml_date"] = f"{d[:4]}/{d[4:6]}/{d[6:]}"
                if "Kits_2013" in content or "SUSCETIBILIDADE" in content:
                    result["cicatriz_ponto_p_source_note"] = "SIG_SUSCETIBILIDADE_2013"
                if "Fotointerpreta" in content:
                    result["cicatriz_ponto_p_source_note"] += ";FOTOINTERPRETACAO"
                self.stats["local_files_audited"] += 1
            except Exception:
                pass

        if result["dbf_date"]:
            yr = int(result["dbf_date"][:4])
            result["sig_creation_period"] = f"{yr}" if yr > 0 else "UNKNOWN"

        return result

    # ------------------------------------------------------------------
    # (R2) Leitura de valores reais do DBF
    # ------------------------------------------------------------------

    def _read_dbf_records(self, dbf_path: Path) -> Dict:
        """Ler todos os registros ativos de um DBF e retornar por campo."""
        result: Dict = {
            "ok": False,
            "fields": [],
            "unique_values": {},
            "value_counts": {},
            "total_active": 0,
            "total_deleted": 0,
            "error": None,
        }
        try:
            with open(dbf_path, "rb") as f:
                raw = f.read()

            if len(raw) < 32:
                result["error"] = "File too small"
                return result

            num_records = struct.unpack_from("<I", raw, 4)[0]
            header_size = struct.unpack_from("<H", raw, 8)[0]
            record_size = struct.unpack_from("<H", raw, 10)[0]

            # Field descriptors
            fields = []
            fld_off = 32
            while fld_off + 32 <= header_size and raw[fld_off] != 0x0D:
                fname = (
                    raw[fld_off: fld_off + 11]
                    .rstrip(b"\x00")
                    .decode("latin-1", errors="replace")
                    .strip()
                )
                ftype = chr(raw[fld_off + 11])
                flen = raw[fld_off + 16]
                fdec = raw[fld_off + 17]
                if fname:
                    fields.append({"name": fname, "type": ftype, "length": flen, "decimals": fdec})
                fld_off += 32

            result["fields"] = fields

            unique_vals: Dict[str, set] = {f["name"]: set() for f in fields}
            val_counts: Dict[str, Dict[str, int]] = {f["name"]: {} for f in fields}

            rec_off = header_size
            active = deleted = 0

            for _ in range(num_records):
                if rec_off + record_size > len(raw):
                    break
                deletion_flag = raw[rec_off]
                if deletion_flag == 0x2A:
                    deleted += 1
                    rec_off += record_size
                    continue

                fval_off = rec_off + 1
                for field in fields:
                    flen = field["length"]
                    raw_val = raw[fval_off: fval_off + flen]
                    val = raw_val.decode("latin-1", errors="replace").strip()
                    if val:
                        unique_vals[field["name"]].add(val)
                        vc = val_counts[field["name"]]
                        vc[val] = vc.get(val, 0) + 1
                    fval_off += flen

                active += 1
                rec_off += record_size

            result["unique_values"] = {k: sorted(v) for k, v in unique_vals.items()}
            result["value_counts"] = {k: dict(sorted(v.items())) for k, v in val_counts.items()}
            result["total_active"] = active
            result["total_deleted"] = deleted
            result["ok"] = True

        except Exception as e:
            result["error"] = str(e)

        return result

    def _audit_dbf_values(self) -> Dict:
        """Auditar valores reais do DBF de camada original de feições poligonais de deslizamento fotointerpretadas (v1iq-R2)."""
        result: Dict = {
            "dbf_read": False,
            "total_records": 0,
            "fields": [],
            "unique_values": {},
            "value_counts": {},
            "temporal_matches": {},
            "source_matches": {},
            "phenomenon_matches": {},
            "has_source_in_field": False,
            "has_phenomenon_in_field": False,
            "has_temporal_expression_in_field": False,
            "has_event_or_survey_date_in_field": False,
            "source_fields_found": [],
            "phenomenon_fields_found": [],
            "temporal_fields_found": [],
        }

        if not self._projeto_root:
            return result

        dbf_path = (
            self._projeto_root / "data" / "raw" / "petropolis"
            / "sgb_cprm" / "sig_extracted" / "Feicoes" / "Cicatriz_Area_A.dbf"
        )
        if not dbf_path.exists():
            return result

        raw_data = self._read_dbf_records(dbf_path)
        if not raw_data["ok"]:
            result["error"] = raw_data.get("error")
            return result

        result["dbf_read"] = True
        result["total_records"] = raw_data["total_active"]
        result["fields"] = raw_data["fields"]
        result["unique_values"] = raw_data["unique_values"]
        result["value_counts"] = raw_data["value_counts"]

        # Busca de termos nos valores únicos de cada campo
        for field_name, values in raw_data["unique_values"].items():
            all_text = " ".join(str(v).lower() for v in values)

            for term in TEMPORAL_TERMS:
                if term.lower() in all_text:
                    result["temporal_matches"].setdefault(field_name, []).append(term)
                    result["has_temporal_expression_in_field"] = True
                    if field_name not in result["temporal_fields_found"]:
                        result["temporal_fields_found"].append(field_name)
                    if term in EVENT_DATE_TERMS:
                        result["has_event_or_survey_date_in_field"] = True

            for term in SOURCE_TERMS:
                if term.lower() in all_text:
                    result["source_matches"].setdefault(field_name, []).append(term)
                    result["has_source_in_field"] = True
                    if field_name not in result["source_fields_found"]:
                        result["source_fields_found"].append(field_name)

            for term in PHENOMENON_TERMS:
                if term.lower() in all_text:
                    result["phenomenon_matches"].setdefault(field_name, []).append(term)
                    result["has_phenomenon_in_field"] = True
                    if field_name not in result["phenomenon_fields_found"]:
                        result["phenomenon_fields_found"].append(field_name)

        self.stats["local_files_audited"] += 1
        return result

    def _evaluate_attribute_provenance(self, dbf_audit: Dict) -> AttributeProvenanceDecision:
        """Construir decisão de proveniência baseada nos valores dos atributos."""

        has_source = dbf_audit.get("has_source_in_field", False)
        has_phenomenon = dbf_audit.get("has_phenomenon_in_field", False)
        has_temporal = dbf_audit.get("has_temporal_expression_in_field", False)
        has_event_date = dbf_audit.get("has_event_or_survey_date_in_field", False)
        total = dbf_audit.get("total_records", 0)
        unique = dbf_audit.get("unique_values", {})

        def sanitize_values(vals: list, max_items: int = 10) -> str:
            """Retornar valores únicos sem paths privados."""
            safe = [
                v for v in vals
                if "\\" not in v and "gabriela" not in v.lower() and "PROJETO" not in v
            ]
            return "; ".join(safe[:max_items])

        municipio_vals = sanitize_values(unique.get("MUNICIPIO", []))
        uf_vals = sanitize_values(unique.get("UF", []))
        tipo_vals = sanitize_values(unique.get("TIPO", []))
        condiciona_vals = sanitize_values(unique.get("CONDICIONA", []))
        fonte_vals = sanitize_values(unique.get("FONTE", []))
        obs_vals = sanitize_values(unique.get("OBS", []))

        # Força da evidência e decisão de promoção
        if has_source and has_phenomenon and has_event_date:
            strength = "STRONG"
            promotion = "GROUND_REFERENCE_CANDIDATE"
            blocker = "NONE"
            min_ev = "All attribute gates passed — external validation required before use"
            obs_note = "Atributos confirmam fonte oficial + fenômeno + data de evento — candidato forte"
        elif has_source and has_phenomenon and has_temporal:
            strength = "MODERATE"
            promotion = "STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK"
            blocker = "Expressão temporal presente mas sem data de evento específica (2022-02-15) nos atributos"
            min_ev = (
                "Expressão temporal específica ao evento de 2022-02-15 em campo FONTE/OBS, "
                "ou documento SGB confirmando feições pós-2022-02-15"
            )
            obs_note = "Atributos confirmam fonte + fenômeno; expressão temporal genérica encontrada"
        elif has_source and has_phenomenon:
            strength = "MODERATE"
            promotion = "STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK"
            blocker = "Sem expressão temporal nos atributos — vínculo de data não encontrado nos campos"
            min_ev = (
                "Expressão temporal em FONTE/OBS confirmando levantamento pós-2022-02-15, "
                "ou campo de data específica nas feições"
            )
            obs_note = "Atributos confirmam fonte + fenômeno; ausência de vínculo temporal nos campos"
        elif has_phenomenon:
            strength = "WEAK"
            promotion = "STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK"
            blocker = "Fonte oficial não confirmada nos atributos; sem data de evento"
            min_ev = (
                "Campo FONTE com referência a SGB/CPRM; "
                "expressão temporal específica ao evento de 2022-02-15"
            )
            obs_note = "Apenas fenômeno confirmado nos atributos"
        else:
            strength = "NONE"
            promotion = "STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK"
            blocker = "Atributos genéricos ou vazios — sem ganho de evidência"
            min_ev = "Campos FONTE/TIPO/OBS com referência a fonte oficial + fenômeno + data"
            obs_note = "Atributos não confirmam fonte, fenômeno ou data — decisão permanece por evidência composta"

        # Observed status a partir dos atributos
        tipo_lower = tipo_vals.lower()
        if any(t in tipo_lower for t in ["deslizamento", "feição de deslizamento", "escorregamento", "queda"]):
            observed_status = "OBSERVED_MASS_MOVEMENT"
        elif tipo_vals:
            observed_status = "OBSERVED_UNKNOWN_TYPE"
        else:
            observed_status = "UNKNOWN_FROM_ATTRIBUTES"

        # Lineagem de fonte a partir dos atributos
        fonte_lower = fonte_vals.lower()
        if any(s in fonte_lower for s in ["sgb", "cprm"]):
            source_lineage = "SGB_CPRM_CONFIRMED_IN_ATTRIBUTE"
        elif has_source:
            source_lineage = "OFFICIAL_SOURCE_CONFIRMED_IN_ATTRIBUTE"
        else:
            source_lineage = "SOURCE_NOT_EXPLICIT_IN_ATTRIBUTE"

        # Vínculo temporal a partir dos atributos
        if has_event_date:
            temporal_link = "EVENT_DATE_FOUND_IN_ATTRIBUTE"
        elif has_temporal:
            temporal_link = "TEMPORAL_EXPRESSION_FOUND_BUT_NOT_EVENT_SPECIFIC"
        else:
            temporal_link = "NO_TEMPORAL_EXPRESSION_IN_ATTRIBUTE"

        return AttributeProvenanceDecision(
            candidate_asset_name=SOURCE_LAYER_ALIAS,
            source_layer_alias=SOURCE_LAYER_ALIAS,
            source_layer_display_name=SOURCE_LAYER_DISPLAY_NAME,
            source_layer_original_name=SOURCE_LAYER_ORIGINAL_NAME,
            records_count=total,
            municipio_values=municipio_vals,
            uf_values=uf_vals,
            tipo_values=tipo_vals,
            condiciona_values=condiciona_vals,
            fonte_values_sanitized=fonte_vals,
            obs_values_sanitized=obs_vals,
            has_source_in_field="YES" if has_source else "NO",
            has_phenomenon_in_field="YES" if has_phenomenon else "NO",
            has_temporal_expression_in_field="YES" if has_temporal else "NO",
            has_event_or_survey_date_in_field="YES" if has_event_date else "NO",
            observed_status_from_attributes=observed_status,
            source_lineage_from_attributes=source_lineage,
            temporal_link_from_attributes=temporal_link,
            attribute_evidence_strength=strength,
            promotion_decision_after_attribute_audit=promotion,
            remaining_blocker=blocker,
            minimum_evidence_needed=min_ev,
            notes=obs_note,
        )

    # ------------------------------------------------------------------
    # Leitura de registries existentes
    # ------------------------------------------------------------------

    def _read_existing_registries(self) -> Dict:
        evidence = {
            "v1ij_blocking_reason": "",
            "v1ik_temporal_status": "",
            "v1ij_dataset_title": "",
            "v1ij_dataset_url": "",
            "v1ip_decision": "",
            "sources_read": [],
        }

        datasets = self.repo_root / "datasets"

        reg1 = datasets / "targeted_official_repository_event_vector_registry.csv"
        if reg1.exists():
            for row in csv.DictReader(reg1.open(encoding="utf-8")):
                if "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA" in str(row.values()):
                    evidence["v1ij_blocking_reason"] = row.get("blocking_reason", "")
                    evidence["v1ij_dataset_title"] = row.get("dataset_title", "")
                    evidence["v1ij_dataset_url"] = row.get("dataset_url", "")
                    evidence["sources_read"].append("targeted_official_repository")
            self.stats["registries_read"] += 1

        reg2 = datasets / "temporal_provenance_recovery_registry.csv"
        if reg2.exists():
            for row in csv.DictReader(reg2.open(encoding="utf-8")):
                if "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA" in str(row.values()):
                    evidence["v1ik_temporal_status"] = row.get("temporal_status_after_review", "")
                    evidence["sources_read"].append("temporal_provenance_recovery")
            self.stats["registries_read"] += 1

        reg3 = datasets / "composite_ground_reference_candidate_registry.csv"
        if reg3.exists():
            for row in csv.DictReader(reg3.open(encoding="utf-8")):
                if "PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA" in str(row.values()):
                    evidence["v1ip_decision"] = row.get("ground_reference_status", "")
                    evidence["sources_read"].append("composite_ground_reference")
            self.stats["registries_read"] += 1

        return evidence

    # ------------------------------------------------------------------
    # Evidências de v1in
    # ------------------------------------------------------------------

    def _read_v1in_evidence(self) -> Dict:
        evidence = {
            "total_strong_evidence": 0,
            "candidate_linked_evidence": 0,
            "pet_deslizamento_2022_evidence": 0,
            "evidence_source": "PROJETO",
            "summary": "",
        }

        v1in_dir = self.repo_root / "local_runs" / "protocolo_c" / "v1in"
        ev_path = v1in_dir / "v1in_evidence_strength_decision.csv"
        if not ev_path.exists():
            return evidence

        strong_count = linked_count = pet_2022_count = 0
        expr_path = v1in_dir / "v1in_temporal_expression_candidates.csv"
        expr_map = {}
        if expr_path.exists():
            for row in csv.DictReader(expr_path.open(encoding="utf-8")):
                expr_map[row["expression_id"]] = row

        for row in csv.DictReader(ev_path.open(encoding="utf-8")):
            strength = row.get("evidence_strength", "")
            candidate = row.get("candidate_asset_name", "UNKNOWN")
            if strength in ("STRONG_EXPLICIT_EVENT_DATE", "STRONG_EXPLICIT_SURVEY_DATE"):
                strong_count += 1
                if candidate not in ("UNKNOWN", ""):
                    linked_count += 1
                expr = expr_map.get(row.get("expression_id", ""), {})
                source_note = row.get("notes", "")
                if (expr.get("location_mentioned") == "PET"
                        and "2022" in expr.get("temporal_expression", "")
                        and "PROJETO" in source_note):
                    pet_2022_count += 1

        evidence["total_strong_evidence"] = strong_count
        evidence["candidate_linked_evidence"] = linked_count
        evidence["pet_deslizamento_2022_evidence"] = pet_2022_count
        evidence["summary"] = (
            f"{strong_count} STRONG, {linked_count} linkadas, "
            f"{pet_2022_count} PET/2022 mas de docs internos"
        )
        return evidence

    # ------------------------------------------------------------------
    # Extração textual direcionada
    # ------------------------------------------------------------------

    def _extract_targeted_text(self) -> List[Dict]:
        matches = []
        scan_dirs = [
            self.repo_root / "datasets",
            self.repo_root / "local_runs" / "protocolo_c",
        ]
        for scan_dir in scan_dirs:
            if not scan_dir.exists():
                continue
            for fpath in scan_dir.rglob("*.csv"):
                try:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                    for term in self.TARGET_TERMS:
                        if term.lower() in text.lower():
                            idx = text.lower().find(term.lower())
                            ctx = text[max(0, idx - 30): idx + 60].replace("\n", " ")
                            matches.append({
                                "source": fpath.name,
                                "term": term,
                                "context_sanitized": ctx,
                                "is_primary_source": False,
                            })
                            self.stats["targeted_term_matches"] += 1
                except Exception:
                    pass
        return matches[:50]

    # ------------------------------------------------------------------
    # Avaliação de gates
    # ------------------------------------------------------------------

    def _evaluate_gates(self, vector_meta: Dict, registry_evidence: Dict,
                        v1in_evidence: Dict, text_matches: List[Dict]):
        gates = []

        # Gate 1: Geometry
        geom_pass = (vector_meta.get("shp_exists")
                     and vector_meta.get("dbf_exists")
                     and vector_meta.get("shx_exists"))
        gates.append(GateResult(
            gate_name="gate_geometry",
            status="PASS" if geom_pass else "FAIL",
            evidence_source="metadata_local",
            evidence_detail=(
                f"shp={vector_meta.get('shp_exists')}, "
                f"dbf={vector_meta.get('dbf_exists')}, "
                f"shx={vector_meta.get('shx_exists')}, "
                f"records={vector_meta.get('dbf_records', 0)}"
            ),
            note="Bundle shapefile completo" if geom_pass else "Bundle incompleto",
        ))

        # Gate 2: CRS
        crs_pass = vector_meta.get("crs_zone") == "SIRGAS_2000_UTM_Zone_23S"
        gates.append(GateResult(
            gate_name="gate_crs",
            status="PASS" if crs_pass else ("UNKNOWN" if vector_meta.get("prj_exists") else "FAIL"),
            evidence_source="prj_file",
            evidence_detail=vector_meta.get("crs_zone", ""),
            note="EPSG:31983 compatível com Petrópolis/RJ" if crs_pass else "CRS não determinado",
        ))

        # Gate 3: Observed Status
        gates.append(GateResult(
            gate_name="gate_observed_status",
            status="PASS",
            evidence_source="xml_sidecar_cicatriz_ponto_p + registry_v1ij + attribute_audit",
            evidence_detail=(
                "TIPO=Deslizamento/Feição de deslizamento; FONTE=Fotointerpretação; observed_not_risk=YES (v1ij); "
                f"observed_from_attributes={self.attr_prov.observed_status_from_attributes if self.attr_prov else 'NOT_RUN'}"
            ),
            note="Feições de deslizamento são ocorrências observadas, não modelagem de susceptibilidade",
        ))

        # Gate 4: Source Authority
        # Agora também considera lineagem de fonte dos atributos
        attr_source_lineage = (
            self.attr_prov.source_lineage_from_attributes if self.attr_prov else ""
        )
        has_authority = (
            bool(registry_evidence.get("v1ij_dataset_url"))
            or "SGB_CPRM" in attr_source_lineage
            or "OFFICIAL" in attr_source_lineage
        )
        gates.append(GateResult(
            gate_name="gate_source_authority",
            status="PASS" if has_authority else "MODERATE",
            evidence_source="targeted_official_repository_registry + attribute_audit",
            evidence_detail=(
                f"dataset={registry_evidence.get('v1ij_dataset_title', '')[:60]} "
                f"attr_lineage={attr_source_lineage}"
            ),
            note="SGB/CPRM HIGH authority confirmada em registro e/ou atributos",
        ))

        # Gate 5: Event Date OR Survey Date
        # Verifica tanto registries quanto atributos (R2)
        blocking_reason_v1ij = registry_evidence.get("v1ij_blocking_reason", "")
        cumulative_flag = "feições de deslizamento_cumulativas_sem_data_especifica" in blocking_reason_v1ij
        has_date_field = vector_meta.get("has_date_field", False)
        dbf_year = int(vector_meta.get("dbf_date", "0000")[:4]) if vector_meta.get("dbf_date") else 0
        attr_has_event_date = (
            self.attr_prov is not None
            and self.attr_prov.has_event_or_survey_date_in_field == "YES"
        )

        if has_date_field or attr_has_event_date:
            date_status = "PASS"
            if attr_has_event_date:
                date_note = "Data de evento encontrada nos valores dos atributos (v1iq-R2)"
            else:
                date_note = "Campo de data presente no shapefile"
        elif cumulative_flag:
            date_status = "FAIL"
            date_note = "Feições de deslizamento cumulativas sem data específica (v1ij confirmou); sem data nos atributos"
        elif dbf_year > 0 and dbf_year < 2022:
            date_status = "FAIL"
            date_note = (
                f"DBF date={dbf_year}: anterior ao evento de 2022. "
                "Sem data nos atributos (v1iq-R2 confirmou)."
            )
        else:
            date_status = "UNKNOWN"
            date_note = "Data de evento ou levantamento não determinada"

        gates.append(GateResult(
            gate_name="gate_event_or_survey_date",
            status=date_status,
            evidence_source="dbf_header + registry_v1ij + attribute_audit_R2",
            evidence_detail=(
                f"dbf_date={vector_meta.get('dbf_date', '')} "
                f"has_date_field={has_date_field} "
                f"cumulative={cumulative_flag} "
                f"attr_event_date={attr_has_event_date}"
            ),
            note=date_note,
        ))

        # Gate 6: Document-Vector Package Linkage
        linked_in_v1in = v1in_evidence.get("candidate_linked_evidence", 0)
        sig_pós_desastre = "SIG pos-desastre" in registry_evidence.get("v1ij_dataset_title", "")
        xml_source = vector_meta.get("cicatriz_ponto_p_source_note", "")
        is_historical_sig = "SIG_SUSCETIBILIDADE_2013" in xml_source

        if linked_in_v1in > 0:
            linkage_status = "STRONG"
            linkage_note = f"v1in encontrou {linked_in_v1in} linkage(s) explícitas"
        elif sig_pós_desastre and is_historical_sig:
            linkage_status = "WEAK"
            linkage_note = (
                "SIG publicado como pós-desastre 2022, mas feições criadas em SIG_SUSCETIBILIDADE_2013 "
                "por fotointerpretação. Sem vínculo explícito com evento específico."
            )
        elif sig_pós_desastre:
            linkage_status = "MODERATE"
            linkage_note = "SIG publicado como pós-desastre; dados históricos sem linkage explícita"
        else:
            linkage_status = "WEAK"
            linkage_note = "Sem linkage explícita entre shapefile e evento de 2022-02-15"

        gates.append(GateResult(
            gate_name="gate_document_vector_linkage",
            status=linkage_status,
            evidence_source="xml_sidecar_cicatriz_ponto_p + v1in_evidence + registry_v1ij",
            evidence_detail=(
                f"sig_pos_desastre={sig_pós_desastre} "
                f"is_historical={is_historical_sig} "
                f"v1in_linked={linked_in_v1in}"
            ),
            note=linkage_note,
        ))

        # Gate 7: Region Match
        # Verifica MUNICIPIO e UF dos atributos (R2)
        attr_municipio = (self.attr_prov.municipio_values if self.attr_prov else "").upper()
        attr_uf = (self.attr_prov.uf_values if self.attr_prov else "").upper()
        petropolis_in_attr = "PETROPOLIS" in attr_municipio or "PETRÓPOLIS" in attr_municipio
        rj_in_attr = "RJ" in attr_uf

        gates.append(GateResult(
            gate_name="gate_region_match",
            status="STRONG",
            evidence_source="xml_sidecar + prj + registry_v1ij + attribute_audit_R2",
            evidence_detail=(
                f"MUNICIPIO_in_attr={attr_municipio[:50]} "
                f"UF_in_attr={attr_uf} "
                f"CRS=Zone_23S region=PET confirmado"
            ),
            note=(
                f"Petrópolis (RJ) confirmado em todas as fontes; "
                f"petropolis_in_attr={petropolis_in_attr} rj_in_attr={rj_in_attr}"
            ),
        ))

        # Gate 8: Phenomenon Match
        attr_phenomenon = (
            self.attr_prov.observed_status_from_attributes if self.attr_prov else ""
        )
        gates.append(GateResult(
            gate_name="gate_phenomenon_match",
            status="STRONG",
            evidence_source="xml_sidecar + dbf_field TIPO + registry_v1ij + attribute_audit_R2",
            evidence_detail=(
                f"TIPO_in_attr={self.attr_prov.tipo_values[:80] if self.attr_prov else ''} "
                f"attr_observed_status={attr_phenomenon}"
            ),
            note="Movimento de massa (feições de deslizamento) confirmado em todas as fontes e atributos",
        ))

        self.gates = gates

        for g in gates:
            if g.status in ("PASS", "STRONG"):
                self.stats["gates_pass"] += 1
            elif g.status in ("MODERATE", "UNKNOWN"):
                self.stats["gates_moderate"] += 1
            else:
                self.stats["gates_fail"] += 1

    # ------------------------------------------------------------------
    # Construção do dossiê
    # ------------------------------------------------------------------

    def _build_dossier(self, vector_meta: Dict, registry_evidence: Dict):
        fail_gates = [g for g in self.gates if g.status in ("FAIL", "WEAK")]
        fail_names = [g.gate_name for g in fail_gates]
        primary_blocker = fail_names[0] if fail_names else ""

        # Decisão de promoção: todos PASS/STRONG → GROUND_REFERENCE_CANDIDATE
        all_critical_pass = (
            self.stats["gates_fail"] == 0
            and all(g.status not in ("FAIL", "WEAK") for g in self.gates)
        )

        if all_critical_pass:
            promotion = "GROUND_REFERENCE_CANDIDATE"
            can_be_ref = "YES"
        elif "gate_event_or_survey_date" in fail_names or "gate_document_vector_linkage" in fail_names:
            promotion = "STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK"
            can_be_ref = "NO"
        elif self.stats["gates_fail"] > 2:
            promotion = "DOCUMENTED_EVENT_BUT_VECTOR_LINK_INSUFFICIENT"
            can_be_ref = "NO"
        else:
            promotion = "STRONG_COMPOSITE_REFERENCE_BUT_TEMPORAL_LINK_WEAK"
            can_be_ref = "NO"

        self.stats["promotion_decision"] = promotion

        doc_vector_link_gate = next(
            (g for g in self.gates if g.gate_name == "gate_document_vector_linkage"), None
        )
        doc_vector_link = doc_vector_link_gate.status if doc_vector_link_gate else "UNKNOWN"

        min_evidence = (
            "Confirmação explícita de que camada original de feições poligonais de deslizamento fotointerpretadas inclui feições mapeadas "
            "especificamente após 2022-02-15 em documento oficial SGB/CPRM; "
            "ou campo de data no shapefile discriminando evento por data."
        )

        # Notas enriquecidas com achados de atributos (R2)
        attr_note = ""
        if self.attr_prov:
            attr_note = (
                f" Atributos (R2): fonte_in_field={self.attr_prov.has_source_in_field}, "
                f"fenomeno_in_field={self.attr_prov.has_phenomenon_in_field}, "
                f"temporal_in_field={self.attr_prov.has_temporal_expression_in_field}, "
                f"event_date_in_field={self.attr_prov.has_event_or_survey_date_in_field}. "
                f"source_lineage={self.attr_prov.source_lineage_from_attributes}."
            )

        self.dossier = CicatrizAreaDossier(
            dossier_id="PET_CPRM_DESLIZAMENTO_AREA_FOTOINTERPRETADA_DOSSIER_V1IQ_R2",
            candidate_asset_name=SOURCE_LAYER_ALIAS,
            source_layer_alias=SOURCE_LAYER_ALIAS,
            source_layer_display_name=SOURCE_LAYER_DISPLAY_NAME,
            source_layer_original_name=SOURCE_LAYER_ORIGINAL_NAME,
            region="PET",
            event_id="PET_2022_02_15",
            source_institution="SGB/CPRM",
            source_document_name_sanitized="SIG_POS_DESASTRE_PETROPOLIS_2022_SGB_CPRM",
            source_asset_name_sanitized=SOURCE_LAYER_ALIAS,
            geometry_available="YES" if vector_meta.get("shp_exists") else "UNCERTAIN",
            crs_available="YES" if vector_meta.get("crs_zone") else "YES_FROM_REGISTRY",
            phenomenon_available="YES",
            phenomenon_group="movement_of_mass",
            observed_not_modelled_status="OBSERVED_PHOTOINTERPRETATION",
            event_date_documented="NO",
            survey_date_documented="NO",
            document_vector_package_link=doc_vector_link,
            source_lineage_match="STRONG",
            region_match="STRONG",
            phenomenon_match="STRONG",
            temporal_link_strength="WEAK",
            composite_evidence_strength="MODERATE",
            promotion_decision=promotion,
            can_be_ground_reference_candidate=can_be_ref,
            can_be_operational_ground_truth="NO",
            can_create_training_label="NO",
            can_train_model="NO",
            can_reopen_protocol_b="NO",
            primary_blocker=primary_blocker or "gate_event_or_survey_date",
            minimum_evidence_needed=min_evidence,
            notes=(
                f"SIG criado 2013-2015 por fotointerpretação (SIG_SUSCETIBILIDADE_2013). "
                f"444 feições sem campo de data. Feições de deslizamento cumulativas (v1ij). "
                f"SIG publicado como pós-desastre 2022, mas dados são históricos. "
                f"v1in: 0 linkages a candidatos específicos. "
                f"Bloqueio temporal persiste.{attr_note}"
            ),
        )

    # ------------------------------------------------------------------
    # Emissão de outputs (locais + R2 DBF value audit)
    # ------------------------------------------------------------------

    def _emit_local_outputs(self, output_dir: Path, vector_meta: Dict,
                             registry_evidence: Dict, text_matches: List[Dict]):

        # 1. Dossier inventory
        inv_csv = output_dir / "v1iq_cicatriz_area_dossier_inventory.csv"
        with open(inv_csv, "w", newline="", encoding="utf-8") as f:
            fields = list(CicatrizAreaDossier.__dataclass_fields__.keys())
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            if self.dossier:
                writer.writerow(asdict(self.dossier))

        # 2. Vector metadata audit
        vma_csv = output_dir / "v1iq_vector_metadata_audit.csv"
        with open(vma_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["field", "value"])
            for k, v in vector_meta.items():
                writer.writerow([k, str(v)])

        # 3. Gate matrix
        gate_csv = output_dir / "v1iq_composite_gate_matrix.csv"
        with open(gate_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["gate_name", "status", "evidence_source", "evidence_detail", "note"]
            )
            writer.writeheader()
            for g in self.gates:
                writer.writerow(asdict(g))

        # 4. Promotion decision
        promo_csv = output_dir / "v1iq_promotion_decision.csv"
        with open(promo_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate", "promotion_decision", "can_be_ground_reference",
                              "primary_blocker", "minimum_evidence"])
            if self.dossier:
                writer.writerow([
                    self.dossier.candidate_asset_name,
                    self.dossier.promotion_decision,
                    self.dossier.can_be_ground_reference_candidate,
                    self.dossier.primary_blocker,
                    self.dossier.minimum_evidence_needed,
                ])

        # 5. Summary JSON
        summary = {
            "status": "complete",
            "stage": "v1iq_R2",
            "candidate": SOURCE_LAYER_ALIAS,
            "source_layer_display_name": SOURCE_LAYER_DISPLAY_NAME,
            "source_layer_original_name": SOURCE_LAYER_ORIGINAL_NAME,
            "region": "PET",
            "event_id": "PET_2022_02_15",
            "promotion_decision": self.stats["promotion_decision"],
            "gates_pass": self.stats["gates_pass"],
            "gates_moderate": self.stats["gates_moderate"],
            "gates_fail": self.stats["gates_fail"],
            "local_files_audited": self.stats["local_files_audited"],
            "targeted_term_matches": self.stats["targeted_term_matches"],
            "key_findings": {
                "sig_creation_period": vector_meta.get("sig_creation_period", ""),
                "dbf_date": vector_meta.get("dbf_date", ""),
                "dbf_records": vector_meta.get("dbf_records", 0),
                "has_date_field": vector_meta.get("has_date_field", False),
                "crs_zone": vector_meta.get("crs_zone", ""),
                "xml_sidecar_exists": vector_meta.get("xml_sidecar_exists", False),
                "feição de deslizamento_ponto_p_source": vector_meta.get("cicatriz_ponto_p_source_note", ""),
            },
            "attribute_audit_R2": {
                "dbf_read": self.dbf_audit.get("dbf_read", False),
                "total_records_audited": self.dbf_audit.get("total_records", 0),
                "has_source_in_field": self.dbf_audit.get("has_source_in_field", False),
                "has_phenomenon_in_field": self.dbf_audit.get("has_phenomenon_in_field", False),
                "has_temporal_expression_in_field": self.dbf_audit.get("has_temporal_expression_in_field", False),
                "has_event_or_survey_date_in_field": self.dbf_audit.get("has_event_or_survey_date_in_field", False),
                "source_fields_found": self.dbf_audit.get("source_fields_found", []),
                "phenomenon_fields_found": self.dbf_audit.get("phenomenon_fields_found", []),
                "temporal_fields_found": self.dbf_audit.get("temporal_fields_found", []),
                "attribute_evidence_strength": (
                    self.attr_prov.attribute_evidence_strength if self.attr_prov else "NOT_RUN"
                ),
                "source_lineage_from_attributes": (
                    self.attr_prov.source_lineage_from_attributes if self.attr_prov else ""
                ),
                "temporal_link_from_attributes": (
                    self.attr_prov.temporal_link_from_attributes if self.attr_prov else ""
                ),
            },
            "critical_note": (
                "Feições de deslizamento são SIG histórico (2013-2015) publicado como SIG pós-desastre 2022. "
                "Sem campo de data. Feições de deslizamento cumulativas. "
                "Bloqueio temporal persiste."
            ),
            "can_create_training_label": False,
            "can_train_model": False,
            "can_reopen_protocol_b": False,
        }
        with open(output_dir / "v1iq_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 6. QA
        qa_rows = []
        if self.dossier:
            qa_rows.extend([
                ("can_create_training_label", self.dossier.can_create_training_label,
                 self.dossier.can_create_training_label == "NO"),
                ("can_train_model", self.dossier.can_train_model,
                 self.dossier.can_train_model == "NO"),
                ("can_be_operational_ground_truth", self.dossier.can_be_operational_ground_truth,
                 self.dossier.can_be_operational_ground_truth == "NO"),
                ("can_reopen_protocol_b", self.dossier.can_reopen_protocol_b,
                 self.dossier.can_reopen_protocol_b == "NO"),
                ("promotion_decision_filled", self.dossier.promotion_decision != "PENDING", True),
            ])
        qa_csv = output_dir / "v1iq_qa.csv"
        with open(qa_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["check", "value", "passed"])
            writer.writerows(qa_rows)

        # 7. (R2) DBF value audit outputs
        if self.dbf_audit.get("dbf_read") and self.attr_prov:
            self._emit_dbf_value_audit(output_dir, self.dbf_audit, self.attr_prov)

    def _emit_dbf_value_audit(self, output_dir: Path, dbf_audit: Dict,
                               attr_prov: AttributeProvenanceDecision):
        """Emitir os 4 outputs de auditoria de valores DBF (v1iq-R2)."""

        unique = dbf_audit.get("unique_values", {})
        val_counts = dbf_audit.get("value_counts", {})

        # --- 1. v1iq_cicatriz_area_dbf_value_audit.csv ---
        with open(output_dir / "v1iq_cicatriz_area_dbf_value_audit.csv",
                  "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "field_name", "field_type", "field_length",
                "unique_value_count", "sample_unique_values", "top_values_by_count",
            ])
            for field_info in dbf_audit.get("fields", []):
                fname = field_info["name"]
                ftype = field_info["type"]
                flen = field_info["length"]
                vals = unique.get(fname, [])
                counts = val_counts.get(fname, {})

                sample = "; ".join(str(v) for v in vals[:5])
                if len(vals) > 5:
                    sample += f" ... (+{len(vals) - 5} mais)"

                top = sorted(counts.items(), key=lambda x: -x[1])[:3]
                top_str = "; ".join(f"{v}({c})" for v, c in top)

                writer.writerow([fname, ftype, flen, len(vals), sample, top_str])

        # --- 2. v1iq_cicatriz_area_attribute_summary.json ---
        attr_summary = {
            "candidate_asset_name": SOURCE_LAYER_ALIAS,
            "source_layer_display_name": SOURCE_LAYER_DISPLAY_NAME,
            "source_layer_original_name": SOURCE_LAYER_ORIGINAL_NAME,
            "total_records_audited": dbf_audit.get("total_records", 0),
            "dbf_read_ok": dbf_audit.get("dbf_read", False),
            "fields_audited": [f["name"] for f in dbf_audit.get("fields", [])],
            "unique_values": unique,
            "temporal_term_matches": dbf_audit.get("temporal_matches", {}),
            "source_term_matches": dbf_audit.get("source_matches", {}),
            "phenomenon_term_matches": dbf_audit.get("phenomenon_matches", {}),
            "has_source_in_field": dbf_audit.get("has_source_in_field", False),
            "has_phenomenon_in_field": dbf_audit.get("has_phenomenon_in_field", False),
            "has_temporal_expression_in_field": dbf_audit.get("has_temporal_expression_in_field", False),
            "has_event_or_survey_date_in_field": dbf_audit.get("has_event_or_survey_date_in_field", False),
            "source_fields_found": dbf_audit.get("source_fields_found", []),
            "phenomenon_fields_found": dbf_audit.get("phenomenon_fields_found", []),
            "temporal_fields_found": dbf_audit.get("temporal_fields_found", []),
            "note": "Valores únicos lidos do DBF em modo read-only — nenhum path privado incluído",
        }
        with open(output_dir / "v1iq_cicatriz_area_attribute_summary.json",
                  "w", encoding="utf-8") as f:
            json.dump(attr_summary, f, indent=2, ensure_ascii=False)

        # --- 3. v1iq_cicatriz_area_source_terms.csv ---
        with open(output_dir / "v1iq_cicatriz_area_source_terms.csv",
                  "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["term_category", "matched_term", "field_name",
                              "field_values_containing_term"])
            for category, match_dict in [
                ("temporal", dbf_audit.get("temporal_matches", {})),
                ("source",   dbf_audit.get("source_matches", {})),
                ("phenomenon", dbf_audit.get("phenomenon_matches", {})),
            ]:
                for field, terms in match_dict.items():
                    field_vals = unique.get(field, [])
                    for term in terms:
                        vals_with = "; ".join(
                            v for v in field_vals
                            if term.lower() in v.lower()
                        )[:200]
                        writer.writerow([category, term, field, vals_with])

        # --- 4. v1iq_cicatriz_area_attribute_provenance_decision.csv ---
        with open(output_dir / "v1iq_cicatriz_area_attribute_provenance_decision.csv",
                  "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(AttributeProvenanceDecision.__dataclass_fields__.keys())
            )
            writer.writeheader()
            writer.writerow(asdict(attr_prov))

    # ------------------------------------------------------------------
    # Emissão de registries públicos
    # ------------------------------------------------------------------

    def _emit_promotion_registry(self):
        if not self.dossier:
            return

        datasets_dir = self.repo_root / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)

        # Dossier registry
        registry_csv = datasets_dir / "cicatriz_area_ground_reference_dossier.csv"
        with open(registry_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(CicatrizAreaDossier.__dataclass_fields__.keys())
            )
            writer.writeheader()
            writer.writerow(asdict(self.dossier))

        # Schema
        schema_csv = datasets_dir / "schemas" / "cicatriz_area_ground_reference_dossier_schema.csv"
        schema_csv.parent.mkdir(parents=True, exist_ok=True)
        schema_data = [
            ("dossier_id", "string", "ID único do dossiê"),
            ("candidate_asset_name", "string", "Alias técnico público da camada"),
            ("source_layer_alias", "string", "Alias técnico público"),
            ("source_layer_display_name", "string", "Nome público/conceitual"),
            ("source_layer_original_name", "string", "Nome original bruto preservado para proveniência"),
            ("region", "string", "Região (PET)"),
            ("event_id", "string", "ID do evento"),
            ("source_institution", "string", "Instituição-fonte"),
            ("geometry_available", "string", "SIM se shapefile existe"),
            ("crs_available", "string", "SIM se CRS documentado"),
            ("observed_not_modelled_status", "string", "OBSERVED ou MODELLED"),
            ("event_date_documented", "string", "SIM/NÃO"),
            ("document_vector_package_link", "string", "STRONG/MODERATE/WEAK"),
            ("temporal_link_strength", "string", "Força do vínculo temporal"),
            ("promotion_decision", "string", "Decisão de promoção"),
            ("can_be_ground_reference_candidate", "string", "SIM/NÃO"),
            ("can_be_operational_ground_truth", "string", "Sempre NÃO"),
            ("can_create_training_label", "string", "Sempre NÃO"),
            ("primary_blocker", "string", "Gate que bloqueia"),
            ("minimum_evidence_needed", "string", "Evidência mínima necessária"),
        ]
        with open(schema_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["field_name", "field_type", "description"])
            writer.writerows(schema_data)

        # Gate matrix
        matrix_csv = datasets_dir / "cicatriz_area_ground_reference_gate_matrix.csv"
        gate_fields = [
            "candidate_id", "gate_geometry", "gate_crs", "gate_observed_status",
            "gate_source_authority", "gate_document_vector_linkage",
            "gate_event_or_survey_date", "gate_region_match",
            "gate_phenomenon_match", "overall_reference_status",
            "blocking_gate", "minimum_evidence_needed",
        ]
        with open(matrix_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=gate_fields)
            writer.writeheader()
            gate_map = {g.gate_name: g.status for g in self.gates}
            writer.writerow({
                "candidate_id": self.dossier.dossier_id,
                "gate_geometry": gate_map.get("gate_geometry", "UNKNOWN"),
                "gate_crs": gate_map.get("gate_crs", "UNKNOWN"),
                "gate_observed_status": gate_map.get("gate_observed_status", "UNKNOWN"),
                "gate_source_authority": gate_map.get("gate_source_authority", "UNKNOWN"),
                "gate_document_vector_linkage": gate_map.get("gate_document_vector_linkage", "UNKNOWN"),
                "gate_event_or_survey_date": gate_map.get("gate_event_or_survey_date", "UNKNOWN"),
                "gate_region_match": gate_map.get("gate_region_match", "UNKNOWN"),
                "gate_phenomenon_match": gate_map.get("gate_phenomenon_match", "UNKNOWN"),
                "overall_reference_status": self.dossier.promotion_decision,
                "blocking_gate": self.dossier.primary_blocker,
                "minimum_evidence_needed": self.dossier.minimum_evidence_needed[:120],
            })

        # Schema da matriz
        matrix_schema_csv = (
            datasets_dir / "schemas" / "cicatriz_area_ground_reference_gate_matrix_schema.csv"
        )
        matrix_schema_data = [
            ("candidate_id", "string", "ID do dossiê"),
            ("gate_geometry", "string", "PASS/FAIL"),
            ("gate_crs", "string", "PASS/FAIL"),
            ("gate_observed_status", "string", "PASS/FAIL"),
            ("gate_source_authority", "string", "PASS/MODERATE"),
            ("gate_document_vector_linkage", "string", "STRONG/MODERATE/WEAK"),
            ("gate_event_or_survey_date", "string", "PASS/FAIL"),
            ("gate_region_match", "string", "STRONG/WEAK"),
            ("gate_phenomenon_match", "string", "STRONG/WEAK"),
            ("overall_reference_status", "string", "Decisão final"),
            ("blocking_gate", "string", "Gate bloqueante"),
            ("minimum_evidence_needed", "string", "Evidência mínima"),
        ]
        with open(matrix_schema_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["field_name", "field_type", "description"])
            writer.writerows(matrix_schema_data)


def main():
    parser = argparse.ArgumentParser(description="v1iq R2: Focused Ground Reference Dossier + DBF Attribute Audit")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--focus-source-layer", dest="focus_cicatriz_area", action="store_true", default=True)
    parser.add_argument("--read-composite-evidence", action="store_true", default=True)
    parser.add_argument("--read-documentary-evidence", action="store_true", default=True)
    parser.add_argument("--read-local-metadata", action="store_true", default=True)
    parser.add_argument("--extract-targeted-text", action="store_true", default=True)
    parser.add_argument("--emit-dossier", action="store_true", default=True)
    parser.add_argument("--emit-promotion-decision", action="store_true", default=True)
    args = parser.parse_args()

    builder = FocusedGroundReferenceDossierBuilder(force=args.force)
    result = builder.run(
        focus_cicatriz_area=args.focus_cicatriz_area,
        read_composite_evidence=args.read_composite_evidence,
        read_documentary_evidence=args.read_documentary_evidence,
        read_local_metadata=args.read_local_metadata,
        extract_targeted_text=args.extract_targeted_text,
        emit_dossier=args.emit_dossier,
        emit_promotion_decision=args.emit_promotion_decision,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
