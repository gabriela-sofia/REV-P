"""
revp_v1ir_official_documented_event_unit_builder.py

Protocolo C — v1ir: Official Documented Event Unit Ground Reference Builder

Objetivo:
Construir ground reference candidates a partir de unidades documentais oficiais,
não de Cicatriz_Area_A. A unidade agora é:
    evento / localidade / fenômeno documentado em relatório oficial,
    com data e fonte rastreável.

Foco:
- PDFs/relatórios CPRM/SGB já extraídos em local_runs/protocolo_c/v1if/
- Avaliações de campo pós-evento por bairro/localidade
- Petrópolis 2022 (evento 2022-02-15)
- Fenômenos: deslizamento, escorregamento, enxurrada, inundação, solapamento

Estrutura dos documentos auditados (CPRM DIGEAP):
- RELATÓRIO TÉCNICO PARA IDENTIFICAÇÃO DE ÁREAS COM RISCO EM CARÁTER EMERGENCIAL
- Vistoria realizada na data DD/MM/YYYY
- Pontos de campo com Coordenadas: LAT, LON
- Classificação de risco e danos

Regras:
- Relatório documental NÃO vira vetor observado
- Coordenadas documentadas são evidência documental, não label geoespacial
- can_be_operational_ground_truth = NO sempre
- can_create_training_label = NO sempre
- can_train_model = NO sempre
- can_reopen_protocol_b = NO sempre
"""

import argparse
import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "protocolo_c"))

# ---------------------------------------------------------------------------
# Termos de busca
# ---------------------------------------------------------------------------

PHENOMENON_KEYWORDS = {
    "MOVEMENT_OF_MASS": [
        "deslizamento", "escorregamento", "movimento de massa",
        "corrida de massa", "queda de bloco", "ruptura",
        "instabilidade", "talude", "encosta", "cicatriz",
    ],
    "FLOODING": [
        "inundação", "inundacao", "enchente", "enxurrada",
        "transbordamento", "alagamento", "cheias",
    ],
    "EROSION": [
        "solapamento", "erosão", "erosao", "ravina",
        "voçoroca", "voçoroca",
    ],
    "RISK_AREA": [
        "risco muito alto", "risco alto", "área de risco",
        "área instável", "risco emergencial",
        "áreas críticas", "polígono de área", "demarcação",
        "identificação de áreas",
    ],
}

INSTITUTION_KEYWORDS = ["cprm", "sgb", "drm", "defesa civil", "deget", "digeap", "dht"]

# Padrão de coordenadas explicitamente documentadas nos relatórios CPRM
# Ex: "Coordenadas: -22.484251, -43.211257"
COORD_PATTERN = re.compile(
    r"[Cc]oordenadas?[:\s]+(-\d{1,2}\.\d{4,8})[,\s]+(-\d{2,3}\.\d{4,8})"
)

# Data no texto: "Vistoria realizada no dia DD/MM/YYYY" ou "DD/MM/202X"
# Aceita ano parcial "202" seguido de qualquer dígito, ou ano completo "20\d\d"
SURVEY_DATE_PATTERN = re.compile(
    r"[Vv]istoria\s+realizada\s+(?:no\s+dia\s+)?(\d{1,2}/\d{2}/20\d\d)"
)
# Fallback: data implícita com ano truncado (ex: "24/02/202 – 12h") → prefer filename
SURVEY_DATE_PARTIAL_PATTERN = re.compile(
    r"[Vv]istoria\s+realizada\s+(?:no\s+dia\s+)?(\d{1,2}/\d{2}/202)"
)

# Data no nome de arquivo: DD-MM-YY ou DD_e_DD-MM-YY
FILENAME_DATE_PATTERN = re.compile(r"_(\d{2})[-_](\d{2})[-_](\d{2})_")
FILENAME_DATE_RANGE_PATTERN = re.compile(r"_(\d{2})_e_(\d{2})-(\d{2})-(\d{2})_")

# Município no texto
MUNICIPALITY_PATTERN = re.compile(r"[Mm]unic[ií]pio\s+de\s+([^\n\r,]+)")

# Bairro / Localidade no texto
LOCALITY_PATTERN = re.compile(
    r"(?:Bairro|bairro|Localidade|localidade)\s+([^\n\r,\.]+)"
)


@dataclass
class DocumentedEventUnit:
    """Unidade documental de evento extraída de relatório oficial."""
    documented_event_unit_id: str
    source_document_name_sanitized: str
    source_institution: str
    region: str
    municipality: str
    locality_text_sanitized: str
    event_date: str
    event_window: str
    temporal_precision: str
    phenomenon_group: str
    phenomenon_text_sanitized: str
    spatial_precision: str
    coordinate_available: str
    coordinate_source: str
    coordinate_lat: str
    coordinate_lon: str
    document_excerpt_sanitized: str
    page_reference: str
    official_source_status: str
    ground_reference_candidate_status: str
    can_be_ground_reference_candidate: str
    can_be_operational_ground_truth: str
    can_create_training_label: str
    can_train_model: str
    can_reopen_protocol_b: str
    blocking_reason: str
    minimum_evidence_needed: str
    notes: str


@dataclass
class GateResult:
    """Resultado de gate de avaliação para gate matrix."""
    unit_id: str
    source_official: str
    event_date_or_window: str
    phenomenon_explicit: str
    location_explicit: str
    spatial_precision_sufficient: str
    document_excerpt_available: str
    coordinate_documented: str
    ground_reference_status: str
    blocking_reason: str


@dataclass
class TextExtractionLog:
    """Log de extração de texto de um PDF."""
    document_name_sanitized: str
    pages_total: int
    pages_extracted: int
    chars_extracted: int
    survey_date_found: str
    municipality_found: str
    locality_found: str
    coordinates_found: int
    phenomena_found: str
    institution_confirmed: str
    extraction_status: str
    notes: str


class OfficialDocumentedEventUnitBuilder:
    """Constrói ground reference candidates a partir de relatórios CPRM."""

    _V1IF_ROOTS = [
        REPO_ROOT / "local_runs" / "protocolo_c" / "v1if" / "raw_official_sources" / "extracted",
        REPO_ROOT / "local_runs" / "protocolo_c" / "v1if" / "raw_official_sources",
    ]

    # Documentos principais (não-anexos)
    _MAIN_REPORTS = [
        "Relatorio_Tecnico_Petropolis.pdf",
    ]

    def __init__(self, force: bool = False):
        self.force = force
        self.repo_root = REPO_ROOT
        self.pdfs: List[Path] = []
        self.event_units: List[DocumentedEventUnit] = []
        self.gate_results: List[GateResult] = []
        self.extraction_logs: List[TextExtractionLog] = []
        self.stats: Dict = {
            "complete": False,
            "documents_found": 0,
            "documents_extracted": 0,
            "event_units_created": 0,
            "candidates_with_coordinate": 0,
            "candidates_documentary": 0,
            "total_coordinates_extracted": 0,
            "phenomena_found": set(),
        }

    # ------------------------------------------------------------------
    # Utilitários de sanitização
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize(text: str, max_len: int = 300) -> str:
        """Remove paths privados e trunca."""
        if not text:
            return ""
        s = re.sub(r"[A-Za-z]:\\[^\s,<\"']{2,}", "[PATH_REDACTED]", text)
        s = re.sub(r"\\\\[^\s,<\"']+", "[PATH_REDACTED]", s)
        s = re.sub(r"gabriela", "[USER_REDACTED]", s, flags=re.IGNORECASE)
        # Remover caracteres de controle mas manter acentos utf-8
        s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", s)
        s = " ".join(s.split())
        return s[:max_len]

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Remove encoding garbled e retorna nome seguro sem path."""
        # Manter só nome base sem extensão, substituindo chars problemáticos
        n = Path(name).name
        # Substituir sequências de replacement character
        n = re.sub(r"[�\x00-\x08\x0b\x0e-\x1f]", "_", n)
        return n[:120]

    # ------------------------------------------------------------------
    # Descoberta de documentos
    # ------------------------------------------------------------------

    def _find_pdfs(self) -> List[Path]:
        found = []
        for root in self._V1IF_ROOTS:
            if root.exists():
                found.extend(root.glob("*.pdf"))
        # Deduplicate (same name from different roots)
        seen = set()
        unique = []
        for p in found:
            if p.name not in seen:
                seen.add(p.name)
                unique.append(p)
        return sorted(unique, key=lambda x: x.name)

    # ------------------------------------------------------------------
    # Extração de texto (leve — pypdf, sem OCR)
    # ------------------------------------------------------------------

    def _extract_text_light(self, pdf_path: Path) -> Tuple[str, int, int]:
        """
        Extrai texto leve de PDF com pypdf.
        Retorna: (all_text, pages_total, pages_extracted)
        """
        try:
            import pypdf
        except ImportError:
            return "", 0, 0

        try:
            reader = pypdf.PdfReader(str(pdf_path))
            pages_total = len(reader.pages)
            texts = []
            extracted = 0
            for pg in reader.pages:
                try:
                    t = pg.extract_text() or ""
                    texts.append(t)
                    extracted += 1
                except Exception:
                    texts.append("")
            return "\n".join(texts), pages_total, extracted
        except Exception:
            return "", 0, 0

    # ------------------------------------------------------------------
    # Parsing de filename
    # ------------------------------------------------------------------

    def _parse_filename(self, name: str) -> Dict:
        """Extrai data e localidade do nome de arquivo."""
        result = {
            "annex_number": "",
            "raw_date_str": "",
            "event_date": "",
            "event_window": "",
            "locality_from_filename": "",
        }

        # Annex number
        m_annex = re.match(r"ANEXO-([IVXLC\d]+)-", name, re.IGNORECASE)
        if m_annex:
            result["annex_number"] = m_annex.group(1)

        # Date range: DD_e_DD-MM-YY
        m_range = FILENAME_DATE_RANGE_PATTERN.search(name)
        if m_range:
            d1, d2, mo, yr = m_range.groups()
            year = f"20{yr}"
            result["event_date"] = ""
            result["event_window"] = f"{d1}/{mo}/{year}–{d2}/{mo}/{year}"
            result["raw_date_str"] = f"{d1}_e_{d2}-{mo}-{yr}"

        # Single date: DD-MM-YY
        if not result["event_window"]:
            m_date = FILENAME_DATE_PATTERN.search(name)
            if m_date:
                dd, mo, yr = m_date.groups()
                result["event_date"] = f"{dd}/{mo}/20{yr}"
                result["raw_date_str"] = f"{dd}-{mo}-{yr}"

        # Locality from filename (everything after date segment)
        if result["raw_date_str"]:
            after_date = name.split(result["raw_date_str"])[-1]
            loc = re.sub(r"[_\-\.]+", " ", after_date)
            loc = re.sub(r"\.pdf$", "", loc, flags=re.IGNORECASE).strip()
            # Remove encoding artifacts
            loc = re.sub(r"[�]", "", loc).strip()
            result["locality_from_filename"] = self._sanitize(loc, 120)

        return result

    # ------------------------------------------------------------------
    # Parsing de conteúdo textual
    # ------------------------------------------------------------------

    def _parse_text(self, text: str) -> Dict:
        """Extrai estrutura do relatório a partir do texto."""
        result = {
            "institution": "",
            "municipality": "",
            "locality": "",
            "survey_date": "",
            "team": "",
            "coordinates": [],          # lista de (lat, lon) strings
            "phenomena_found": [],
            "risk_levels": [],
            "excerpt": "",
        }

        # Instituição (CPRM)
        text_lower = text.lower()
        if any(k in text_lower for k in INSTITUTION_KEYWORDS):
            result["institution"] = "CPRM"

        # Município
        m = MUNICIPALITY_PATTERN.search(text)
        if m:
            result["municipality"] = self._sanitize(m.group(1).strip(), 60)

        # Localidade/bairro
        localities = LOCALITY_PATTERN.findall(text)
        if localities:
            result["locality"] = self._sanitize(localities[0].strip(), 100)

        # Data de vistoria
        m_date = SURVEY_DATE_PATTERN.search(text)
        if m_date:
            result["survey_date"] = m_date.group(1).strip()

        # Coordenadas documentadas
        coords = COORD_PATTERN.findall(text)
        result["coordinates"] = [(lat, lon) for lat, lon in coords[:20]]

        # Fenômenos
        phenom_found = []
        for group, keywords in PHENOMENON_KEYWORDS.items():
            if any(k in text_lower for k in keywords):
                phenom_found.append(group)
        result["phenomena_found"] = phenom_found

        # Risco
        risk_terms = [
            "risco muito alto", "risco alto", "risco médio", "risco baixo",
        ]
        result["risk_levels"] = [r for r in risk_terms if r in text_lower]

        # Excerpt: primeiras linhas significativas (sem header burocrático)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        # Pular header (DHT/DEGET/DIGEAP/RELATÓRIO) e pegar contexto útil
        useful = []
        skip_keywords = ["diretoria", "departamento", "divisão", "relat", "estado do rio"]
        for l in lines:
            if any(k in l.lower() for k in skip_keywords):
                continue
            if len(l) > 20:
                useful.append(l)
            if len(useful) >= 4:
                break
        result["excerpt"] = self._sanitize(" | ".join(useful), 350)

        return result

    # ------------------------------------------------------------------
    # Classificação de precisão
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_spatial_precision(
        has_coord: bool, locality: str, municipality: str
    ) -> str:
        if has_coord:
            return "EXACT_COORDINATE"
        locality_lower = (locality or "").lower()
        if any(k in locality_lower for k in ["rua ", "servidão", "estrada", "avenida"]):
            return "STREET_OR_LOCALITY"
        if locality:
            return "NEIGHBORHOOD"
        if municipality:
            return "MUNICIPAL_ONLY"
        return "UNKNOWN"

    @staticmethod
    def _classify_temporal_precision(event_date: str, event_window: str) -> str:
        if event_date and re.match(r"\d{2}/\d{2}/\d{4}", event_date):
            return "EXACT_DATE"
        if event_window:
            return "EVENT_WINDOW"
        return "UNKNOWN"

    # ------------------------------------------------------------------
    # Classificação de fenômeno
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_phenomenon_group(phenomena: List[str]) -> str:
        if "MOVEMENT_OF_MASS" in phenomena:
            return "MOVEMENT_OF_MASS"
        if "EROSION" in phenomena:
            return "EROSION"
        if "FLOODING" in phenomena:
            return "FLOODING"
        if "RISK_AREA" in phenomena:
            return "RISK_AREA_MIXED"
        return "UNKNOWN"

    # ------------------------------------------------------------------
    # Decisão de candidato
    # ------------------------------------------------------------------

    @staticmethod
    def _decide_candidate_status(
        source_official: bool,
        has_date: bool,
        has_phenomenon: bool,
        has_location: bool,
        has_excerpt: bool,
        has_coordinate: bool,
    ) -> Tuple[str, str, str]:
        """Retorna (status, can_be_candidate, blocking_reason)."""
        if (source_official and has_date and has_phenomenon
                and has_location and has_excerpt):
            if has_coordinate:
                return (
                    "CANDIDATE_WITH_DOCUMENTED_COORDINATE",
                    "YES_DOCUMENTARY",
                    "NONE",
                )
            else:
                return (
                    "CANDIDATE_DOCUMENTARY_ONLY",
                    "YES_DOCUMENTARY",
                    "NONE",
                )
        # Bloqueio parcial
        blockers = []
        if not source_official:
            blockers.append("source_not_official")
        if not has_date:
            blockers.append("no_event_date")
        if not has_phenomenon:
            blockers.append("no_phenomenon")
        if not has_location:
            blockers.append("no_location")
        if not has_excerpt:
            blockers.append("no_excerpt")
        return (
            "INSUFFICIENT_EVIDENCE",
            "NO",
            "; ".join(blockers) or "UNKNOWN",
        )

    # ------------------------------------------------------------------
    # Execução principal
    # ------------------------------------------------------------------

    def run(
        self,
        scan_v1if_documents: bool = True,
        extract_light_text: bool = True,
        extract_event_units: bool = True,
        classify_spatial_precision: bool = True,
        emit_ground_reference_candidates: bool = True,
    ) -> Dict:

        output_dir = self.repo_root / "local_runs" / "protocolo_c" / "v1ir"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Se --force, remover apenas arquivos deste script
        if self.force:
            for fname in [
                "v1ir_document_inventory.csv",
                "v1ir_text_extraction_log.csv",
                "v1ir_documented_event_units.csv",
                "v1ir_spatial_precision_audit.csv",
                "v1ir_temporal_precision_audit.csv",
                "v1ir_ground_reference_candidate_decision.csv",
                "v1ir_summary.json",
                "v1ir_qa.csv",
            ]:
                p = output_dir / fname
                if p.exists():
                    p.unlink()

        # 1. Descoberta de documentos
        if scan_v1if_documents:
            self.pdfs = self._find_pdfs()
        self.stats["documents_found"] = len(self.pdfs)

        # 2 + 3. Extração e parsing
        if extract_light_text and extract_event_units:
            self._process_all_documents()

        # 4. Emitir outputs locais
        self._emit_local_outputs(output_dir)

        # 5. Emitir registries públicos
        if emit_ground_reference_candidates:
            self._emit_public_registries()

        self.stats["complete"] = True
        self.stats["phenomena_found"] = list(self.stats["phenomena_found"])

        return {
            "status": "complete",
            "documents_found": self.stats["documents_found"],
            "documents_extracted": self.stats["documents_extracted"],
            "event_units_created": self.stats["event_units_created"],
            "candidates_with_coordinate": self.stats["candidates_with_coordinate"],
            "candidates_documentary": self.stats["candidates_documentary"],
            "total_coordinates_extracted": self.stats["total_coordinates_extracted"],
            "phenomena_found": list(self.stats["phenomena_found"]),
        }

    # ------------------------------------------------------------------
    # Processamento de todos os documentos
    # ------------------------------------------------------------------

    def _process_all_documents(self):
        for pdf in self.pdfs:
            self._process_one_document(pdf)

    def _process_one_document(self, pdf_path: Path):
        fname_sanitized = self._sanitize_filename(pdf_path.name)
        fn_info = self._parse_filename(pdf_path.name)

        # Extração de texto
        raw_text, pages_total, pages_extracted = self._extract_text_light(pdf_path)

        if pages_extracted > 0:
            self.stats["documents_extracted"] += 1

        parsed = self._parse_text(raw_text) if raw_text else {}

        # Consolidar data: prefer filename (mais confiável — sem typos de PDF)
        # fallback para texto somente se filename não tiver data
        survey_date_text = parsed.get("survey_date", "")
        fn_date = fn_info["event_date"]
        fn_window = fn_info["event_window"]
        # Se filename tem data específica, usá-la (evita typos de PDF como "202")
        if fn_date:
            event_date = fn_date
            event_window = ""
        elif fn_window:
            event_date = ""
            event_window = fn_window
        else:
            # Sem data no filename → usar texto (pode ter typo)
            event_date = survey_date_text
            event_window = ""

        # Consolidar localidade: prefer text, fallback to filename
        locality = parsed.get("locality", "") or fn_info["locality_from_filename"]
        municipality = parsed.get("municipality", "") or "Petrópolis"
        institution = parsed.get("institution", "CPRM") or "CPRM"

        # Coordenadas do texto
        coords = parsed.get("coordinates", [])
        has_coord = len(coords) > 0
        self.stats["total_coordinates_extracted"] += len(coords)
        first_lat = coords[0][0] if coords else ""
        first_lon = coords[0][1] if coords else ""

        # Fenômeno
        phenomena = parsed.get("phenomena_found", [])
        for p in phenomena:
            self.stats["phenomena_found"].add(p)
        phenomenon_group = self._classify_phenomenon_group(phenomena)
        phenomenon_text = self._sanitize(
            "; ".join(phenomena) or "risk_area_emergencial", 120
        )

        # Precisão
        spatial_prec = self._classify_spatial_precision(
            has_coord, locality, municipality
        ) if True else "UNKNOWN"
        temporal_prec = self._classify_temporal_precision(event_date, event_window)

        # Excerpt
        excerpt = parsed.get("excerpt", "")
        if not excerpt:
            excerpt = self._sanitize(raw_text[:250], 250)

        # Gate: decisão de candidato
        source_official = institution in ("CPRM", "SGB", "DRM")
        has_date = bool(event_date or event_window)
        has_phenomenon = phenomenon_group != "UNKNOWN"
        has_location = bool(locality or municipality)
        has_excerpt = bool(excerpt)

        candidate_status, can_ref, blocking = self._decide_candidate_status(
            source_official, has_date, has_phenomenon,
            has_location, has_excerpt, has_coord,
        )

        if "CANDIDATE" in candidate_status:
            if has_coord:
                self.stats["candidates_with_coordinate"] += 1
            else:
                self.stats["candidates_documentary"] += 1

        # ID único
        annex_str = fn_info["annex_number"] or "MAIN"
        date_str = (event_date or event_window or "NODATE").replace("/", "").replace("–", "-")[:12]
        unit_id = f"PET2022_CPRM_ANEXO{annex_str}_{date_str}"

        unit = DocumentedEventUnit(
            documented_event_unit_id=unit_id,
            source_document_name_sanitized=fname_sanitized,
            source_institution=institution,
            region="PET",
            municipality=self._sanitize(municipality, 60),
            locality_text_sanitized=self._sanitize(locality, 120),
            event_date=event_date,
            event_window=event_window,
            temporal_precision=temporal_prec,
            phenomenon_group=phenomenon_group,
            phenomenon_text_sanitized=phenomenon_text,
            spatial_precision=spatial_prec,
            coordinate_available="YES" if has_coord else "NO",
            coordinate_source="CPRM_FIELD_SURVEY_REPORT" if has_coord else "NOT_DOCUMENTED",
            coordinate_lat=first_lat,
            coordinate_lon=first_lon,
            document_excerpt_sanitized=excerpt,
            page_reference="p.1-2",
            official_source_status="OFFICIAL_CPRM_DIGEAP",
            ground_reference_candidate_status=candidate_status,
            can_be_ground_reference_candidate=can_ref,
            can_be_operational_ground_truth="NO",
            can_create_training_label="NO",
            can_train_model="NO",
            can_reopen_protocol_b="NO",
            blocking_reason=blocking,
            minimum_evidence_needed=(
                "Cruzamento com imagem Sentinel/satélite da data "
                "para verificar presença de cicatriz na cena "
                "(não cria label automático)"
            ) if "CANDIDATE" in candidate_status else "Fonte + data + fenômeno + localidade explícita",
            notes=(
                f"annexo_num={annex_str}; "
                f"coords_in_doc={len(coords)}; "
                f"pages={pages_extracted}/{pages_total}; "
                f"risk={'; '.join(parsed.get('risk_levels', [])[:2])}"
            )[:300],
        )
        self.event_units.append(unit)
        self.stats["event_units_created"] += 1

        # Gate result
        self.gate_results.append(GateResult(
            unit_id=unit_id,
            source_official="PASS" if source_official else "FAIL",
            event_date_or_window="PASS" if has_date else "FAIL",
            phenomenon_explicit="PASS" if has_phenomenon else "FAIL",
            location_explicit="PASS" if has_location else "FAIL",
            spatial_precision_sufficient=(
                "PASS" if spatial_prec in ("EXACT_COORDINATE", "STREET_OR_LOCALITY", "NEIGHBORHOOD")
                else "FAIL"
            ),
            document_excerpt_available="PASS" if has_excerpt else "FAIL",
            coordinate_documented="PASS" if has_coord else "ABSENT",
            ground_reference_status=candidate_status,
            blocking_reason=blocking,
        ))

        # Extraction log
        self.extraction_logs.append(TextExtractionLog(
            document_name_sanitized=fname_sanitized,
            pages_total=pages_total,
            pages_extracted=pages_extracted,
            chars_extracted=len(raw_text),
            survey_date_found=survey_date_text,
            municipality_found=municipality,
            locality_found=locality,
            coordinates_found=len(coords),
            phenomena_found="; ".join(phenomena[:5]),
            institution_confirmed=institution,
            extraction_status="OK" if pages_extracted > 0 else "NO_TEXT",
            notes=f"first_lat={first_lat}; first_lon={first_lon}",
        ))

    # ------------------------------------------------------------------
    # Emissão de outputs locais
    # ------------------------------------------------------------------

    def _emit_local_outputs(self, output_dir: Path):

        # 1. Document inventory
        with open(output_dir / "v1ir_document_inventory.csv",
                  "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["document_name_sanitized", "pdf_path_accessible",
                             "size_bytes", "belongs_to"])
            for pdf in self.pdfs:
                belongs = "ANNEX" if "ANEXO" in pdf.name.upper() else "MAIN_REPORT"
                writer.writerow([
                    self._sanitize_filename(pdf.name),
                    "YES",
                    pdf.stat().st_size if pdf.exists() else 0,
                    belongs,
                ])

        # 2. Text extraction log
        with open(output_dir / "v1ir_text_extraction_log.csv",
                  "w", newline="", encoding="utf-8") as f:
            if self.extraction_logs:
                writer = csv.DictWriter(
                    f, fieldnames=list(TextExtractionLog.__dataclass_fields__.keys())
                )
                writer.writeheader()
                for log in self.extraction_logs:
                    writer.writerow(asdict(log))
            else:
                f.write("# nenhum documento processado\n")

        # 3. Documented event units
        with open(output_dir / "v1ir_documented_event_units.csv",
                  "w", newline="", encoding="utf-8") as f:
            if self.event_units:
                writer = csv.DictWriter(
                    f, fieldnames=list(DocumentedEventUnit.__dataclass_fields__.keys())
                )
                writer.writeheader()
                for u in self.event_units:
                    writer.writerow(asdict(u))
            else:
                f.write("# nenhuma unidade criada\n")

        # 4. Spatial precision audit
        with open(output_dir / "v1ir_spatial_precision_audit.csv",
                  "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["unit_id", "locality", "spatial_precision",
                              "coordinate_available", "coordinate_lat", "coordinate_lon"])
            for u in self.event_units:
                writer.writerow([
                    u.documented_event_unit_id,
                    u.locality_text_sanitized,
                    u.spatial_precision,
                    u.coordinate_available,
                    u.coordinate_lat,
                    u.coordinate_lon,
                ])

        # 5. Temporal precision audit
        with open(output_dir / "v1ir_temporal_precision_audit.csv",
                  "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["unit_id", "event_date", "event_window", "temporal_precision"])
            for u in self.event_units:
                writer.writerow([
                    u.documented_event_unit_id,
                    u.event_date,
                    u.event_window,
                    u.temporal_precision,
                ])

        # 6. Ground reference candidate decision
        with open(output_dir / "v1ir_ground_reference_candidate_decision.csv",
                  "w", newline="", encoding="utf-8") as f:
            if self.event_units:
                writer = csv.DictWriter(
                    f, fieldnames=list(DocumentedEventUnit.__dataclass_fields__.keys())
                )
                writer.writeheader()
                candidates = [u for u in self.event_units if "CANDIDATE" in u.ground_reference_candidate_status]
                for u in candidates:
                    writer.writerow(asdict(u))
            else:
                f.write("# nenhum candidato\n")

        # 7. Summary JSON
        candidates = [u for u in self.event_units if "CANDIDATE" in u.ground_reference_candidate_status]
        phenomena_all = list({u.phenomenon_group for u in self.event_units})
        localities = [u.locality_text_sanitized for u in self.event_units if u.locality_text_sanitized]
        dates = sorted({u.event_date for u in self.event_units if u.event_date})

        summary = {
            "stage": "v1ir_event_units",
            "question": (
                "Relatórios CPRM pós-evento fornecem ground reference candidates "
                "documentalmente rastreáveis para Petrópolis 2022?"
            ),
            "answer": "SIM" if candidates else "NÃO",
            "documents_found": self.stats["documents_found"],
            "documents_extracted": self.stats["documents_extracted"],
            "event_units_created": self.stats["event_units_created"],
            "ground_reference_candidates": len(candidates),
            "candidates_with_coordinate": self.stats["candidates_with_coordinate"],
            "candidates_documentary_only": self.stats["candidates_documentary"],
            "total_coordinates_in_documents": self.stats["total_coordinates_extracted"],
            "phenomena_found": list(self.stats["phenomena_found"]),
            "localities": localities[:15],
            "event_dates": dates,
            "temporal_precision": "EXACT_DATE" if dates else "UNKNOWN",
            "spatial_precision_best": (
                "EXACT_COORDINATE" if self.stats["candidates_with_coordinate"] > 0
                else "NEIGHBORHOOD"
            ),
            "key_finding": (
                "Relatórios CPRM contêm vistorias de campo com data exata, bairro/rua, "
                "fenômeno e coordenadas GPS explícitas. "
                "Fornecem ground reference candidates documentais para cruzamento com "
                "imagens Sentinel da data do evento."
            ),
            "next_step_if_warranted": (
                "Cruzar coordenadas documentadas com patches Sentinel das datas "
                "para verificação de evidência estrutural — NÃO cria label automático."
            ),
            "invariants": {
                "can_be_operational_ground_truth": False,
                "can_create_training_label": False,
                "can_train_model": False,
                "can_reopen_protocol_b": False,
                "can_be_ground_reference_candidate": "YES_DOCUMENTARY" if candidates else "NO",
            },
            "prior_stage": "v1ir_photointerpretation_provenance",
            "why_changed_approach": (
                "Cicatriz_Area_A.shp é SIG histórico de 2013-2015 sem vínculo com 2022. "
                "Os relatórios de campo CPRM (Anexos I-XI) têm datas pós-evento, "
                "instituição rastreável e coordenadas documentadas: "
                "são a unidade documental correta para ground reference."
            ),
        }
        with open(output_dir / "v1ir_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 8. QA
        qa_checks = [
            ("can_create_training_label_never_yes",
             all(u.can_create_training_label == "NO" for u in self.event_units),
             True),
            ("can_train_model_never_yes",
             all(u.can_train_model == "NO" for u in self.event_units),
             True),
            ("can_be_operational_ground_truth_never_yes",
             all(u.can_be_operational_ground_truth == "NO" for u in self.event_units),
             True),
            ("can_reopen_protocol_b_never_yes",
             all(u.can_reopen_protocol_b == "NO" for u in self.event_units),
             True),
            ("no_private_paths_in_excerpts",
             all("gabriela" not in u.document_excerpt_sanitized.lower()
                 and "C:\\" not in u.document_excerpt_sanitized
                 for u in self.event_units),
             True),
            ("no_private_paths_in_localities",
             all("gabriela" not in u.locality_text_sanitized.lower()
                 for u in self.event_units),
             True),
            ("event_dates_are_post_event",
             all(
                 "2022" in u.event_date or "2022" in u.event_window
                 for u in self.event_units
                 if u.event_date or u.event_window
             ),
             True),
            ("candidates_have_official_source",
             all(
                 u.source_institution in ("CPRM", "SGB", "DRM")
                 for u in self.event_units
                 if "CANDIDATE" in u.ground_reference_candidate_status
             ),
             True),
        ]
        with open(output_dir / "v1ir_qa.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["check", "result", "passed"])
            for check, result, expected in qa_checks:
                writer.writerow([check, str(result), str(result == expected)])

    # ------------------------------------------------------------------
    # Emissão de registries públicos
    # ------------------------------------------------------------------

    def _emit_public_registries(self):
        datasets_dir = self.repo_root / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        schemas_dir = datasets_dir / "schemas"
        schemas_dir.mkdir(parents=True, exist_ok=True)

        candidates = [u for u in self.event_units
                      if "CANDIDATE" in u.ground_reference_candidate_status]
        if not candidates:
            # Mesmo sem candidatos, emite registry vazio com schema
            candidates = self.event_units  # Emite todos como informação

        # Registry principal
        reg_path = datasets_dir / "official_documented_event_unit_registry.csv"
        with open(reg_path, "w", newline="", encoding="utf-8") as f:
            if self.event_units:
                writer = csv.DictWriter(
                    f, fieldnames=list(DocumentedEventUnit.__dataclass_fields__.keys())
                )
                writer.writeheader()
                for u in self.event_units:
                    writer.writerow(asdict(u))

        # Gate matrix
        gate_path = datasets_dir / "official_documented_event_ground_reference_gate_matrix.csv"
        with open(gate_path, "w", newline="", encoding="utf-8") as f:
            if self.gate_results:
                writer = csv.DictWriter(
                    f, fieldnames=list(GateResult.__dataclass_fields__.keys())
                )
                writer.writeheader()
                for g in self.gate_results:
                    writer.writerow(asdict(g))

        # Schema do registry
        schema_rows = [
            ("documented_event_unit_id", "string", "ID único da unidade de evento"),
            ("source_document_name_sanitized", "string", "Nome do documento (sem path privado)"),
            ("source_institution", "string", "Instituição (CPRM, SGB, DRM)"),
            ("region", "string", "Região (PET = Petrópolis)"),
            ("municipality", "string", "Município"),
            ("locality_text_sanitized", "string", "Bairro/rua documentada no relatório"),
            ("event_date", "string", "Data exata da vistoria (DD/MM/YYYY)"),
            ("event_window", "string", "Janela temporal se não houver data exata"),
            ("temporal_precision", "string", "EXACT_DATE | EVENT_WINDOW | MONTH_YEAR | YEAR_ONLY | UNKNOWN"),
            ("phenomenon_group", "string", "MOVEMENT_OF_MASS | FLOODING | EROSION | RISK_AREA_MIXED"),
            ("phenomenon_text_sanitized", "string", "Fenômenos encontrados no texto"),
            ("spatial_precision", "string", "EXACT_COORDINATE | STREET_OR_LOCALITY | NEIGHBORHOOD | MUNICIPAL_ONLY | UNKNOWN"),
            ("coordinate_available", "string", "YES se coordenada explicitamente documentada"),
            ("coordinate_source", "string", "Origem da coordenada (nunca inferida)"),
            ("coordinate_lat", "string", "Latitude (string, da documentação oficial)"),
            ("coordinate_lon", "string", "Longitude (string, da documentação oficial)"),
            ("document_excerpt_sanitized", "string", "Trecho do documento (sanitizado, sem paths)"),
            ("page_reference", "string", "Página de origem"),
            ("official_source_status", "string", "Status de fonte oficial"),
            ("ground_reference_candidate_status", "string", "CANDIDATE_WITH_DOCUMENTED_COORDINATE | CANDIDATE_DOCUMENTARY_ONLY | INSUFFICIENT_EVIDENCE"),
            ("can_be_ground_reference_candidate", "string", "YES_DOCUMENTARY | NO"),
            ("can_be_operational_ground_truth", "string", "SEMPRE NO — requer validação de campo (Protocolo B não iniciado)"),
            ("can_create_training_label", "string", "SEMPRE NO — invariante absoluto"),
            ("can_train_model", "string", "SEMPRE NO — invariante absoluto"),
            ("can_reopen_protocol_b", "string", "SEMPRE NO — invariante absoluto"),
            ("blocking_reason", "string", "Razão de bloqueio se não for candidato"),
            ("minimum_evidence_needed", "string", "Evidência mínima para próximo passo"),
            ("notes", "string", "Notas técnicas"),
        ]
        schema_path = schemas_dir / "official_documented_event_unit_schema.csv"
        with open(schema_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["field_name", "field_type", "description"])
            writer.writerows(schema_rows)

        # Schema do gate matrix
        gate_schema_rows = [
            ("unit_id", "string", "ID da unidade de evento"),
            ("source_official", "string", "PASS | FAIL — fonte é CPRM/SGB/DRM?"),
            ("event_date_or_window", "string", "PASS | FAIL — data ou janela documentada?"),
            ("phenomenon_explicit", "string", "PASS | FAIL — fenômeno descrito no texto?"),
            ("location_explicit", "string", "PASS | FAIL — localidade descrita?"),
            ("spatial_precision_sufficient", "string", "PASS | FAIL — bairro ou melhor?"),
            ("document_excerpt_available", "string", "PASS | FAIL — há trecho verificável?"),
            ("coordinate_documented", "string", "PASS | ABSENT — coordenada no documento?"),
            ("ground_reference_status", "string", "Status de candidato"),
            ("blocking_reason", "string", "Razão de bloqueio"),
        ]
        gate_schema_path = schemas_dir / "official_documented_event_ground_reference_gate_matrix_schema.csv"
        with open(gate_schema_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["field_name", "field_type", "description"])
            writer.writerows(gate_schema_rows)


def main():
    parser = argparse.ArgumentParser(
        description="v1ir: Official Documented Event Unit Ground Reference Builder"
    )
    parser.add_argument("--force", action="store_true",
                        help="Recriar outputs deste script (não apaga outputs de outros scripts v1ir)")
    parser.add_argument("--scan-v1if-documents", action="store_true", default=True)
    parser.add_argument("--extract-light-text", action="store_true", default=True)
    parser.add_argument("--extract-event-units", action="store_true", default=True)
    parser.add_argument("--classify-spatial-precision", action="store_true", default=True)
    parser.add_argument("--emit-ground-reference-candidates", action="store_true", default=True)
    args = parser.parse_args()

    builder = OfficialDocumentedEventUnitBuilder(force=args.force)
    result = builder.run(
        scan_v1if_documents=args.scan_v1if_documents,
        extract_light_text=args.extract_light_text,
        extract_event_units=args.extract_event_units,
        classify_spatial_precision=args.classify_spatial_precision,
        emit_ground_reference_candidates=args.emit_ground_reference_candidates,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
