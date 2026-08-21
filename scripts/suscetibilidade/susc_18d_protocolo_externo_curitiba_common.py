"""SUSC-18D protocolo externo e ingestao de resposta oficial Curitiba.

Esta etapa nao busca dado fora do repositorio, nao inventa geometria e nao muda
score_v6. Ela transforma o pacote 18C em protocolo executavel de solicitacao
externa e prepara a ingestao de resposta oficial quando ela chegar em
local_runs/suscetibilidade/18d_resposta_oficial_curitiba/.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

import susc_18c_curitiba_geometria_common as s18c  # noqa: E402
from susc_io import ensure_dir, read_csv, read_json, rel, write_csv, write_json, write_markdown  # noqa: E402

OUT_18C = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_18c_aquisicao_geometria_oficial_curitiba"
OUT_DATA = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_18d_protocolo_externo_curitiba"
INPUT_DIR = ROOT / "local_runs" / "suscetibilidade" / "18d_resposta_oficial_curitiba"
DAT_SUSC = ROOT / "datasets" / "suscetibilidade"
SCORE_V6 = DAT_SUSC / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT_SUSC / "susc_score_v7_candidate_by_patch_v1.csv"

OFICIO_DC = OUT_DATA / "oficio_solicitacao_defesa_civil_curitiba.md"
OFICIO_IPPUC = OUT_DATA / "oficio_solicitacao_geocuritiba_ippuc.md"
CHECKLIST = OUT_DATA / "checklist_envio_solicitacao.csv"
PROTOCOLO = OUT_DATA / "protocolo_acompanhamento_resposta.csv"
SCHEMA_RESPOSTA = OUT_DATA / "schema_resposta_oficial_curitiba.json"
MODELO_RESPOSTA = OUT_DATA / "modelo_resposta_oficial_curitiba.csv"
INSTRUCOES = OUT_DATA / "instrucoes_ingestao_resposta.md"
STATUS_PROTOCOLO = OUT_DATA / "status_protocolo_externo_curitiba.csv"
MATRIZ_INGESTAO = OUT_DATA / "matriz_ingestao_resposta_curitiba.csv"
GEOMETRIAS = OUT_DATA / "geometrias_resposta_oficial_curitiba.csv"
VINCULOS = OUT_DATA / "vinculos_resposta_patch_curitiba.csv"
RELATORIO_INGESTAO = OUT_DATA / "relatorio_ingestao_resposta_curitiba.md"
GATE_CUR = OUT_DATA / "gate_curitiba_pos_18d.csv"
GATE_17B = OUT_DATA / "gate_prontidao_17b_pos_18d.csv"
SUMMARY = OUT_DATA / "summary.json"
PREFLIGHT = OUT_DATA / "preflight.json"

REQUIRED_18C = [
    OUT_18C / "summary.json",
    OUT_18C / "solicitacao_geometria_ocorrencia_curitiba.md",
    OUT_18C / "schema_resposta_esperada_curitiba.json",
    OUT_18C / "modelo_planilha_resposta_curitiba.csv",
]

REQUIRED_OUTPUTS = [
    OFICIO_DC, OFICIO_IPPUC, CHECKLIST, PROTOCOLO, SCHEMA_RESPOSTA,
    MODELO_RESPOSTA, INSTRUCOES, STATUS_PROTOCOLO, MATRIZ_INGESTAO,
    GEOMETRIAS, VINCULOS, RELATORIO_INGESTAO, GATE_CUR, GATE_17B,
    SUMMARY, PREFLIGHT,
]

HARD_DEFAULTS = {
    "ground_truth": "false",
    "eligible_for_training": "false",
    "score_v7_allowed": "false",
    "review_only": "true",
}

STATUS_18D_ALLOWED = [
    "18D_AGUARDANDO_RESPOSTA_OFICIAL",
    "18D_RESPOSTA_OFICIAL_INGERIDA_COM_OVERLAY",
    "18D_RESPOSTA_RECEBIDA_BLOQUEADA",
    "18D_BLOQUEADO_FAIL_CLOSED",
]

STATUS_17B_ALLOWED = [
    "17B_BLOQUEADO_POR_GEOMETRIA",
    "17B_APROXIMACAO_REGIONAL_COM_REFERENCIAS_PARCIAS",
    "17B_APROXIMACAO_COM_SEGUNDA_REGIAO_FORTE",
    "17B_BLOQUEADO_FAIL_CLOSED",
]

GEOM_FIELDS = [
    "geometry_id", "candidate_event_id", "data_evento", "tipo_fenomeno",
    "geometry_type", "crs", "lon", "lat", "bbox", "uncertainty_m",
    "geometry_source_type", "is_event_specific", "geometria_de_ocorrencia",
    "status_geometria", "review_only", "justificativa_tecnica",
]

VINCULO_FIELDS = [
    "patch_link_id", "candidate_event_id", "geometry_id", "geometry_type",
    "patch_id", "classe_vinculo", "metrica", "vinculo_forte",
    "review_only", "justificativa_tecnica",
]

MATRIZ_FIELDS = [
    "linha_id", "arquivo_origem", "candidate_event_id", "data_evento",
    "tipo_fenomeno", "geometry_id", "status_ingestao", "status_geometria",
    "patch_links", "patch_links_fortes", "ground_truth",
    "eligible_for_training", "score_v7_allowed", "review_only",
    "justificativa_tecnica",
]

GATE_FIELDS = ["criterio", "valor_observado", "limiar", "passou", "observacao"]
STATUS_FIELDS = ["item", "status", "responsavel_sugerido", "prazo_sugerido", "observacao"]
CHECKLIST_FIELDS = ["ordem", "item", "status", "evidencia_esperada"]
PROTOCOLO_FIELDS = ["protocolo_id", "destinatario", "status", "data_envio", "data_retorno", "proxima_acao"]

PUBLIC_FORBIDDEN_RE = re.compile(
    r"\b(?:agentic|agente|codex|llm|ia|automacao|automatizacao|automatizado|automatica|automatico)\b",
    re.IGNORECASE,
)


def _bool(v):
    return "true" if v else "false"


def _run_git(args):
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _score_v6_changed():
    return bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]))


def _input_files(input_dir=INPUT_DIR):
    if not Path(input_dir).exists():
        return []
    allowed = {".csv", ".json", ".geojson", ".wkt", ".shp"}
    return [p for p in sorted(Path(input_dir).glob("*")) if p.is_file() and p.suffix.lower() in allowed]


def _csv_records(path):
    rows = read_csv(path)
    for row in rows:
        row["_arquivo_origem"] = rel(path)
    return rows


def _json_records(path):
    rows = s18c._json_records(path)
    for row in rows:
        row["_arquivo_origem"] = rel(path)
    return rows


def _wkt_records(path):
    rows = s18c._wkt_records(path)
    for row in rows:
        row["_arquivo_origem"] = rel(path)
    return rows


def _shp_records(path):
    try:
        import shapefile  # type: ignore
    except Exception:
        return []
    rows = []
    reader = shapefile.Reader(str(path))
    fields = [f[0] for f in reader.fields[1:]]
    for sr in reader.iterShapeRecords():
        row = {k: v for k, v in zip(fields, sr.record)}
        bbox = sr.shape.bbox if getattr(sr.shape, "bbox", None) else []
        if len(bbox) == 4:
            row.setdefault("geometry_type", "bbox")
            row.setdefault("bbox", ",".join(str(v) for v in bbox))
        row["_arquivo_origem"] = rel(path)
        rows.append(row)
    return rows


def load_response_records(input_dir=INPUT_DIR):
    rows = []
    for path in _input_files(input_dir):
        suffix = path.suffix.lower()
        if suffix == ".csv":
            rows.extend(_csv_records(path))
        elif suffix in {".json", ".geojson"}:
            rows.extend(_json_records(path))
        elif suffix == ".wkt":
            rows.extend(_wkt_records(path))
        elif suffix == ".shp":
            rows.extend(_shp_records(path))
    return rows


def _coerce_record(rec):
    out = dict(rec)
    if "geometry_source" in out and "geometry_source_type" not in out:
        out["geometry_source_type"] = out["geometry_source"]
    if "geometry_type" in out and "geom_kind" not in out:
        out["geom_kind"] = out["geometry_type"]
    if "data_ocorrencia" in out and "data_evento" not in out:
        out["data_evento"] = out["data_ocorrencia"]
    if "evento_id" in out and "candidate_event_id" not in out:
        out["candidate_event_id"] = out["evento_id"]
    out.setdefault("crs", "EPSG:4326")
    out.setdefault("is_event_specific", "true")
    return out


def _normalize_crs(rec):
    crs = str(rec.get("crs", "EPSG:4326")).upper()
    if crs in {"EPSG:4326", "4326", "WGS84", "WGS 84"}:
        rec["crs"] = "EPSG:4326"
        return rec, None
    try:
        from pyproj import Transformer  # type: ignore
    except Exception:
        return rec, f"crs_nao_convertido:{crs}"
    try:
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        if rec.get("lon") not in {"", None} and rec.get("lat") not in {"", None}:
            lon, lat = transformer.transform(float(rec["lon"]), float(rec["lat"]))
            rec["lon"], rec["lat"] = str(lon), str(lat)
        bbox = s18c._parse_bbox(rec.get("bbox", ""))
        if bbox:
            lon1, lat1 = transformer.transform(bbox[0], bbox[1])
            lon2, lat2 = transformer.transform(bbox[2], bbox[3])
            rec["bbox"] = f"{min(lon1, lon2)},{min(lat1, lat2)},{max(lon1, lon2)},{max(lat1, lat2)}"
        rec["crs"] = "EPSG:4326"
        return rec, None
    except Exception as exc:
        return rec, f"crs_nao_convertido:{crs}:{exc.__class__.__name__}"


def normalize_response_record(rec, idx=1):
    coerced = _coerce_record(rec)
    coerced, crs_error = _normalize_crs(coerced)
    norm = s18c.normalize_geometry(coerced, idx)
    norm["geometry_id"] = f"S18D_CUR_GEOM_{idx:04d}"
    if crs_error:
        norm["status_geometria"] = "bloqueada_area_administrativa"
        norm["_status"] = "bloqueada_area_administrativa"
        norm["geometria_de_ocorrencia"] = "false"
        norm["justificativa_tecnica"] = f"{crs_error}; resposta nao usada para overlay"
    missing = [
        field for field in ("candidate_event_id", "data_evento", "tipo_fenomeno", "geometry_type", "geometry_source_type", "crs")
        if not coerced.get(field if field != "geometry_source_type" else "geometry_source_type")
    ]
    if missing and norm["_status"] in s18c.GEOM_STATUS_FORTE:
        norm["status_geometria"] = "bloqueada_sem_data" if "data_evento" in missing else "bloqueada_area_administrativa"
        norm["_status"] = norm["status_geometria"]
        norm["geometria_de_ocorrencia"] = "false"
        norm["justificativa_tecnica"] = f"schema incompleto: {','.join(missing)}; resposta bloqueada"
    norm["_arquivo_origem"] = rec.get("_arquivo_origem", "registro_em_memoria")
    return norm


def process_records(records):
    norms = [normalize_response_record(r, i) for i, r in enumerate(records, start=1)]
    patches = s18c.load_cur_patch_polygons()
    links = []
    counter = 0
    for norm in norms:
        for patch_id, classe, metric in s18c.overlay_geometry(norm, patches):
            counter += 1
            links.append({
                "patch_link_id": f"S18D_LINK_{counter:04d}",
                "candidate_event_id": norm["candidate_event_id"],
                "geometry_id": norm["geometry_id"],
                "geometry_type": norm["geometry_type"],
                "patch_id": patch_id,
                "classe_vinculo": classe,
                "metrica": metric,
                "vinculo_forte": _bool(classe in s18c.CLASSE_VINCULO_FORTE),
                "review_only": "true",
                "justificativa_tecnica": (
                    f"overlay somente revisao entre resposta oficial e patch {patch_id}"
                    if classe in s18c.CLASSE_VINCULO_FORTE else
                    f"candidato tecnico nao forte ({classe})"
                ),
            })
        if norm["_status"] not in s18c.GEOM_STATUS_FORTE:
            counter += 1
            links.append({
                "patch_link_id": f"S18D_LINK_{counter:04d}",
                "candidate_event_id": norm.get("candidate_event_id", "not_available"),
                "geometry_id": "not_available",
                "geometry_type": norm.get("geometry_type", "none"),
                "patch_id": "not_available",
                "classe_vinculo": "insufficient_for_patch_link",
                "metrica": "sem_geometria_de_ocorrencia",
                "vinculo_forte": "false",
                "review_only": "true",
                "justificativa_tecnica": "resposta bloqueada; sem geometria oficial de ocorrencia valida para overlay",
            })
    return norms, links


def matriz_rows(records, norms, links):
    if not records:
        return [{
            "linha_id": "S18D_MAT_0001",
            "arquivo_origem": "not_available",
            "candidate_event_id": "S17C_REF_0060",
            "data_evento": "2022-01-15..2022-01-16",
            "tipo_fenomeno": "flood_inundation_alagamento",
            "geometry_id": "not_available",
            "status_ingestao": "aguardando_resposta",
            "status_geometria": "ausente",
            "patch_links": "0",
            "patch_links_fortes": "0",
            **HARD_DEFAULTS,
            "justificativa_tecnica": "nenhum arquivo oficial recebido; nenhuma geometria foi inferida",
        }]
    by_geom = Counter(l["geometry_id"] for l in links)
    strong_by_geom = Counter(l["geometry_id"] for l in links if l["vinculo_forte"] == "true")
    rows = []
    for idx, norm in enumerate(norms, start=1):
        strong = strong_by_geom.get(norm["geometry_id"], 0)
        status = "resposta_bloqueada"
        if norm["_status"] in s18c.GEOM_STATUS_FORTE and strong:
            status = "resposta_oficial_com_overlay"
        elif norm["_status"] in s18c.GEOM_STATUS_FORTE:
            status = "geometria_resolvida_sem_patch_link"
        rows.append({
            "linha_id": f"S18D_MAT_{idx:04d}",
            "arquivo_origem": norm.get("_arquivo_origem", "not_available"),
            "candidate_event_id": norm.get("candidate_event_id", ""),
            "data_evento": norm.get("data_evento", ""),
            "tipo_fenomeno": norm.get("tipo_fenomeno", ""),
            "geometry_id": norm.get("geometry_id", "not_available"),
            "status_ingestao": status,
            "status_geometria": norm.get("status_geometria", ""),
            "patch_links": str(by_geom.get(norm["geometry_id"], 0)),
            "patch_links_fortes": str(strong),
            **HARD_DEFAULTS,
            "justificativa_tecnica": norm.get("justificativa_tecnica", ""),
        })
    return rows


def status_18d(records, norms, links):
    if not records:
        return "18D_AGUARDANDO_RESPOSTA_OFICIAL"
    if any(l["vinculo_forte"] == "true" for l in links):
        return "18D_RESPOSTA_OFICIAL_INGERIDA_COM_OVERLAY"
    return "18D_RESPOSTA_RECEBIDA_BLOQUEADA"


def status_17b(status18d):
    if status18d == "18D_AGUARDANDO_RESPOSTA_OFICIAL":
        return "17B_BLOQUEADO_POR_GEOMETRIA"
    if status18d == "18D_RESPOSTA_OFICIAL_INGERIDA_COM_OVERLAY":
        return "17B_APROXIMACAO_REGIONAL_COM_REFERENCIAS_PARCIAS"
    return "17B_BLOQUEADO_POR_GEOMETRIA"


def summary_obj(records, norms, links, input_files):
    status18d = status_18d(records, norms, links)
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "input_dir": rel(INPUT_DIR),
        "input_files_count": len(input_files),
        "respostas_lidas": len(records),
        "geometrias_resolvidas": sum(1 for n in norms if n["_status"] in s18c.GEOM_STATUS_FORTE),
        "patch_polys_disponiveis": len(s18c.load_cur_patch_polygons()),
        "patch_links": len(links),
        "strong_patch_links": sum(1 for l in links if l["vinculo_forte"] == "true"),
        "ground_truth_true_count": 0,
        "eligible_for_training_true_count": 0,
        "score_v7_allowed_true_count": 0,
        "benchmark_17b_criado": False,
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
        "status_18d": status18d,
        "status_17b": status_17b(status18d),
        "review_only": True,
        "ground_truth": False,
        "trainable": False,
    }


def preflight_obj(input_files):
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "input_dir": rel(INPUT_DIR),
        "input_dir_exists": INPUT_DIR.exists(),
        "input_files": [rel(p) for p in input_files],
        "patch_polys_disponiveis": len(s18c.load_cur_patch_polygons()),
        "required_18c": [{"path": rel(p), "exists": p.exists()} for p in REQUIRED_18C],
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
    }


def gate_curitiba_rows(summary):
    rows = [
        ("resposta_oficial_recebida", _bool(summary["respostas_lidas"] > 0), "true", summary["respostas_lidas"] > 0, "arquivo oficial no diretorio local de resposta"),
        ("geometria_oficial_resolvida", str(summary["geometrias_resolvidas"]), ">=1", summary["geometrias_resolvidas"] >= 1, "ponto, poligono ou bbox aceito"),
        ("patch_link_forte_criado", str(summary["strong_patch_links"]), ">=1", summary["strong_patch_links"] >= 1, "overlay com patch CUR"),
        ("ground_truth_zero", "0", "0", True, "sem ground truth"),
        ("trainable_zero", "0", "0", True, "sem treino"),
        ("score_v7_zero", "0", "0", True, "sem score_v7"),
        ("score_v6_intacto", _bool(not summary["score_v6_changed"]), "true", not summary["score_v6_changed"], "score_v6 sem alteracao"),
    ]
    return [{"criterio": c, "valor_observado": v, "limiar": t, "passou": _bool(p), "observacao": o} for c, v, t, p, o in rows]


def gate_17b_rows(summary):
    rows = [
        ("segunda_regiao_com_overlay", _bool(summary["strong_patch_links"] > 0), "true", summary["strong_patch_links"] > 0, "Curitiba ainda depende de resposta oficial"),
        ("benchmark_17b_criado", "false", "false", True, "nenhum benchmark criado"),
        ("ground_truth_zero", "0", "0", True, "sem ground truth"),
        ("trainable_zero", "0", "0", True, "sem treino"),
        ("score_v7_zero", "0", "0", True, "sem score_v7"),
    ]
    return [{"criterio": c, "valor_observado": v, "limiar": t, "passou": _bool(p), "observacao": o} for c, v, t, p, o in rows]


def oficio_defesa_civil():
    return """# Oficio tecnico - solicitacao de geometria de ocorrencia em Curitiba

Destinatario sugerido: Defesa Civil Municipal de Curitiba.

Solicita-se, para uso academico e somente revisao, a geometria oficial de ocorrencias de
alagamento, inundacao ou enxurrada associadas ao evento de 15 e 16 de janeiro de 2022
em Curitiba (`CUR_2022_01_15` / `S17C_REF_0060`).

## Dados solicitados

1. Data da ocorrencia ou intervalo curto.
2. Tipo de fenomeno confirmado.
3. Geometria da ocorrencia: ponto, poligono ou bbox.
4. Endereco textual, se houver, apenas como complemento.
5. CRS, precisao espacial, fonte responsavel e metodo de coleta.
6. Confirmacao de autorizacao de uso academico.

Sem coordenadas oficiais, bairro, rua, centroide, area administrativa, alerta e risco
nao serao usados como geometria de ocorrencia.
"""


def oficio_ippuc():
    return """# Oficio tecnico - solicitacao ao GeoCuritiba / IPPUC

Destinatario sugerido: GeoCuritiba / IPPUC.

Solicita-se camada leve ou tabela com geometria oficial de ocorrencia de
alagamento, inundacao ou enxurrada para o evento de 15 e 16 de janeiro de 2022
em Curitiba. Formatos aceitos: GeoJSON, CSV com lat/lon, CSV com bbox, JSON,
WKT, shapefile ou XLSX com coordenadas.

O dado sera usado em analise observacional somente revisao. A ausencia de resposta
mantem o bloqueio por geometria; nenhum valor sera inferido.
"""


def instrucoes_markdown():
    return f"""# Instrucoes de ingestao da resposta oficial Curitiba

Coloque a resposta oficial em:

`{rel(INPUT_DIR)}`

Formatos aceitos: CSV com `lat`/`lon`, CSV com `bbox`, GeoJSON, JSON, WKT e shapefile
quando houver suporte local. Campos minimos: `candidate_event_id`, `data_evento`,
`tipo_fenomeno`, `geometry_type`, `geometry_source` e `crs`.

Depois execute:

`python scripts\\suscetibilidade\\ingest_susc_18d_resposta_oficial_curitiba.py`

O resultado esperado e a normalizacao para EPSG:4326, overlay com os 43 poligonos CUR
e geracao de patch-links. Bairro, rua, centroide, area administrativa, alerta e risco
continuam bloqueados.
"""


def schema_resposta():
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Schema de resposta oficial Curitiba SUSC-18D",
        "type": "object",
        "required": ["candidate_event_id", "data_evento", "tipo_fenomeno", "geometry_type", "geometry_source", "crs"],
        "properties": {
            "candidate_event_id": {"type": "string", "examples": ["S17C_REF_0060"]},
            "data_evento": {"type": "string"},
            "tipo_fenomeno": {"enum": ["flood_inundation_alagamento", "inundacao_alagamento_enxurrada"]},
            "geometry_type": {"enum": ["point", "polygon", "bbox", "footprint", "textual"]},
            "geometry_source": {"type": "string"},
            "crs": {"type": "string", "default": "EPSG:4326"},
            "lon": {"type": "number"},
            "lat": {"type": "number"},
            "bbox": {"type": "string"},
            "is_event_specific": {"type": "boolean"},
            "is_neighborhood_centroid": {"type": "boolean"},
            "is_risk_area_general": {"type": "boolean"},
            "precisao_m": {"type": "number"},
        },
        "input_dir": rel(INPUT_DIR),
        "hard_defaults": HARD_DEFAULTS,
    }


def modelo_resposta_rows():
    return [{
        "candidate_event_id": "S17C_REF_0060",
        "data_evento": "2022-01-15..2022-01-16",
        "tipo_fenomeno": "flood_inundation_alagamento",
        "geometry_type": "point|polygon|bbox",
        "geometry_source": "PREENCHER_fonte_responsavel",
        "crs": "EPSG:4326",
        "lon": "PREENCHER_se_point",
        "lat": "PREENCHER_se_point",
        "bbox": "minlon,minlat,maxlon,maxlat_se_bbox_ou_poligono",
        "is_event_specific": "true",
        "is_neighborhood_centroid": "false",
        "is_risk_area_general": "false",
        "precisao_m": "PREENCHER",
    }]


def checklist_rows():
    return [
        {"ordem": "1", "item": "Validar destinatario institucional", "status": "pendente", "evidencia_esperada": "contato ou protocolo oficial"},
        {"ordem": "2", "item": "Enviar oficio Defesa Civil", "status": "pendente", "evidencia_esperada": "numero de protocolo ou e-mail"},
        {"ordem": "3", "item": "Enviar oficio GeoCuritiba/IPPUC", "status": "pendente", "evidencia_esperada": "numero de protocolo ou e-mail"},
        {"ordem": "4", "item": "Salvar resposta em local_runs", "status": "pendente", "evidencia_esperada": rel(INPUT_DIR)},
        {"ordem": "5", "item": "Executar ingestao 18D", "status": "pendente", "evidencia_esperada": rel(MATRIZ_INGESTAO)},
    ]


def protocolo_rows():
    return [
        {"protocolo_id": "S18D_PROTO_DEFESA_CIVIL", "destinatario": "Defesa Civil Municipal de Curitiba", "status": "nao_enviado", "data_envio": "not_available", "data_retorno": "not_available", "proxima_acao": "enviar oficio e registrar protocolo"},
        {"protocolo_id": "S18D_PROTO_GEOCURITIBA_IPPUC", "destinatario": "GeoCuritiba / IPPUC", "status": "nao_enviado", "data_envio": "not_available", "data_retorno": "not_available", "proxima_acao": "enviar oficio e registrar protocolo"},
    ]


def status_rows(summary):
    return [
        {"item": "pacote_18c_herdado", "status": "presente", "responsavel_sugerido": "projeto", "prazo_sugerido": "concluido", "observacao": rel(OUT_18C)},
        {"item": "resposta_oficial", "status": "recebida" if summary["respostas_lidas"] else "aguardando_resposta", "responsavel_sugerido": "instituicao_resposta", "prazo_sugerido": "externo", "observacao": rel(INPUT_DIR)},
        {"item": "overlay_curitiba", "status": "executado" if summary["strong_patch_links"] else "bloqueado_por_geometria", "responsavel_sugerido": "projeto", "prazo_sugerido": "apos_resposta", "observacao": "43 poligonos CUR disponiveis"},
        {"item": "status_18d", "status": summary["status_18d"], "responsavel_sugerido": "projeto", "prazo_sugerido": "atual", "observacao": summary["status_17b"]},
    ]


def relatorio_ingestao(summary, input_files):
    files = "\n".join(f"- {rel(p)}" for p in input_files) or "- nenhum arquivo recebido"
    return f"""# Relatorio de ingestao SUSC-18D Curitiba

## Entrada

Diretorio monitorado: `{rel(INPUT_DIR)}`

{files}

## Resultado

- Respostas lidas: {summary['respostas_lidas']}
- Geometrias resolvidas: {summary['geometrias_resolvidas']}
- Patch-links: {summary['patch_links']}
- Patch-links fortes: {summary['strong_patch_links']}
- Status 18D: {summary['status_18d']}
- Status 17B: {summary['status_17b']}

## Guardrails

ground_truth=false; eligible_for_training=false; score_v7_allowed=false; review_only=true.
Nenhum benchmark 17B foi criado e nenhum valor ausente foi inferido.
"""


def _required_inputs_ok():
    return all(p.exists() for p in REQUIRED_18C)


def build_all(records=None, input_files=None):
    if not _required_inputs_ok():
        missing = [rel(p) for p in REQUIRED_18C if not p.exists()]
        raise FileNotFoundError("; ".join(missing))
    ensure_dir(OUT_DATA)
    input_files = _input_files() if input_files is None else input_files
    records = load_response_records() if records is None else records
    norms, links = process_records(records)
    matrix = matriz_rows(records, norms, links)
    summary = summary_obj(records, norms, links, input_files)

    write_markdown(OFICIO_DC, oficio_defesa_civil())
    write_markdown(OFICIO_IPPUC, oficio_ippuc())
    write_csv(CHECKLIST, checklist_rows(), CHECKLIST_FIELDS)
    write_csv(PROTOCOLO, protocolo_rows(), PROTOCOLO_FIELDS)
    write_json(SCHEMA_RESPOSTA, schema_resposta())
    write_csv(MODELO_RESPOSTA, modelo_resposta_rows())
    write_markdown(INSTRUCOES, instrucoes_markdown())
    write_csv(STATUS_PROTOCOLO, status_rows(summary), STATUS_FIELDS)
    write_csv(MATRIZ_INGESTAO, matrix, MATRIZ_FIELDS)
    write_csv(GEOMETRIAS, [{k: n.get(k, "") for k in GEOM_FIELDS} for n in norms], GEOM_FIELDS)
    write_csv(VINCULOS, links, VINCULO_FIELDS)
    write_markdown(RELATORIO_INGESTAO, relatorio_ingestao(summary, input_files))
    write_csv(GATE_CUR, gate_curitiba_rows(summary), GATE_FIELDS)
    write_csv(GATE_17B, gate_17b_rows(summary), GATE_FIELDS)
    write_json(SUMMARY, summary)
    write_json(PREFLIGHT, preflight_obj(input_files))
    return summary


def _public_output_paths():
    paths = list(OUT_DATA.glob("*.csv")) + list(OUT_DATA.glob("*.json")) + list(OUT_DATA.glob("*.md"))
    return [p for p in paths if p.exists()]


def public_text_violations_text(text):
    return sorted({m.group(0) for m in PUBLIC_FORBIDDEN_RE.finditer(text)})


def public_text_violations_files(paths=None):
    errors = []
    for path in paths or _public_output_paths():
        hits = public_text_violations_text(path.read_text(encoding="utf-8", errors="ignore"))
        if hits:
            errors.append(f"{rel(path)}:{','.join(hits)}")
    return errors


def validate_links(rows):
    errors = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("patch_link_id", f"link{idx}")
        classe = row.get("classe_vinculo")
        geometry_absent = row.get("geometry_id") in {"", "not_available"}
        patch_absent = row.get("patch_id") in {"", "not_available"}
        if classe in s18c.CLASSE_VINCULO_FORTE and (geometry_absent or patch_absent):
            errors.append(f"{rid}:classe_forte_sem_geometry_ou_patch")
        if row.get("vinculo_forte") == "true" and (geometry_absent or patch_absent):
            errors.append(f"{rid}:vinculo_forte_sem_geometry_ou_patch")
        if row.get("review_only") != "true":
            errors.append(f"{rid}:review_only_false")
    return errors


def validate_matrix(rows):
    errors = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("linha_id", f"linha{idx}")
        for field in ("ground_truth", "eligible_for_training", "score_v7_allowed"):
            if row.get(field) == "true":
                errors.append(f"{rid}:{field}_true_proibido")
        if not row.get("justificativa_tecnica", "").strip():
            errors.append(f"{rid}:justificativa_vazia")
    return errors


def validate():
    summary = build_all()
    errors = []
    errors.extend(validate_links(read_csv(VINCULOS)))
    errors.extend(validate_matrix(read_csv(MATRIZ_INGESTAO)))
    errors.extend(public_text_violations_files())
    if summary["status_18d"] not in STATUS_18D_ALLOWED:
        errors.append(f"status_18d_fora_enum:{summary['status_18d']}")
    if summary["status_17b"] not in STATUS_17B_ALLOWED:
        errors.append(f"status_17b_fora_enum:{summary['status_17b']}")
    if summary["respostas_lidas"] == 0 and summary["strong_patch_links"] != 0:
        errors.append("sem_resposta_com_patch_link_forte")
    if summary["geometrias_resolvidas"] == 0 and summary["strong_patch_links"] != 0:
        errors.append("sem_geometria_resolvida_com_patch_link_forte")
    if summary["benchmark_17b_criado"]:
        errors.append("benchmark_17b_criado_proibido")
    if summary["score_v6_changed"]:
        errors.append("score_v6_changed_forbidden")
    if summary["score_v7_created"]:
        errors.append("score_v7_created_forbidden")
    for field in ("ground_truth_true_count", "eligible_for_training_true_count", "score_v7_allowed_true_count"):
        if summary[field] != 0:
            errors.append(f"{field}_nonzero")
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(
        "18D protocolo externo Curitiba validado: "
        f"respostas={summary['respostas_lidas']} geom={summary['geometrias_resolvidas']} "
        f"patch_links={summary['patch_links']} fortes={summary['strong_patch_links']} "
        f"status18D={summary['status_18d']} status17B={summary['status_17b']}"
    )
    return 0


def run_all():
    summary = build_all()
    print(
        "18D protocolo externo Curitiba gerado: "
        f"respostas={summary['respostas_lidas']} geom={summary['geometrias_resolvidas']} "
        f"patch_links={summary['patch_links']} fortes={summary['strong_patch_links']} "
        f"status18D={summary['status_18d']} status17B={summary['status_17b']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_all())
