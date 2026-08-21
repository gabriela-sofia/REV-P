"""Fechamento dos downloads externos (MV1) e integracao com o marco label-free.

Esta passada e *review-only*. Ela NAO promove nenhuma fonte externa a
ground truth operacional, NAO cria label, NAO cria negativo formal e NAO
libera treino supervisionado. O objetivo e:

- tentar baixar diretamente as fontes externas ainda nao baixadas;
- armazenar o bruto apenas em quarentena local (local_only/);
- calcular SHA256 real do arquivo em disco (nunca inventado);
- separar fonte baixada, fonte que exige download manual, fonte sem download
  direto, falha HTTP e bloqueio;
- gerar artefatos revisados (sufixo _downloads_mv1) sem sobrescrever a
  curadoria original;
- integrar o pacote externo revisado com o marco label-free MV1.

Reutiliza o modulo de curadoria original
(revp_curadoria_evidencias_externas_mv1.py) para as regras de auditoria,
eventos e geometrias, garantindo consistencia.

Uso:
    python scripts/curadoria_externa/revp_fechamento_downloads_e_integracao_evidencias_externas_mv1.py
    python scripts/curadoria_externa/revp_fechamento_downloads_e_integracao_evidencias_externas_mv1.py --check
    python scripts/curadoria_externa/revp_fechamento_downloads_e_integracao_evidencias_externas_mv1.py --offline
"""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import ssl
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
ORIG_SCRIPT = ROOT / "scripts" / "curadoria_externa" / "revp_curadoria_evidencias_externas_mv1.py"
QUARENTENA = ROOT / "local_only" / "evidencias_externas_quarentena"

# Importa o modulo de curadoria original (helpers e registro de fontes).
_spec = importlib.util.spec_from_file_location("revp_curadoria_externa_mv1", ORIG_SCRIPT)
assert _spec and _spec.loader
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)


# ---------------------------------------------------------------------------
# Plano de download (fatos observados nesta rodada, atualizaveis ao vivo).
# As fontes ja presentes no repositorio (status original
# "ja_presente_no_repositorio") viram "ja_existia_localmente".
# ---------------------------------------------------------------------------

PLANO_DOWNLOAD: dict[str, dict] = {
    "FONTE_INT_002": {
        "kind": "direct_download",
        "status": "baixado",
        "codigo_http": "200",
        "content_type": "application/pdf",
        "subdir": "fontes_internacionais",
        "filename": "nhess_23_1157_2023_petropolis_fev2022.pdf",
        "download_url": "https://nhess.copernicus.org/articles/23/1157/2023/nhess-23-1157-2023.pdf",
        "motivo": "PDF open-access (CC BY 4.0) baixado diretamente; a URL do artigo (HTML) registrada falhou por TLS, mas o PDF canonico do mesmo artigo respondeu 200.",
    },
    "FONTE_NAC_001": {
        "kind": "portal_manual", "status": "requer_download_manual",
        "codigo_http": "200", "content_type": "text/html", "subdir": "fontes_nacionais",
        "motivo": "CEMADEN Mapa Interativo e aplicacao JavaScript; o download exige interacao humana, nao ha URL de arquivo direto.",
    },
    "FONTE_NAC_002": {
        "kind": "portal_manual", "status": "requer_download_manual",
        "codigo_http": "200", "content_type": "text/html", "subdir": "fontes_nacionais",
        "motivo": "ANA HidroWeb/SNIRH exige busca por estacao e formulario; a URL retorna HTML de portal, nao o dataset.",
    },
    "FONTE_NAC_003": {
        "kind": "portal_manual", "status": "requer_download_manual",
        "codigo_http": "200", "content_type": "text/html", "subdir": "fontes_nacionais",
        "motivo": "MapBiomas exige selecao de colecao/recorte (Toolkit/GEE ou formulario); a URL retorna HTML, nao o arquivo de cobertura.",
    },
    "FONTE_NAC_004": {
        "kind": "portal_manual", "status": "requer_download_manual",
        "codigo_http": "200", "content_type": "text/html", "subdir": "fontes_nacionais",
        "motivo": "IBGE Downloads Geociencias e pagina-indice/FTP; exige navegacao ate o arquivo de malha de setores censitarios.",
    },
    "FONTE_NAC_005": {
        "kind": "portal_manual", "status": "requer_download_manual",
        "codigo_http": "200", "content_type": "text/html", "subdir": "fontes_nacionais",
        "motivo": "APAC retorna portal HTML; os boletins pluviometricos exigem navegacao/selecao de data.",
    },
    "FONTE_NAC_006": {
        "kind": "http_error", "status": "falha_download",
        "codigo_http": "503", "content_type": "", "subdir": "fontes_nacionais",
        "motivo": "DRM-RJ retornou HTTP 503 (Service Unavailable); alem disso nao ha URL de arquivo direto (requer solicitacao formal).",
    },
    "FONTE_INT_001": {
        "kind": "portal_manual", "status": "requer_download_manual",
        "codigo_http": "200", "content_type": "text/html", "subdir": "fontes_internacionais",
        "motivo": "Copernicus EMS Rapid Mapping exige selecao da ativacao (EMSR) e download por produto; o ID exato para Petropolis nao foi confirmado.",
    },
    "FONTE_CUR_003": {
        "kind": "portal_manual", "status": "requer_download_manual",
        "codigo_http": "200", "content_type": "text/html", "subdir": "curitiba",
        "motivo": "Portal IPPUC/GeoCuritiba retorna HTML; o dataset exige navegacao/selecao humana, sem URL de arquivo direto.",
    },
}

DATA_ACESSO_ROUND = "2026-06-18"


# ---------------------------------------------------------------------------
# Download real (idempotente, falha-suave).
# ---------------------------------------------------------------------------

def baixar_fonte_direta(plano: dict, do_network: bool) -> dict:
    """Baixa um arquivo direto para a quarentena se ainda nao existir.

    Retorna dict com status final, codigo_http, content_type, tamanho_bytes,
    caminho_local (relativo a ROOT) e motivo. Nunca inventa conteudo: se o
    download falhar, registra falha_download.
    """
    subdir = plano["subdir"]
    dest_abs = QUARENTENA / subdir / plano["filename"]
    rel = dest_abs.relative_to(ROOT).as_posix()

    if dest_abs.is_file() and dest_abs.stat().st_size > 0:
        return {"status": "baixado", "codigo_http": plano["codigo_http"],
                "content_type": plano["content_type"], "caminho_local": rel,
                "motivo": plano["motivo"]}

    if not do_network:
        return {"status": "falha_download", "codigo_http": "",
                "content_type": "", "caminho_local": "",
                "motivo": "Arquivo ausente na quarentena e execucao offline; download nao tentado."}

    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(plano["download_url"],
                                     headers={"User-Agent": "Mozilla/5.0 (REV-P curadoria externa)"})
        with urllib.request.urlopen(req, timeout=90, context=ctx) as resp:
            data = resp.read()
            ct = resp.headers.get("Content-Type") or plano["content_type"]
            code = str(resp.status)
        dest_abs.parent.mkdir(parents=True, exist_ok=True)
        dest_abs.write_bytes(data)
        return {"status": "baixado", "codigo_http": code, "content_type": ct,
                "caminho_local": rel, "motivo": plano["motivo"]}
    except Exception as e:  # noqa: BLE001 - registrar falha real, sem inventar
        return {"status": "falha_download", "codigo_http": "",
                "content_type": "", "caminho_local": "",
                "motivo": f"Falha tecnica no download direto: {type(e).__name__}: {str(e)[:120]}"}


def content_type_por_extensao(rel_path: str) -> str:
    ext = Path(rel_path).suffix.lower()
    return {
        ".pdf": "application/pdf", ".csv": "text/csv", ".json": "application/json",
        ".geojson": "application/geo+json", ".html": "text/html", ".zip": "application/zip",
        ".kml": "application/vnd.google-earth.kml+xml", ".xml": "application/xml",
    }.get(ext, "application/octet-stream")


# ---------------------------------------------------------------------------
# Construcao das linhas revisadas (reutiliza helpers do modulo original).
# ---------------------------------------------------------------------------

LOG_COLS = [
    "id_fonte", "url_origem", "tentativa_download", "status_download",
    "codigo_http", "content_type", "tamanho_bytes", "caminho_local_quarentena",
    "sha256_arquivo", "motivo_falha_ou_bloqueio", "data_acesso", "observacao",
]


def construir(do_network: bool):
    fontes = copy.deepcopy(M.FONTES)
    manifest, audit, events, geoms, log = [], [], [], [], []

    for f in fontes:
        fid = f["id_fonte"]

        if f["status_download"] == "ja_presente_no_repositorio":
            sha, size = M.sha256_of(f["caminho_local"])
            status_final = "ja_existia_localmente" if size >= 0 else "falha_download"
            log.append({
                "id_fonte": fid, "url_origem": f["url_origem"],
                "tentativa_download": "nao (arquivo ja presente no repositorio)",
                "status_download": status_final, "codigo_http": "",
                "content_type": content_type_por_extensao(f["caminho_local"]),
                "tamanho_bytes": size if size >= 0 else "",
                "caminho_local_quarentena": f["caminho_local"], "sha256_arquivo": sha,
                "motivo_falha_ou_bloqueio": "" if size >= 0 else "Arquivo local esperado nao encontrado.",
                "data_acesso": f["data_acesso"],
                "observacao": "Fonte ja existia localmente (git-ignored); SHA256 recalculado.",
            })
            f["status_download"] = status_final

        elif fid in PLANO_DOWNLOAD:
            plano = PLANO_DOWNLOAD[fid]
            if plano["kind"] == "direct_download":
                res = baixar_fonte_direta(plano, do_network)
                if res["caminho_local"]:
                    f["caminho_local"] = res["caminho_local"]
                f["status_download"] = res["status"]
                sha, size = M.sha256_of(f["caminho_local"]) if res["caminho_local"] else ("", -1)
                log.append({
                    "id_fonte": fid, "url_origem": f["url_origem"],
                    "tentativa_download": f"sim (GET {plano['download_url']})",
                    "status_download": res["status"], "codigo_http": res["codigo_http"],
                    "content_type": res["content_type"],
                    "tamanho_bytes": size if size >= 0 else "",
                    "caminho_local_quarentena": res["caminho_local"], "sha256_arquivo": sha,
                    "motivo_falha_ou_bloqueio": "" if res["status"] == "baixado" else res["motivo"],
                    "data_acesso": DATA_ACESSO_ROUND,
                    "observacao": res["motivo"],
                })
            else:
                f["status_download"] = plano["status"]
                log.append({
                    "id_fonte": fid, "url_origem": f["url_origem"],
                    "tentativa_download": f"sim (probe HTTP {f['url_origem']})",
                    "status_download": plano["status"], "codigo_http": plano["codigo_http"],
                    "content_type": plano["content_type"], "tamanho_bytes": "",
                    "caminho_local_quarentena": "", "sha256_arquivo": "",
                    "motivo_falha_ou_bloqueio": plano["motivo"],
                    "data_acesso": DATA_ACESSO_ROUND,
                    "observacao": plano["motivo"],
                })
        else:
            # Sem plano: preserva como bloqueado sem url.
            f["status_download"] = "bloqueado_sem_url"
            log.append({
                "id_fonte": fid, "url_origem": f["url_origem"],
                "tentativa_download": "nao", "status_download": "bloqueado_sem_url",
                "codigo_http": "", "content_type": "", "tamanho_bytes": "",
                "caminho_local_quarentena": "", "sha256_arquivo": "",
                "motivo_falha_ou_bloqueio": "Sem URL de download direto registrada.",
                "data_acesso": f["data_acesso"], "observacao": "",
            })

        # Manifesto revisado (mesma estrutura do original).
        sha, _ = M.sha256_of(f["caminho_local"])
        manifest.append({
            "id_fonte": fid, "regiao": f["regiao"], "cidade": f["cidade"],
            "estado": f["estado"], "pais": f["pais"], "instituicao": f["instituicao"],
            "familia_fonte": f["familia_fonte"], "tipo_fonte": f["tipo_fonte"],
            "titulo": f["titulo"], "descricao_curta": f["descricao_curta"],
            "url_origem": f["url_origem"], "data_acesso": f["data_acesso"],
            "data_publicacao": f["data_publicacao"],
            "periodo_referencia_inicio": f["periodo_referencia_inicio"],
            "periodo_referencia_fim": f["periodo_referencia_fim"],
            "categoria_evento": f["categoria_evento"], "formato_original": f["formato_original"],
            "caminho_local_quarentena": f["caminho_local"], "sha256_arquivo": sha,
            "licenca_ou_uso": f["licenca_ou_uso"], "status_download": f["status_download"],
            "status_auditoria": f["status_auditoria"], "observacao": f["observacao"],
        })

        audit.extend(M.build_audit_rows(f))

        if f["representa_evento"]:
            janela = "fechada" if f["periodo_referencia_inicio"] and f["periodo_referencia_fim"] else "aberta_ou_indefinida"
            pode = M.decide_pode_label(f, janela)
            events.append({
                "id_evento_externo": "EV_" + fid, "id_fonte": fid, "cidade": f["cidade"],
                "regiao": f["regiao"], "categoria_evento": f["categoria_evento"],
                "data_evento_inicio": f["periodo_referencia_inicio"],
                "data_evento_fim": f["periodo_referencia_fim"], "janela_temporal_status": janela,
                "descricao_evento": f["titulo"],
                "fonte_observacional_independente": "sim" if f["familia_fonte"] in {"agencia_internacional", "orgao_federal", "instituto_pesquisa"} else "parcial",
                "geometria_disponivel": "sim" if f["tem_geometria"] else "nao",
                "geometria_tipo": f["geometria_tipo"], "crs": f["crs"],
                "escala_espacial": "municipal_ou_bairro",
                "pode_sustentar_label_patch_level": "true" if pode else "false",
                "motivo_nao_label": M.motivo_nao_label(f, janela, pode),
                "status_evento": "candidato_para_revisao_futura", "observacao": f["observacao"],
            })

        if f["tem_geometria"]:
            _, motivo = M.avalia_overlay(f)
            sha_g, _ = M.sha256_of(f["caminho_local"])
            geoms.append({
                "id_geometria": "GEO_" + fid, "id_fonte": fid, "cidade": f["cidade"],
                "regiao": f["regiao"], "tipo_geometria": f["geometria_tipo"],
                "formato": f["formato_original"], "crs": f["crs"], "escala": "local",
                "caminho_local_quarentena": f["caminho_local"], "sha256_arquivo": sha_g,
                "representa_evento_observado": "true" if f["representa_evento"] else "false",
                "representa_suscetibilidade": "true" if f["representa_suscet"] else "false",
                "representa_vulnerabilidade": "true" if f["representa_vulner"] else "false",
                "representa_contexto": "true" if f["representa_contexto"] else "false",
                "adequada_para_overlay_patch": "false", "motivo_bloqueio_overlay": motivo,
                "status_geometria": "geometria_candidata_nao_promovida", "observacao": f["observacao"],
            })

    return manifest, audit, events, geoms, log


# ---------------------------------------------------------------------------
# Metricas revisadas e integracao.
# ---------------------------------------------------------------------------

def metricas_revisadas(manifest, events, geoms, log) -> dict:
    by_status = {}
    for r in log:
        by_status[r["status_download"]] = by_status.get(r["status_download"], 0) + 1
    com_sha = [m for m in manifest if m["sha256_arquivo"]]
    return {
        "total_fontes": len(manifest),
        "contagem_por_status_download": by_status,
        "total_baixadas_nesta_rodada": by_status.get("baixado", 0),
        "total_ja_existiam_localmente": by_status.get("ja_existia_localmente", 0),
        "total_requer_download_manual": by_status.get("requer_download_manual", 0),
        "total_bloqueado_sem_download_direto": by_status.get("bloqueado_sem_download_direto", 0),
        "total_falha_download": by_status.get("falha_download", 0),
        "total_com_sha256_real": len(com_sha),
        "total_eventos_candidatos": len(events),
        "total_geometrias_candidatas": len(geoms),
        "guardrails_preservados": GUARDRAILS,
        "proximo_passo_recomendado": PROXIMO_PASSO,
    }


GUARDRAILS = [
    "Sem treino supervisionado; sem label binario; sem positivo formal; sem negativo formal.",
    "Sem ground truth operacional patch-level nesta passada.",
    "unknown nao vira negativo; ausencia de evidencia nao vira classe 0.",
    "Curitiba nao vira negativo formal.",
    "Evidencia externa nao vira label automaticamente.",
    "DINOv2 nao prova inundacao.",
    "Fonte textual sem geometria nao fecha patch-level.",
    "Landslide scar nao prova flood extent.",
    "Geometria de suscetibilidade nao e geometria de evento observado.",
    "Download de fonte nao significa validacao operacional.",
]

PROXIMO_PASSO = (
    "Executar manualmente os downloads marcados como requer_download_manual (CEMADEN, ANA/HidroWeb, "
    "MapBiomas, IBGE, APAC, Copernicus EMS, IPPUC), salvar o bruto na quarentena e re-rodar este script; "
    "e solicitar formalmente as fontes em falha/bloqueio (DRM-RJ). Nenhuma fonte e promovida a label nesta passada."
)

# Artefatos do marco label-free (apenas leitura/deteccao).
MARCO_ARTEFATOS = [
    ROOT / "outputs_public" / "execution_reports" / "revp_fechamento_marco_validacao_label_free_evidencia_estrutural_mv1.md",
    ROOT / "outputs_public" / "tables" / "revp_manifesto_marco_validacao_label_free_evidencia_estrutural_mv1.csv",
    ROOT / "outputs_public" / "metrics" / "revp_fechamento_marco_validacao_label_free_evidencia_estrutural_mv1.json",
    ROOT / "outputs_public" / "tables" / "revp_proximos_passos_pos_marco_label_free_mv1.csv",
]

CURADORIA_ORIGINAL = [
    ROOT / "outputs_public" / "tables" / "revp_manifesto_evidencias_externas_mv1.csv",
    ROOT / "outputs_public" / "tables" / "revp_auditoria_fontes_externas_mv1.csv",
    ROOT / "outputs_public" / "tables" / "revp_indice_eventos_externos_candidatos_mv1.csv",
    ROOT / "outputs_public" / "tables" / "revp_indice_geometrias_externas_candidatas_mv1.csv",
    ROOT / "outputs_public" / "execution_reports" / "revp_curadoria_evidencias_externas_mv1.md",
    ROOT / "outputs_public" / "metrics" / "revp_curadoria_evidencias_externas_mv1.json",
]

INTEGRACAO_COLS = [
    "item", "tipo", "origem", "artefato_relacionado", "status",
    "pode_apoiar_ground_truth_futuro", "pode_virar_label_agora", "bloqueador",
    "acao_recomendada", "observacao_metodologica",
]


def construir_integracao(manifest, geoms):
    rows = []

    # Artefatos do marco label-free.
    for p in MARCO_ARTEFATOS:
        rows.append({
            "item": p.name, "tipo": "artefato_marco_label_free", "origem": "marco_label_free",
            "artefato_relacionado": p.relative_to(ROOT).as_posix(),
            "status": "encontrado" if p.is_file() else "ausente",
            "pode_apoiar_ground_truth_futuro": "sim",
            "pode_virar_label_agora": "false",
            "bloqueador": "marco_metodologico_interno_review_only",
            "acao_recomendada": "manter_como_referencia_metodologica",
            "observacao_metodologica": "Fechamento interno label-free; nao define geometria de evento.",
        })

    # Artefatos da curadoria externa revisada.
    for p in (P_MANIFEST, P_AUDIT, P_EVENTS, P_GEOMS, P_REPORT, P_METRICS, P_LOG):
        rows.append({
            "item": p.name, "tipo": "artefato_curadoria_externa_revisada", "origem": "curadoria_externa_downloads",
            "artefato_relacionado": p.relative_to(ROOT).as_posix(),
            "status": "gerado",
            "pode_apoiar_ground_truth_futuro": "sim",
            "pode_virar_label_agora": "false",
            "bloqueador": "review_only",
            "acao_recomendada": "usar_como_inventario_auditavel",
            "observacao_metodologica": "Pacote externo revisado; nenhuma fonte promovida a label.",
        })

    # Uma linha por fonte externa.
    geom_ids = {g["id_fonte"] for g in geoms}
    for m in manifest:
        tem_geom = m["id_fonte"] in geom_ids
        if tem_geom:
            tipo = "fonte_externa_com_geometria_candidata"
            apoio = "sim"
            bloq = "geometria_candidata_requer_revisao_humana_e_crs_confirmado"
            acao = "encaminhar_geometria_para_revisao_humana"
        elif m["status_download"] in {"baixado", "ja_existia_localmente"}:
            tipo = "fonte_externa_baixada"
            apoio = "sim"
            bloq = "sem_geometria_de_evento_auditavel" if m["sha256_arquivo"] else "sem_arquivo"
            acao = "tratar_como_evidencia_contextual_para_revisao_humana"
        else:
            tipo = "fonte_externa_candidata_pendente"
            apoio = "potencial"
            bloq = m["status_download"]
            acao = "executar_download_manual_ou_solicitacao_formal"
        rows.append({
            "item": m["id_fonte"], "tipo": tipo, "origem": "curadoria_externa_downloads",
            "artefato_relacionado": m["caminho_local_quarentena"] or m["url_origem"],
            "status": m["status_download"],
            "pode_apoiar_ground_truth_futuro": apoio,
            "pode_virar_label_agora": "false",
            "bloqueador": bloq,
            "acao_recomendada": acao,
            "observacao_metodologica": m["observacao"][:240],
        })

    return rows


def metricas_integracao(manifest, events, geoms, metrics_rev) -> dict:
    rev_potenciais = [g for g in geoms]  # geometrias candidatas -> revisao humana
    bloqueados = [m for m in manifest if m["status_download"] in {
        "requer_download_manual", "falha_download", "bloqueado_sem_download_direto", "bloqueado_sem_url"}]
    return {
        "status_integracao": "integrado_review_only",
        "artefatos_marco_encontrados": [p.name for p in MARCO_ARTEFATOS if p.is_file()],
        "artefatos_curadoria_externa_originais_encontrados": [p.name for p in CURADORIA_ORIGINAL if p.is_file()],
        "artefatos_curadoria_externa_revisados_encontrados": [
            p.name for p in (P_MANIFEST, P_AUDIT, P_EVENTS, P_GEOMS, P_REPORT, P_METRICS, P_LOG) if p.is_file()],
        "total_fontes_manifesto_original": _contar_manifesto_original(),
        "total_fontes_baixadas_revisao": metrics_rev["total_baixadas_nesta_rodada"],
        "total_fontes_ja_existiam_localmente": metrics_rev["total_ja_existiam_localmente"],
        "total_fontes_sem_download_direto": metrics_rev["total_requer_download_manual"] + metrics_rev["total_bloqueado_sem_download_direto"],
        "total_fontes_requerem_download_manual": metrics_rev["total_requer_download_manual"],
        "total_fontes_com_sha256_real": metrics_rev["total_com_sha256_real"],
        "total_eventos_candidatos": len(events),
        "total_geometrias_candidatas": len(geoms),
        "itens_potenciais_para_revisao_humana": len(rev_potenciais),
        "itens_bloqueados": len(bloqueados),
        "proximo_passo_recomendado": PROXIMO_PASSO,
        "guardrails_preservados": GUARDRAILS,
    }


def _contar_manifesto_original() -> int:
    p = ROOT / "outputs_public" / "tables" / "revp_manifesto_evidencias_externas_mv1.csv"
    if not p.is_file():
        return 0
    with p.open(encoding="utf-8", newline="") as fh:
        return sum(1 for _ in csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Escrita
# ---------------------------------------------------------------------------

OUT_TABLES = ROOT / "outputs_public" / "tables"
OUT_REPORTS = ROOT / "outputs_public" / "execution_reports"
OUT_METRICS = ROOT / "outputs_public" / "metrics"

P_MANIFEST = OUT_TABLES / "revp_manifesto_evidencias_externas_downloads_mv1.csv"
P_AUDIT = OUT_TABLES / "revp_auditoria_fontes_externas_downloads_mv1.csv"
P_EVENTS = OUT_TABLES / "revp_indice_eventos_externos_candidatos_downloads_mv1.csv"
P_GEOMS = OUT_TABLES / "revp_indice_geometrias_externas_candidatas_downloads_mv1.csv"
P_LOG = OUT_TABLES / "revp_log_downloads_evidencias_externas_mv1.csv"
P_REPORT = OUT_REPORTS / "revp_fechamento_downloads_evidencias_externas_mv1.md"
P_METRICS = OUT_METRICS / "revp_fechamento_downloads_evidencias_externas_mv1.json"

P_INT_TABLE = OUT_TABLES / "revp_integracao_marco_label_free_evidencias_externas_mv1.csv"
P_INT_METRICS = OUT_METRICS / "revp_integracao_marco_label_free_evidencias_externas_mv1.json"
P_INT_REPORT = OUT_REPORTS / "revp_integracao_marco_label_free_evidencias_externas_mv1.md"


def _write_csv(path: Path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _write_report_downloads(manifest, log, metrics):
    L = ["# Fechamento dos downloads de evidencias externas (MV1)\n"]
    L.append("> Passada review-only. Nenhuma fonte foi promovida a label, negativo formal ou ground truth. "
             "Downloads salvos apenas em quarentena local (git-ignored); SHA256 calculado do arquivo real.\n")
    L.append("## Resumo por status de download\n")
    for k, v in metrics["contagem_por_status_download"].items():
        L.append(f"- `{k}`: {v}")
    L.append("")
    L.append("## Fontes baixadas nesta rodada\n")
    baix = [r for r in log if r["status_download"] == "baixado"]
    L.extend(f"- `{r['id_fonte']}` — {r['caminho_local_quarentena']} — SHA256 `{r['sha256_arquivo'][:16]}...` "
             f"({r['tamanho_bytes']} bytes)" for r in baix)
    if not baix:
        L.append("- (nenhuma)")
    L.append("\n## Fontes que ja existiam localmente\n")
    L.extend(f"- `{r['id_fonte']}` — {r['caminho_local_quarentena']} — SHA256 `{r['sha256_arquivo'][:16]}...`"
             for r in log if r["status_download"] == "ja_existia_localmente")
    L.append("\n## Fontes que exigem download manual\n")
    L.extend(f"- `{r['id_fonte']}` — {r['url_origem']} — {r['motivo_falha_ou_bloqueio']}"
             for r in log if r["status_download"] == "requer_download_manual")
    L.append("\n## Falhas de download\n")
    L.extend(f"- `{r['id_fonte']}` — HTTP {r['codigo_http']} — {r['motivo_falha_ou_bloqueio']}"
             for r in log if r["status_download"] == "falha_download")
    if not any(r["status_download"] == "falha_download" for r in log):
        L.append("- (nenhuma)")
    L.append("\n## Guardrails preservados\n")
    L.extend(f"- {g}" for g in GUARDRAILS)
    L.append(f"\n## Proximo passo recomendado\n\n{PROXIMO_PASSO}\n")
    P_REPORT.parent.mkdir(parents=True, exist_ok=True)
    P_REPORT.write_text("\n".join(L), encoding="utf-8")


def _write_report_integracao(manifest, events, geoms, metrics_int):
    contexto = [m for m in manifest if m["id_fonte"] not in {g["id_fonte"] for g in geoms}]
    L = ["# Integracao: marco label-free MV1 + evidencias externas revisadas\n"]
    L.append("> Passada review-only. `pode_virar_label_agora` e sempre `false`. Nenhum evento candidato "
             "e promovido automaticamente a label. Evidencia externa permanece como suporte observacional/contextual.\n")

    L.append("## 1. Artefatos internos do marco label-free\n")
    L.extend(f"- `{p.name}` — {'encontrado' if p.is_file() else 'ausente'}" for p in MARCO_ARTEFATOS)

    L.append("\n## 2. Artefatos externos originais\n")
    L.extend(f"- `{p.name}` — {'encontrado' if p.is_file() else 'ausente'}" for p in CURADORIA_ORIGINAL)

    L.append("\n## 3. Artefatos externos revisados\n")
    L.extend(f"- `{p.name}`" for p in (P_MANIFEST, P_AUDIT, P_EVENTS, P_GEOMS, P_REPORT, P_METRICS, P_LOG))

    sem_download = sum(1 for m in manifest if m["status_download"] in {
        "requer_download_manual", "falha_download", "bloqueado_sem_download_direto", "bloqueado_sem_url"})
    L.append("\n## 4-6. Contagem de downloads\n")
    L.append(f"- Baixadas de fato nesta rodada: {metrics_int['total_fontes_baixadas_revisao']}")
    L.append(f"- Ja existiam localmente: {metrics_int['total_fontes_ja_existiam_localmente']}")
    L.append(f"- Continuam sem download (manual/falha/bloqueio): {sem_download}")

    L.append("\n## 7. Por que cada fonte sem download nao foi baixada\n")
    for m in manifest:
        if m["status_download"] in {"requer_download_manual", "falha_download", "bloqueado_sem_download_direto", "bloqueado_sem_url"}:
            L.append(f"- `{m['id_fonte']}` ({m['status_download']}): {m['observacao'][:200]}")

    L.append("\n## 8. Fontes com SHA256 real\n")
    L.extend(f"- `{m['id_fonte']}` — `{m['sha256_arquivo'][:16]}...`" for m in manifest if m["sha256_arquivo"])

    L.append("\n## 9. Fontes com geometria candidata\n")
    L.extend(f"- `{g['id_fonte']}` ({g['tipo_geometria']}, {g['crs']}) — overlay bloqueado: {g['motivo_bloqueio_overlay']}"
             for g in geoms)
    if not geoms:
        L.append("- (nenhuma)")

    L.append("\n## 10. Fontes apenas como contexto documental\n")
    L.extend(f"- `{m['id_fonte']}` — {m['tipo_fonte']}" for m in contexto)

    L.append("\n## 11. Fontes que podem apoiar revisao humana futura\n")
    L.extend(f"- `{g['id_fonte']}` — geometria candidata para revisao humana" for g in geoms)

    L.append("\n## 12. Fontes bloqueadas para label patch-level\n")
    L.append("- Todas as fontes: nenhuma sustenta label patch-level nesta passada "
             "(pode_virar_label_agora=false para todos os itens da tabela integrada).")

    L.append("\n## 13. Riscos que permanecem\n")
    L.append("- CRS: geometrias candidatas precisam de CRS confirmado e revisado.\n"
             "- Licenca: varias fontes com licenca a confirmar permanecem bloqueadas para uso publico direto.\n"
             "- Circularidade: bases de servico urbano exigem isolamento de fonte independente.\n"
             "- Landslide vs flood: Petropolis/Charter referem-se a deslizamento; nao confundir com flood extent.\n"
             "- Fonte textual sem geometria nao fecha patch-level.\n")

    L.append("## 14. Proximo passo recomendado\n")
    L.append(PROXIMO_PASSO + "\n")

    L.append("## Guardrails preservados\n")
    L.extend(f"- {g}" for g in GUARDRAILS)
    L.append("")

    P_INT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    P_INT_REPORT.write_text("\n".join(L), encoding="utf-8")


def run(do_network: bool = True) -> dict:
    manifest, audit, events, geoms, log = construir(do_network)
    metrics_rev = metricas_revisadas(manifest, events, geoms, log)

    _write_csv(P_MANIFEST, M.MANIFEST_COLS, manifest)
    _write_csv(P_AUDIT, M.AUDIT_COLS, audit)
    _write_csv(P_EVENTS, M.EVENT_COLS, events)
    _write_csv(P_GEOMS, M.GEOM_COLS, geoms)
    _write_csv(P_LOG, LOG_COLS, log)
    P_METRICS.parent.mkdir(parents=True, exist_ok=True)
    P_METRICS.write_text(json.dumps(metrics_rev, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report_downloads(manifest, log, metrics_rev)

    int_rows = construir_integracao(manifest, geoms)
    metrics_int = metricas_integracao(manifest, events, geoms, metrics_rev)
    _write_csv(P_INT_TABLE, INTEGRACAO_COLS, int_rows)
    P_INT_METRICS.write_text(json.dumps(metrics_int, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report_integracao(manifest, events, geoms, metrics_int)

    return {"revisada": metrics_rev, "integracao": metrics_int,
            "gerado_em": datetime.now(timezone.utc).isoformat()}


def check() -> int:
    problems = []
    for path, cols in [(P_MANIFEST, M.MANIFEST_COLS), (P_AUDIT, M.AUDIT_COLS),
                       (P_EVENTS, M.EVENT_COLS), (P_GEOMS, M.GEOM_COLS),
                       (P_LOG, LOG_COLS), (P_INT_TABLE, INTEGRACAO_COLS)]:
        if not path.is_file():
            problems.append(f"CSV ausente: {path}")
            continue
        with path.open(encoding="utf-8", newline="") as fh:
            header = next(csv.reader(fh))
        faltando = [c for c in cols if c not in header]
        if faltando:
            problems.append(f"{path.name}: colunas faltando {faltando}")

    for path in (P_METRICS, P_INT_METRICS):
        if not path.is_file():
            problems.append(f"JSON ausente: {path}")
        else:
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                problems.append(f"JSON invalido {path.name}: {e}")

    for path in (P_REPORT, P_INT_REPORT):
        if not path.is_file():
            problems.append(f"Relatorio ausente: {path}")
        elif "Guardrails preservados" not in path.read_text(encoding="utf-8"):
            problems.append(f"{path.name}: sem secao de guardrails")

    # pode_virar_label_agora sempre false
    if P_INT_TABLE.is_file():
        with P_INT_TABLE.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if r["pode_virar_label_agora"] != "false":
                    problems.append(f"pode_virar_label_agora != false em {r['item']}")
                    break

    if problems:
        print("CHECK: FALHOU")
        for p in problems:
            print("  -", p)
        return 1
    print("CHECK: OK (revisados, log, integracao, JSONs e guardrails validados)")
    return 0


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(check())
    offline = "--offline" in sys.argv
    out = run(do_network=not offline)
    print("Fechamento concluido.")
    print(json.dumps({
        "baixadas_nesta_rodada": out["revisada"]["total_baixadas_nesta_rodada"],
        "ja_existiam_localmente": out["revisada"]["total_ja_existiam_localmente"],
        "requer_download_manual": out["revisada"]["total_requer_download_manual"],
        "falha_download": out["revisada"]["total_falha_download"],
        "com_sha256_real": out["revisada"]["total_com_sha256_real"],
        "eventos_candidatos": out["revisada"]["total_eventos_candidatos"],
        "geometrias_candidatas": out["revisada"]["total_geometrias_candidatas"],
    }, ensure_ascii=False, indent=2))
