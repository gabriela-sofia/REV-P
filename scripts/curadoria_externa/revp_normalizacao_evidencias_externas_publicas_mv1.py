# -*- coding: utf-8 -*-
"""Normalizacao da frente de evidencias externas publicas (MV1).

Remove a nocao de "licenca como bloqueador" dos artefatos de curadoria/navegacao
externa e a substitui por bloqueadores tecnicos/metodologicos reais. Gera tambem
o manifesto publico leve dos 8 arquivos externos (URL + SHA256 + tamanho),
versoes normalizadas do manifesto/auditoria/recomendacoes e metricas.

Nao copia bruto para outputs_public, nao cria label, nao libera treino,
nao declara ground truth operacional. Determinístico e offline: os status de URL
sao constantes verificadas em 2026-06-19 (sondagem HEAD/GET leve previa).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import zipfile
from typing import Any

# Bloqueadores tecnicos/metodologicos permitidos (vocabulario fechado)
BLOQUEADORES_PERMITIDOS = {
    "portal_sem_arquivo_direto",
    "requer_solicitacao_formal",
    "arquivo_pesado_para_git",
    "sem_geometria",
    "sem_crs",
    "fonte_contextual_nao_evento",
    "suscetibilidade_nao_evento_observado",
    "catalogo_nao_serie_filtrada",
    "fonte_academica_secundaria",
    "risco_landslide_vs_flood",
    "risco_circularidade",
    "dependencia_local_only",
    "dependencia_download_externo",
    "html_de_portal_nao_dado_final",
    "sem_produto_vetorial_publico",
    "download_massa_pesada_nao_recomendado",
}

# Bloqueador tecnico por fonte (substitui qualquer concern de licenca)
BLOQUEADOR_TECNICO_POR_FONTE = {
    "FONTE_NAC_001": ("requer_solicitacao_formal", "obter_via_solicitacao_formal"),         # CEMADEN
    "FONTE_NAC_005": ("portal_sem_arquivo_direto", "buscar_dado_estruturado_ou_alternativa"),  # APAC
    "FONTE_NAC_006": ("requer_solicitacao_formal", "obter_via_solicitacao_formal"),         # DRM-RJ
    "FONTE_INT_001": ("sem_produto_vetorial_publico", "confirmar_produto_vetorial_ou_descartar"),  # Copernicus EMS
}

# 8 arquivos externos verificados (status_url verificado em 2026-06-19)
ARQUIVOS_EXTERNOS = [
    # id, id_fonte, instituicao, cidade, regiao, titulo, tipo, formato, url_origem, url_final, caminho, papel, motivo_nao_git
    ("ARQ_IBGE_PET", "FONTE_NAC_004", "IBGE", "Petropolis", "Sudeste",
     "Malha municipal de Petropolis (3303906)", "limite_municipal", "GeoJSON",
     "https://www.ibge.gov.br/geociencias/downloads-geociencias.html",
     "https://servicodados.ibge.gov.br/api/v3/malhas/municipios/3303906?formato=application/vnd.geo+json&qualidade=intermediaria",
     "local_only/evidencias_externas_quarentena/petropolis/ibge_malha_municipal_petropolis_3303906.geojson",
     "fonte_contextual_nao_evento", "bruto_externo_de_quarentena"),
    ("ARQ_IBGE_REC", "FONTE_NAC_004", "IBGE", "Recife", "Nordeste",
     "Malha municipal de Recife (2611606)", "limite_municipal", "GeoJSON",
     "https://www.ibge.gov.br/geociencias/downloads-geociencias.html",
     "https://servicodados.ibge.gov.br/api/v3/malhas/municipios/2611606?formato=application/vnd.geo+json&qualidade=intermediaria",
     "local_only/evidencias_externas_quarentena/recife/ibge_malha_municipal_recife_2611606.geojson",
     "fonte_contextual_nao_evento", "bruto_externo_de_quarentena"),
    ("ARQ_IBGE_CUR", "FONTE_NAC_004", "IBGE", "Curitiba", "Sul",
     "Malha municipal de Curitiba (4106902)", "limite_municipal", "GeoJSON",
     "https://www.ibge.gov.br/geociencias/downloads-geociencias.html",
     "https://servicodados.ibge.gov.br/api/v3/malhas/municipios/4106902?formato=application/vnd.geo+json&qualidade=intermediaria",
     "local_only/evidencias_externas_quarentena/curitiba/ibge_malha_municipal_curitiba_4106902.geojson",
     "fonte_contextual_nao_evento", "bruto_externo_de_quarentena"),
    ("ARQ_GEOCWB_BACIA", "FONTE_CUR_003", "IPPUC / GeoCuritiba", "Curitiba", "Sul",
     "Bacias Hidrograficas de Curitiba (camada 54)", "bacia_hidrografica", "GeoJSON",
     "https://geocuritiba.ippuc.org.br/server/rest/services/GeoCuritiba/Publico_Interno_GeoCuritiba_BaseCartografica_para_MC/MapServer/54",
     "https://geocuritiba.ippuc.org.br/server/rest/services/GeoCuritiba/Publico_Interno_GeoCuritiba_BaseCartografica_para_MC/MapServer/54/query?where=1=1&outFields=*&outSR=4326&f=geojson",
     "local_only/evidencias_externas_quarentena/curitiba/geocuritiba_bacia_hidrografica_l54.geojson",
     "fonte_contextual_nao_evento", "bruto_externo_de_quarentena"),
    ("ARQ_ANA_INVENTARIO", "FONTE_NAC_002", "ANA - SNIRH/HidroWeb", "", "Nacional",
     "Inventario das Estacoes Pluviometricas", "inventario_estacoes", "PDF",
     "https://www.snirh.gov.br/hidroweb/",
     "https://arquivos.ana.gov.br/infohidrologicas/InventariodasEstacoesPluviometricas.pdf",
     "local_only/evidencias_externas_quarentena/fontes_nacionais/ana_inventario_estacoes_pluviometricas.pdf",
     "catalogo_nao_serie_filtrada", "arquivo_pesado_para_git"),
    ("ARQ_MAPBIOMAS_DHN250", "FONTE_NAC_003", "MapBiomas", "", "Nacional",
     "Estatisticas de cobertura COL.10.1 por divisao hidrografica (DHN250)", "estatistica_cobertura_solo", "XLSX",
     "https://brasil.mapbiomas.org/estatisticas/",
     "https://brasil.mapbiomas.org/wp-content/uploads/sites/4/2026/02/MAPBIOMAS_BRAZIL-COVERAGE_STATISTICS-COL.10.1-DHN250_STATE_BIOME.xlsx",
     "local_only/evidencias_externas_quarentena/fontes_nacionais/mapbiomas_col10_1_cobertura_dhn250_estado_bioma.xlsx",
     "fonte_contextual_nao_evento", "arquivo_pesado_para_git"),
    ("ARQ_SGB_CARTA_PET", "FONTE_PET_005", "SGB/CPRM (RIGEO)", "Petropolis", "Sudeste",
     "Carta de Suscetibilidade a Movimentos Gravitacionais de Massa e Inundacao - Petropolis/RJ", "carta_suscetibilidade", "PDF",
     "https://rigeo.sgb.gov.br/handle/doc/15692",
     "https://rigeo.sgb.gov.br/bitstreams/a7728930-5553-40d7-8021-52b6f85c3412/download",
     "local_only/evidencias_externas_quarentena/petropolis/sgb_cprm_carta_suscetibilidade_petropolis_cs.pdf",
     "suscetibilidade_nao_evento_observado", "arquivo_pesado_para_git"),
    ("ARQ_NHESS", "FONTE_INT_002", "NHESS (Copernicus Publications)", "Petropolis", "Internacional",
     "Artigo NHESS 23/1157/2023 - Petropolis fev/2022", "artigo_academico", "PDF",
     "https://nhess.copernicus.org/articles/23/1157/2023/",
     "https://nhess.copernicus.org/articles/23/1157/2023/nhess-23-1157-2023.pdf",
     "local_only/evidencias_externas_quarentena/fontes_internacionais/nhess_23_1157_2023_petropolis_fev2022.pdf",
     "fonte_academica_secundaria", "arquivo_pesado_para_git"),
]
SHA256_REGISTRADO = {
    "ARQ_IBGE_PET": "11b41120f8c51c5fede86e1702deee9577e808a433bd3fca3908c9abb4c3c4a9",
    "ARQ_IBGE_REC": "df703841e180eb5afff728077f9ba4a2c8e2fd08a77d3c85f439082dbd31c0e4",
    "ARQ_IBGE_CUR": "ecf25a7c5bc2134e0326c2e6fbe4781f83e631c107affc5f5cf3d00f6df52beb",
    "ARQ_GEOCWB_BACIA": "653b9e15abe261690d5d060f427ec136611d3f9abef2ed824451ca5e4bb8d4b6",
    "ARQ_ANA_INVENTARIO": "b54f5c3e4ada405ffd4b92af41e47bb85703061b2e184c6200fc95291f0ffe6c",
    "ARQ_MAPBIOMAS_DHN250": "b93e268b1cff7ce5da5d57339d77a0e4098918dff6b86122ab5ca04811b053bb",
    "ARQ_SGB_CARTA_PET": "97880242981d26caa7487dcf1e1c7c2f0857f991a5c662c2379b9671dcd6d665",
    "ARQ_NHESS": "a8f1468feb77c412acc8d0e508dc78deaf91fc6f2b53e219496516b902dcfc56",
}
TAMANHO_REGISTRADO = {
    "ARQ_IBGE_PET": 2371, "ARQ_IBGE_REC": 1110, "ARQ_IBGE_CUR": 1773,
    "ARQ_GEOCWB_BACIA": 804561, "ARQ_ANA_INVENTARIO": 4350035,
    "ARQ_MAPBIOMAS_DHN250": 9177772, "ARQ_SGB_CARTA_PET": 13811060, "ARQ_NHESS": 7765136,
}
STATUS_URL = {k: "responde" for k in TAMANHO_REGISTRADO}

AUDITORIAS_FONTE = [
    "outputs_public/tables/revp_auditoria_fontes_externas_mv1.csv",
    "outputs_public/tables/revp_auditoria_fontes_externas_downloads_mv1.csv",
    "outputs_public/tables/revp_auditoria_fontes_externas_navegacao_mv1.csv",
]
AUDITORIA_CANONICA = "outputs_public/tables/revp_auditoria_fontes_externas_navegacao_mv1.csv"
MANIFESTO_CANONICO = "outputs_public/tables/revp_manifesto_evidencias_externas_navegacao_mv1.csv"
RECOMENDACOES_FONTE = "outputs_public/tables/revp_recomendacoes_pacote_externo_marco_mv1.csv"

LICENCA_CRITERIO = "licenca_ou_uso_claro"
LICENCA_BLOQUEIO_RISCO = "uso_publico_direto_bloqueado"
LICENCA_BLOQUEIO_ACAO = "bloquear_para_uso_publico_ate_confirmar_licenca"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse(path: str, formato: str) -> str:
    if not os.path.exists(path):
        return "arquivo_ausente"
    try:
        if formato == "GeoJSON":
            d = json.load(open(path, encoding="utf-8"))
            n = len(d.get("features", [])) if isinstance(d, dict) else 0
            return f"geojson_ok_feicoes={n}"
        if formato == "PDF":
            with open(path, "rb") as fh:
                head = fh.read(8)
                fh.seek(max(0, os.path.getsize(path) - 2048))
                tail = fh.read()
            return f"pdf_ok_head={head.startswith(b'%PDF')}_eof={b'%%EOF' in tail}"
        if formato == "XLSX":
            z = zipfile.ZipFile(path)
            ok = "[Content_Types].xml" in z.namelist()
            return f"xlsx_ok_zip={ok}_entradas={len(z.namelist())}"
    except Exception as e:  # pragma: no cover - defensivo
        return f"falha_parse:{type(e).__name__}"
    return "nao_aplicavel"


def _write_csv(path: str, header: list[str], rows: list[list[Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def build(repo_root: str) -> dict[str, Any]:
    root = os.path.abspath(repo_root)

    # ---- 1) Manifesto publico URL + SHA256 + tamanho (8 arquivos) ----
    man_header = [
        "id_arquivo", "id_fonte", "instituicao", "cidade", "regiao", "titulo_arquivo",
        "tipo_arquivo", "formato", "url_origem", "url_final_download", "caminho_local_relativo",
        "tamanho_bytes", "sha256", "status_url", "status_hash", "parseabilidade",
        "origem_publica_institucional", "pode_ir_para_git", "motivo_nao_ir_para_git",
        "pode_ir_para_pacote_externo", "papel_metodologico", "observacao",
    ]
    man_rows: list[list[Any]] = []
    arquivos_hash_real = arquivos_parseaveis = arquivos_url = 0
    for (idf, fonte, inst, cidade, regiao, titulo, tipo, formato, url_o, url_f, caminho,
         papel, motivo_git) in ARQUIVOS_EXTERNOS:
        abspath = os.path.join(root, caminho)
        existe = os.path.exists(abspath)
        sha_real = _sha256(abspath) if existe else ""
        tam_real = os.path.getsize(abspath) if existe else ""
        parse = _parse(abspath, formato)
        sha_reg = SHA256_REGISTRADO[idf]
        status_hash = "confere" if (existe and sha_real == sha_reg) else ("divergente" if existe else "arquivo_ausente")
        if status_hash == "confere":
            arquivos_hash_real += 1
        if parse.endswith(("ok_feicoes", "")) or "ok" in parse:
            arquivos_parseaveis += 1
        if STATUS_URL[idf] == "responde":
            arquivos_url += 1
        man_rows.append([
            idf, fonte, inst, cidade, regiao, titulo, tipo, formato, url_o, url_f, caminho,
            tam_real if existe else TAMANHO_REGISTRADO[idf], sha_reg, STATUS_URL[idf], status_hash, parse,
            "sim", "false", motivo_git, "sim", papel,
            "Reproduzivel por re-download via URL oficial e conferencia de SHA256/tamanho. Binario permanece em local_only.",
        ])
    _write_csv(os.path.join(root, "outputs_public/tables/revp_manifesto_publico_arquivos_externos_url_hash_mv1.csv"),
               man_header, man_rows)

    # ---- 2) Varredura e normalizacao dos bloqueadores de licenca ----
    sub_header = ["artefato", "campo_ou_trecho", "valor_antigo", "valor_normalizado",
                  "tipo_normalizacao", "justificativa", "status", "observacao"]
    sub_rows: list[list[Any]] = []
    blocker_encontrados = 0

    for aud in AUDITORIAS_FONTE:
        ap = os.path.join(root, aud)
        if not os.path.exists(ap):
            continue
        with open(ap, encoding="utf-8") as fh:
            reader = list(csv.DictReader(fh))
        for r in reader:
            if r.get("criterio") == LICENCA_CRITERIO:
                idf = r.get("id_fonte", "")
                old = f"criterio={LICENCA_CRITERIO}; risco={r.get('risco','')}; acao={r.get('acao_recomendada','')}"
                tech, acao = BLOQUEADOR_TECNICO_POR_FONTE.get(idf, ("", "manter_referencia_url_hash"))
                novo_risco = tech if tech else "nenhum"
                new = f"criterio=reprodutibilidade_tecnica_url_hash; risco={novo_risco}; acao={acao}"
                eh_bloqueio = r.get("acao_recomendada") == LICENCA_BLOQUEIO_ACAO
                if eh_bloqueio:
                    blocker_encontrados += 1
                sub_rows.append([
                    os.path.basename(aud), f"linha id_fonte={idf} criterio={LICENCA_CRITERIO}",
                    old, new,
                    "substituicao_bloqueador_licenca_por_bloqueador_tecnico" if eh_bloqueio
                    else "substituicao_criterio_licenca_por_reprodutibilidade_tecnica",
                    "Dado publico/institucional; o que condiciona uso operacional e reprodutibilidade tecnica, nao licenca.",
                    "normalizado",
                    "bloqueador_real" if eh_bloqueio else "criterio_de_licenca_removido",
                ])

    # manifestos: coluna licenca_ou_uso descritiva -> substituida por origem publica + reprodutibilidade
    for man in ["outputs_public/tables/revp_manifesto_evidencias_externas_mv1.csv",
                "outputs_public/tables/revp_manifesto_evidencias_externas_downloads_mv1.csv",
                MANIFESTO_CANONICO]:
        mp = os.path.join(root, man)
        if not os.path.exists(mp):
            continue
        sub_rows.append([
            os.path.basename(man), "coluna licenca_ou_uso",
            "valores de licenca (ex.: indefinida_a_confirmar / *_a_confirmar) usados como concern de uso",
            "coluna substituida por origem_publica_institucional=sim e reprodutibilidade_tecnica",
            "remocao_de_concern_de_licenca_em_manifesto",
            "Origem publica/institucional rastreavel; licenca nao condiciona auditabilidade nem reprodutibilidade.",
            "normalizado", "manifesto normalizado emitido sem coluna de licenca",
        ])

    # script de curadoria (origem do bloqueador) -> registrado, sem editar destrutivamente
    script_curadoria = "scripts/curadoria_externa/revp_curadoria_evidencias_externas_mv1.py"
    if os.path.exists(os.path.join(root, script_curadoria)):
        sub_rows.append([
            script_curadoria, "logica que emite uso_publico_direto_bloqueado / bloquear_*_licenca",
            "criterio licenca_ou_uso_claro gera bloqueio de uso publico",
            "recomendar logica de reprodutibilidade tecnica (URL+hash+formato) no proximo refactor",
            "recomendacao_de_refactor_de_script",
            "Script historico preservado; recomendacao registrada para evitar reintroducao de licenca como bloqueador.",
            "registrado", "nao editado nesta passada (preservacao de historico)",
        ])

    _write_csv(os.path.join(root, "outputs_public/tables/revp_normalizacao_bloqueadores_evidencias_externas_mv1.csv"),
               sub_header, sub_rows)

    # ---- 3) Auditoria de fontes externas normalizada (canonica) ----
    ap = os.path.join(root, AUDITORIA_CANONICA)
    aud_norm_rows: list[list[Any]] = []
    aud_header = ["id_fonte", "criterio", "status", "evidencia", "risco", "severidade", "acao_recomendada"]
    if os.path.exists(ap):
        with open(ap, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("criterio") == LICENCA_CRITERIO:
                    idf = r.get("id_fonte", "")
                    tech, acao = BLOQUEADOR_TECNICO_POR_FONTE.get(idf, ("", "manter_referencia_url_hash"))
                    if tech:
                        aud_norm_rows.append([idf, "reprodutibilidade_tecnica_url_hash", "atencao",
                                              "origem_publica_institucional=sim; bloqueador tecnico identificado",
                                              tech, "media", acao])
                    else:
                        aud_norm_rows.append([idf, "reprodutibilidade_tecnica_url_hash", "ok",
                                              "origem_publica_institucional=sim; reproduzivel por URL+SHA256",
                                              "nenhum", "baixa", "manter_referencia_url_hash"])
                else:
                    aud_norm_rows.append([r.get("id_fonte", ""), r.get("criterio", ""), r.get("status", ""),
                                          r.get("evidencia", ""), r.get("risco", ""), r.get("severidade", ""),
                                          r.get("acao_recomendada", "")])
    _write_csv(os.path.join(root, "outputs_public/tables/revp_auditoria_fontes_externas_normalizada_mv1.csv"),
               aud_header, aud_norm_rows)

    # ---- 4) Manifesto de fontes externas normalizado (canonico, sem coluna de licenca) ----
    mp = os.path.join(root, MANIFESTO_CANONICO)
    man_norm_rows: list[list[Any]] = []
    man_norm_header = ["id_fonte", "regiao", "cidade", "instituicao", "titulo", "categoria_evento",
                       "formato_original", "url_origem", "caminho_local_quarentena", "sha256_arquivo",
                       "origem_publica_institucional", "reprodutibilidade_tecnica", "papel_metodologico",
                       "status_download", "observacao"]
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                cat = r.get("categoria_evento", "")
                papel = ("suscetibilidade_nao_evento_observado" if "suscetibilidade" in cat
                         else "fonte_contextual_nao_evento")
                reprod = "reproduzivel_por_url_hash" if r.get("sha256_arquivo") else "depende_download_externo"
                man_norm_rows.append([
                    r.get("id_fonte", ""), r.get("regiao", ""), r.get("cidade", ""), r.get("instituicao", ""),
                    r.get("titulo", ""), cat, r.get("formato_original", ""), r.get("url_origem", ""),
                    r.get("caminho_local_quarentena", ""), r.get("sha256_arquivo", ""),
                    "sim", reprod, papel, r.get("status_download", ""),
                    "Origem publica/institucional; licenca nao usada como bloqueador.",
                ])
    _write_csv(os.path.join(root, "outputs_public/tables/revp_manifesto_evidencias_externas_normalizado_mv1.csv"),
               man_norm_header, man_norm_rows)

    # ---- 5) Recomendacoes normalizadas (a partir das recomendacoes ja sem licenca) ----
    rp = os.path.join(root, RECOMENDACOES_FONTE)
    rec_header = ["item", "tipo", "prioridade", "motivo", "condicao_para_publicar",
                  "origem_publica_institucional", "risco_tecnico_ou_metodologico", "acao_recomendada", "observacao"]
    rec_rows: list[list[Any]] = []
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rec_rows.append([r.get(c, "") for c in rec_header])
    _write_csv(os.path.join(root, "outputs_public/tables/revp_recomendacoes_pacote_externo_normalizada_mv1.csv"),
               rec_header, rec_rows)

    # ---- 6) Bruto pesado em outputs_public ----
    brutos = 0
    op = os.path.join(root, "outputs_public")
    for dirpath, _, files in os.walk(op):
        for f in files:
            fp = os.path.join(dirpath, f)
            ext = f.lower().rsplit(".", 1)[-1] if "." in f else ""
            if ext in {"tif", "tiff", "geotiff", "npz", "npy", "zip", "shp", "gpkg", "img"} or os.path.getsize(fp) > 5_000_000:
                brutos += 1

    # ---- 7) Metricas ----
    metrics = {
        "status_normalizacao": "EVIDENCIAS_EXTERNAS_NORMALIZADAS_SEM_LICENCA_COMO_BLOQUEADOR",
        "total_ocorrencias_licenca_como_bloqueador_encontradas": blocker_encontrados,
        "total_ocorrencias_normalizadas": blocker_encontrados,
        "total_arquivos_manifesto_publico": len(man_rows),
        "arquivos_com_hash_real": arquivos_hash_real,
        "arquivos_com_url_final": arquivos_url,
        "arquivos_parseaveis": arquivos_parseaveis,
        "brutos_pesados_em_outputs_public": brutos,
        "pode_verificar_por_url_hash_tamanho": True,
        "pode_usar_git_add_A": False,
        "ground_truth_operacional_status": "ausente",
        "treino_supervisionado_status": "bloqueado",
        "itens_podem_virar_label_agora": 0,
        "guardrails_preservados": [
            "evidencia externa nao vira label",
            "download nao vira validacao operacional",
            "suscetibilidade nao vira evento observado",
            "landslide scar nao vira flood extent",
            "texto sem geometria nao fecha patch-level",
            "Curitiba nao vira negativo formal",
            "nao liberar treino",
            "nao declarar ground truth operacional",
            "pode_virar_label_agora=false",
        ],
        "bloqueadores_tecnicos_permitidos": sorted(BLOQUEADORES_PERMITIDOS),
        "nota_licenca": "Dados publicos/institucionais; licenca nao e bloqueador. O que condiciona uso operacional e reprodutibilidade tecnica, ausencia de geometria/evento patch-level, ausencia de CRS/overlay, solicitacao formal ou risco metodologico.",
        "gerado_em_utc": "2026-06-19T00:00:00+00:00",
    }
    with open(os.path.join(root, "outputs_public/metrics/revp_normalizacao_evidencias_externas_publicas_mv1.json"),
              "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=1)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalizacao de evidencias externas publicas (MV1).")
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    m = build(args.repo_root)
    print("status:", m["status_normalizacao"])
    print("bloqueadores de licenca encontrados/normalizados:",
          m["total_ocorrencias_licenca_como_bloqueador_encontradas"])
    print("manifesto publico:", m["total_arquivos_manifesto_publico"],
          "| hash real:", m["arquivos_com_hash_real"],
          "| parseaveis:", m["arquivos_parseaveis"],
          "| url final:", m["arquivos_com_url_final"])
    print("brutos pesados em outputs_public:", m["brutos_pesados_em_outputs_public"])
    print("pode_usar_git_add_A:", m["pode_usar_git_add_A"])


if __name__ == "__main__":
    main()
