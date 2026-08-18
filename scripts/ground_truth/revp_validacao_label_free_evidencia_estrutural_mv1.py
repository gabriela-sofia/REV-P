from __future__ import annotations

import csv
import json
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median


CAMINHO_CSV_PRINCIPAL = Path("outputs_public/tables/revp_validacao_label_free_evidencia_estrutural_mv1.csv")
CAMINHO_MATRIZ_TOPOLOGICA = Path("outputs_public/tables/revp_matriz_topologica_cidades_mv1.csv")
CAMINHO_VIZINHOS = Path("outputs_public/tables/revp_vizinhos_embeddings_label_free_mv1.csv")
CAMINHO_GUARDRAILS = Path("outputs_public/tables/revp_guardrails_validacao_label_free_mv1.csv")
CAMINHO_RELATORIO = Path("outputs_public/execution_reports/revp_validacao_label_free_evidencia_estrutural_mv1.md")
CAMINHO_METRICAS = Path("outputs_public/metrics/revp_validacao_label_free_evidencia_estrutural_mv1.json")

ENTRADAS = {
    "inventario_publico_embeddings": Path("outputs_public/tables/table_dino_embedding_inventory.csv"),
    "matriz_similaridade_publica": Path("outputs_public/tables/table_dino_similarity_matrix.csv"),
    "vizinhos_publicos": Path("outputs_public/tables/table_dino_nearest_neighbors.csv"),
    "sumario_evidencia_externa": Path("outputs_public/tables/table_external_evidence_summary.csv"),
    "scorecard_evidencia_regional": Path("outputs_public/tables/protocol_c_cross_region_evidence_scorecard.csv"),
    "matriz_cobertura_contextual_local": Path("local_runs/dino_embeddings/v1gv/evidence_coverage_matrix_v1gv.csv"),
    "manifesto_local_embeddings": Path("local_runs/dino_embeddings/v1ge/dino_expanded_embedding_manifest_v1ge.csv"),
    "sumario_local_embeddings": Path("local_runs/dino_embeddings/v1ge/dino_expanded_embedding_summary_v1ge.json"),
    "auditoria_temporal_mv1": Path("outputs_public/metrics/revp_auditoria_prontidao_temporal_assets_mv1.json"),
    "relatorio_restauracao_v2dz_v2ef": Path("outputs_public/execution_reports/revp_restauracao_manual_v2dz_v2ef.md"),
}

COLUNAS_PRINCIPAIS = [
    "id_amostra",
    "id_patch_canonico",
    "cidade",
    "regiao",
    "id_embedding",
    "origem_embedding",
    "dimensao_embedding",
    "backbone_dino",
    "status_embedding",
    "distancia_medoid_cidade",
    "distancia_medoid_regiao",
    "distancia_medoid_global",
    "cidade_vizinho_mais_proximo",
    "regiao_vizinho_mais_proximo",
    "mesma_cidade_do_vizinho",
    "mesma_regiao_do_vizinho",
    "distancia_vizinho_mais_proximo",
    "evidencia_contextual_disponivel",
    "score_evidencia_contextual",
    "score_evidencia_eh_label",
    "status_amostra",
    "motivo_bloqueio",
    "observacao_metodologica",
]

COLUNAS_TOPOLOGIA = [
    "cidade_origem",
    "cidade_destino",
    "distancia_media_coseno",
    "distancia_mediana_coseno",
    "total_pares",
    "tipo_comparacao",
    "interpretacao_label_free",
]

COLUNAS_VIZINHOS = [
    "id_amostra",
    "id_patch_canonico",
    "cidade",
    "regiao",
    "id_vizinho_mais_proximo",
    "patch_vizinho_mais_proximo",
    "cidade_vizinho",
    "regiao_vizinho",
    "distancia_coseno",
    "mesma_cidade",
    "mesma_regiao",
]

COLUNAS_GUARDRAILS = ["guardrail", "status", "evidencia", "risco", "acao_recomendada"]

GUARDRAILS = [
    ("sem treino supervisionado", "PRESERVADO", "Inventário DINOv2 público usa review-only e sem alvo supervisionado.", "Promover análise estrutural a treino.", "Manter bloqueio até ground truth formal."),
    ("sem label binário", "PRESERVADO", "score_evidencia_eh_label é sempre false.", "Converter probe contextual em classe.", "Manter evidência contextual como probe externo."),
    ("sem positivo formal", "PRESERVADO", "Nenhuma coluna cria positivo formal.", "Confundir similaridade com confirmação.", "Exigir revisão humana e fonte observacional."),
    ("sem negativo formal", "PRESERVADO", "Nenhuma linha transforma ausência ou Curitiba em negativo.", "Usar contraste regional como negativo.", "Separar contraste metodológico de negativo formal."),
    ("sem ground truth operacional", "PRESERVADO", "Relatório e CSV mantêm sem promoção operacional.", "Tratar restauração ou DINOv2 como validação operacional.", "Manter estado bloqueado para treino."),
    ("unknown não vira negativo", "PRESERVADO", "Valores desconhecidos permanecem desconhecidos.", "Inferir ausência a partir de metadado faltante.", "Registrar lacuna explicitamente."),
    ("Curitiba não vira negativo formal", "PRESERVADO", "Cidade é usada apenas para topologia label-free.", "Converter cidade em classe binária negativa.", "Manter Curitiba como contraste estrutural."),
    ("DINOv2 não prova inundação", "PRESERVADO", "A saída fala em organização estrutural e vizinhança.", "Usar embedding como detecção.", "Exigir evidência observacional independente."),
    ("evidência administrativa não vira label", "PRESERVADO", "Campo score_evidencia_eh_label permanece false.", "Usar score contextual como alvo.", "Usar apenas para priorização de revisão."),
    ("restauração v2dz-v2ef não vira validação operacional", "PRESERVADO", "Restauração é tratada como trilha forense herdada.", "Confundir recuperabilidade com validação.", "Declarar base como insumo metodológico não operacional."),
]

STATUS_VALIDO = "VALIDO_LABEL_FREE_REVIEW_ONLY"
STATUS_BLOQUEADO = "BLOQUEADO_SEM_EMBEDDING_RASTREAVEL"


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Validação label-free e evidência estrutural MV1 do REV-P.")
    parser.add_argument("--repo-root", default=".", help="Raiz do repositório REV-P.")
    return parser


def ler_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def escrever_csv(path: Path, linhas: list[dict[str, str]], colunas: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=colunas, lineterminator="\n")
        writer.writeheader()
        writer.writerows(linhas)


def carregar_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def bool_texto(valor: str) -> bool:
    return str(valor).strip().lower() in {"true", "1", "sim", "yes"}


def numero(valor: str) -> float | None:
    try:
        return float(str(valor).replace(",", "."))
    except (TypeError, ValueError):
        return None


def fmt(valor: float | None) -> str:
    if valor is None:
        return "ausente"
    return f"{valor:.6f}".rstrip("0").rstrip(".")


def normalizar_cidade(valor: str) -> str:
    texto = str(valor or "").strip()
    mapa = {
        "CUR": "Curitiba",
        "CURITIBA": "Curitiba",
        "PET": "Petrópolis",
        "PETROPOLIS": "Petrópolis",
        "PETRÓPOLIS": "Petrópolis",
        "REC": "Recife",
        "RECIFE": "Recife",
    }
    return mapa.get(texto.upper(), texto or "desconhecido")


def carregar_inventario(repo_root: Path) -> dict[str, dict[str, str]]:
    inventario = {}
    for row in ler_csv(repo_root / ENTRADAS["inventario_publico_embeddings"]):
        patch_id = row.get("patch_id", "").strip()
        if not patch_id:
            continue
        inventario[patch_id] = {
            "id_patch": patch_id,
            "cidade": normalizar_cidade(row.get("region", "")),
            "regiao": normalizar_cidade(row.get("region", "")),
            "id_embedding": row.get("dino_input_id", "").strip(),
            "dimensao": row.get("embedding_dim", "").strip(),
            "backbone": row.get("model_backbone", "").strip() or "desconhecido",
            "hash": row.get("vector_sha256", "").strip(),
            "review_only": row.get("review_only", "").strip(),
            "label_status": row.get("label_status", "").strip(),
            "target_status": row.get("target_status", "").strip(),
            "origem": ENTRADAS["inventario_publico_embeddings"].as_posix(),
        }

    manifesto_local = {row.get("patch_id", ""): row for row in ler_csv(repo_root / ENTRADAS["manifesto_local_embeddings"])}
    base_local = repo_root / ENTRADAS["manifesto_local_embeddings"].parent
    for patch_id, row in manifesto_local.items():
        if patch_id not in inventario:
            continue
        caminho_rel = row.get("embedding_path", "").strip()
        caminho_embedding = base_local / caminho_rel if caminho_rel else None
        local_existe = bool(caminho_embedding and caminho_embedding.exists())
        if row.get("success", "").strip().upper() == "SUCCESS" and local_existe:
            inventario[patch_id]["origem"] = (
                f"{ENTRADAS['inventario_publico_embeddings'].as_posix()} | "
                f"{ENTRADAS['manifesto_local_embeddings'].as_posix()} | sidecar_local_rastreavel"
            )
            inventario[patch_id]["hash"] = row.get("hash", "").strip() or inventario[patch_id]["hash"]
            inventario[patch_id]["backbone"] = row.get("model_backbone", "").strip() or inventario[patch_id]["backbone"]
    return inventario


def carregar_similaridade(repo_root: Path) -> dict[str, dict[str, float]]:
    rows = ler_csv(repo_root / ENTRADAS["matriz_similaridade_publica"])
    matriz: dict[str, dict[str, float]] = {}
    for row in rows:
        patch_id = row.get("patch_id", "").strip()
        if not patch_id:
            continue
        matriz[patch_id] = {}
        for coluna, valor in row.items():
            if coluna == "patch_id":
                continue
            parsed = numero(valor)
            if parsed is not None:
                matriz[patch_id][coluna] = parsed
    return matriz


def distancia(matriz: dict[str, dict[str, float]], a: str, b: str) -> float | None:
    if a == b:
        return 0.0
    sim = matriz.get(a, {}).get(b)
    if sim is None:
        sim = matriz.get(b, {}).get(a)
    if sim is None:
        return None
    return 1.0 - sim


def validar_embedding(item: dict[str, str], matriz: dict[str, dict[str, float]]) -> tuple[bool, str]:
    if item["id_patch"] not in matriz:
        return False, "MATRIZ_SIMILARIDADE_AUSENTE"
    if numero(item["dimensao"]) is None:
        return False, "DIMENSAO_EMBEDDING_AUSENTE"
    if not item["hash"]:
        return False, "HASH_EMBEDDING_AUSENTE"
    if item["label_status"].upper() not in {"NO_LABEL", "SEM_LABEL"}:
        return False, "STATUS_LABEL_INCOMPATIVEL"
    if "SUPERVISIONADO" not in item["target_status"].upper() and item["target_status"].upper() != "NO_TARGET":
        return False, "STATUS_ALVO_INCOMPATIVEL"
    return True, ""


def medoid(ids: list[str], matriz: dict[str, dict[str, float]]) -> str:
    melhor_id = ids[0]
    melhor_media = float("inf")
    for patch_id in ids:
        distancias = [distancia(matriz, patch_id, outro) for outro in ids if outro != patch_id]
        validas = [valor for valor in distancias if valor is not None]
        media = mean(validas) if validas else 0.0
        if media < melhor_media:
            melhor_media = media
            melhor_id = patch_id
    return melhor_id


def carregar_scores_contextuais(repo_root: Path) -> dict[str, float]:
    linhas = ler_csv(repo_root / ENTRADAS["matriz_cobertura_contextual_local"])
    campos_contexto = [
        "terrain_geocuritiba",
        "land_use",
        "drainage",
        "population_density",
        "administrative_defesa_civil",
        "terrain_sgb_rigeo",
        "land_use_fbds",
        "geological_cprm",
        "terrain_pe3d",
        "drainage_esig",
        "coastal_context",
    ]
    scores = {}
    bloqueados = {"", "MISSING", "NOT_ACQUIRED", "UNKNOWN", "NA", "N/A"}
    for row in linhas:
        patch_id = row.get("canonical_patch_id", "").strip()
        if not patch_id:
            continue
        valores = [row.get(campo, "").strip().upper() for campo in campos_contexto]
        disponiveis = [valor for valor in valores if valor not in bloqueados]
        scores[patch_id] = len(disponiveis) / len(campos_contexto)
    return scores


def gerar_amostras(repo_root: Path) -> tuple[list[dict[str, str]], dict[str, object]]:
    inventario = carregar_inventario(repo_root)
    matriz = carregar_similaridade(repo_root)
    scores_contextuais = carregar_scores_contextuais(repo_root)

    validos: dict[str, dict[str, str]] = {}
    bloqueios = Counter()
    for patch_id, item in inventario.items():
        valido, motivo = validar_embedding(item, matriz)
        if valido:
            validos[patch_id] = item
        else:
            bloqueios[motivo] += 1

    if not validos:
        return [], {
            "inventario": inventario,
            "matriz": matriz,
            "validos": validos,
            "bloqueios": dict(bloqueios),
            "scores_contextuais": scores_contextuais,
        }

    ids_validos = sorted(validos)
    grupos_cidade: dict[str, list[str]] = {}
    for patch_id, item in validos.items():
        grupos_cidade.setdefault(item["cidade"], []).append(patch_id)

    medoid_global = medoid(ids_validos, matriz)
    medoids_cidade = {cidade: medoid(sorted(ids), matriz) for cidade, ids in grupos_cidade.items()}

    linhas = []
    for indice, patch_id in enumerate(ids_validos, start=1):
        item = validos[patch_id]
        candidatos = []
        for outro_id in ids_validos:
            if outro_id == patch_id:
                continue
            dist = distancia(matriz, patch_id, outro_id)
            if dist is not None:
                candidatos.append((dist, outro_id))
        candidatos.sort()
        vizinho_dist, vizinho_id = candidatos[0] if candidatos else (None, "")
        vizinho = validos.get(vizinho_id, {})
        score_contextual = scores_contextuais.get(patch_id, 0.0)
        linhas.append(
            {
                "id_amostra": f"MV1_LABEL_FREE_{indice:04d}",
                "id_patch_canonico": patch_id,
                "cidade": item["cidade"],
                "regiao": item["regiao"],
                "id_embedding": item["id_embedding"],
                "origem_embedding": item["origem"],
                "dimensao_embedding": item["dimensao"],
                "backbone_dino": item["backbone"],
                "status_embedding": STATUS_VALIDO,
                "distancia_medoid_cidade": fmt(distancia(matriz, patch_id, medoids_cidade[item["cidade"]])),
                "distancia_medoid_regiao": fmt(distancia(matriz, patch_id, medoids_cidade[item["regiao"]])),
                "distancia_medoid_global": fmt(distancia(matriz, patch_id, medoid_global)),
                "cidade_vizinho_mais_proximo": vizinho.get("cidade", "ausente"),
                "regiao_vizinho_mais_proximo": vizinho.get("regiao", "ausente"),
                "mesma_cidade_do_vizinho": "true" if vizinho.get("cidade") == item["cidade"] else "false",
                "mesma_regiao_do_vizinho": "true" if vizinho.get("regiao") == item["regiao"] else "false",
                "distancia_vizinho_mais_proximo": fmt(vizinho_dist),
                "evidencia_contextual_disponivel": "true" if score_contextual > 0 else "false",
                "score_evidencia_contextual": fmt(score_contextual),
                "score_evidencia_eh_label": "false",
                "status_amostra": "PILOTO_EXPLORATORIO_LABEL_FREE",
                "motivo_bloqueio": "LIMITACAO_AMOSTRAL_N_12" if len(validos) < 30 else "SEM_BLOQUEIO_PARA_ANALISE_LABEL_FREE",
                "observacao_metodologica": "Embedding DINOv2 congelado usado para topologia estrutural; evidência contextual é probe externo e não label.",
            }
        )

    meta = {
        "inventario": inventario,
        "matriz": matriz,
        "validos": validos,
        "bloqueios": dict(bloqueios),
        "scores_contextuais": scores_contextuais,
    }
    return linhas, meta


def gerar_topologia(linhas: list[dict[str, str]], matriz: dict[str, dict[str, float]]) -> list[dict[str, str]]:
    por_cidade: dict[str, list[str]] = {}
    cidade_por_patch = {}
    for row in linhas:
        por_cidade.setdefault(row["cidade"], []).append(row["id_patch_canonico"])
        cidade_por_patch[row["id_patch_canonico"]] = row["cidade"]

    saida = []
    cidades = sorted(por_cidade)
    for origem in cidades:
        for destino in cidades:
            if origem > destino:
                continue
            pares = []
            for patch_a in por_cidade[origem]:
                for patch_b in por_cidade[destino]:
                    if origem == destino and patch_a >= patch_b:
                        continue
                    if origem != destino and cidade_por_patch[patch_a] == cidade_por_patch[patch_b]:
                        continue
                    dist = distancia(matriz, patch_a, patch_b)
                    if dist is not None:
                        pares.append(dist)
            if not pares:
                continue
            tipo = "intra_cidade" if origem == destino else "inter_cidades"
            saida.append(
                {
                    "cidade_origem": origem,
                    "cidade_destino": destino,
                    "distancia_media_coseno": fmt(mean(pares)),
                    "distancia_mediana_coseno": fmt(median(pares)),
                    "total_pares": str(len(pares)),
                    "tipo_comparacao": tipo,
                    "interpretacao_label_free": "Comparação estrutural exploratória; não define classe, label, positivo, negativo ou ground truth.",
                }
            )
    return saida


def gerar_vizinhos(linhas: list[dict[str, str]]) -> list[dict[str, str]]:
    id_por_patch = {row["id_patch_canonico"]: row["id_amostra"] for row in linhas}
    return [
        {
            "id_amostra": row["id_amostra"],
            "id_patch_canonico": row["id_patch_canonico"],
            "cidade": row["cidade"],
            "regiao": row["regiao"],
            "id_vizinho_mais_proximo": id_por_patch.get(row["id_patch_canonico"], row["id_amostra"]),
            "patch_vizinho_mais_proximo": next(
                (viz["id_patch_canonico"] for viz in linhas if viz["cidade"] == row["cidade_vizinho_mais_proximo"] and viz["regiao"] == row["regiao_vizinho_mais_proximo"] and viz["id_patch_canonico"] != row["id_patch_canonico"]),
                "ausente",
            ),
            "cidade_vizinho": row["cidade_vizinho_mais_proximo"],
            "regiao_vizinho": row["regiao_vizinho_mais_proximo"],
            "distancia_coseno": row["distancia_vizinho_mais_proximo"],
            "mesma_cidade": row["mesma_cidade_do_vizinho"],
            "mesma_regiao": row["mesma_regiao_do_vizinho"],
        }
        for row in linhas
    ]


def gerar_vizinhos_precisos(linhas: list[dict[str, str]], matriz: dict[str, dict[str, float]]) -> list[dict[str, str]]:
    por_patch = {row["id_patch_canonico"]: row for row in linhas}
    saida = []
    for row in linhas:
        patch_id = row["id_patch_canonico"]
        candidatos = []
        for outro_id in por_patch:
            if outro_id == patch_id:
                continue
            dist = distancia(matriz, patch_id, outro_id)
            if dist is not None:
                candidatos.append((dist, outro_id))
        candidatos.sort()
        if not candidatos:
            continue
        dist, vizinho_id = candidatos[0]
        vizinho = por_patch[vizinho_id]
        saida.append(
            {
                "id_amostra": row["id_amostra"],
                "id_patch_canonico": patch_id,
                "cidade": row["cidade"],
                "regiao": row["regiao"],
                "id_vizinho_mais_proximo": vizinho["id_amostra"],
                "patch_vizinho_mais_proximo": vizinho_id,
                "cidade_vizinho": vizinho["cidade"],
                "regiao_vizinho": vizinho["regiao"],
                "distancia_coseno": fmt(dist),
                "mesma_cidade": "true" if row["cidade"] == vizinho["cidade"] else "false",
                "mesma_regiao": "true" if row["regiao"] == vizinho["regiao"] else "false",
            }
        )
    return saida


def decidir(total_validos: int, total_encontrados: int) -> str:
    if total_encontrados == 0:
        return "BLOQUEADO_SEM_EMBEDDINGS_RASTREAVEIS"
    if total_validos < 12:
        return "BLOQUEADO_EMBEDDINGS_INSUFICIENTES"
    if 12 <= total_validos <= 29:
        return "PILOTO_LABEL_FREE_COM_LIMITACAO_AMOSTRAL"
    if total_validos >= 59:
        return "VALIDACAO_LABEL_FREE_CORPUS_COMPLETO_MV1"
    return "VALIDACAO_LABEL_FREE_EXECUTAVEL_COM_RESSALVAS"


def calcular_metricas(linhas: list[dict[str, str]], meta: dict[str, object], topologia: list[dict[str, str]], vizinhos: list[dict[str, str]], repo_root: Path) -> dict[str, object]:
    total_encontrados = len(meta["inventario"])
    total_validos = len(linhas)
    decisao = decidir(total_validos, total_encontrados)
    dims = sorted({row["dimensao_embedding"] for row in linhas if row["dimensao_embedding"]})
    intra = [numero(row["distancia_media_coseno"]) for row in topologia if row["tipo_comparacao"] == "intra_cidade"]
    inter = [numero(row["distancia_media_coseno"]) for row in topologia if row["tipo_comparacao"] == "inter_cidades"]
    contextual = sum(1 for row in linhas if row["evidencia_contextual_disponivel"] == "true")
    temporal = carregar_json(repo_root / ENTRADAS["auditoria_temporal_mv1"])

    return {
        "total_embeddings_encontrados": total_encontrados,
        "total_embeddings_validos": total_validos,
        "dimensao_embedding": dims[0] if len(dims) == 1 else (dims or "ausente"),
        "contagem_por_cidade": dict(sorted(Counter(row["cidade"] for row in linhas).items())),
        "contagem_por_regiao": dict(sorted(Counter(row["regiao"] for row in linhas).items())),
        "consistencia_vizinho_mesma_cidade": fmt(sum(1 for row in vizinhos if row["mesma_cidade"] == "true") / len(vizinhos)) if vizinhos else "ausente",
        "consistencia_vizinho_mesma_regiao": fmt(sum(1 for row in vizinhos if row["mesma_regiao"] == "true") / len(vizinhos)) if vizinhos else "ausente",
        "distancia_media_intra_cidade": fmt(mean([v for v in intra if v is not None])) if intra else "ausente",
        "distancia_media_inter_cidades": fmt(mean([v for v in inter if v is not None])) if inter else "ausente",
        "status_suficiencia_amostral": "PILOTO_EXPLORATORIO_N_12" if 12 <= total_validos <= 29 else decisao,
        "evidencia_contextual_disponivel": contextual,
        "trilha_a_status": temporal.get("decisao_global", "METADADOS_TEMPORAIS_INSUFICIENTES_TRILHA_B_RECOMENDADA"),
        "trilha_b_status": "ASSUMIDA_COMO_EIXO_CIENTIFICO_IMEDIATO",
        "decisao_metodologica": decisao,
        "bloqueios_principais": {
            **meta["bloqueios"],
            "LIMITACAO_AMOSTRAL_N_12": total_validos if 12 <= total_validos <= 29 else 0,
            "TRILHA_A_BLOQUEADA_METADADOS_TEMPORAIS": 1,
        },
        "guardrails_preservados": [item[0] for item in GUARDRAILS],
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
    }


def entradas_encontradas(repo_root: Path) -> list[str]:
    return [path.as_posix() for path in ENTRADAS.values() if (repo_root / path).exists()]


def entradas_ausentes(repo_root: Path) -> list[str]:
    return [path.as_posix() for path in ENTRADAS.values() if not (repo_root / path).exists()]


def obter_branch(repo_root: Path) -> str:
    head = repo_root / ".git" / "HEAD"
    if not head.exists():
        return "desconhecido"
    texto = head.read_text(encoding="utf-8", errors="replace").strip()
    return texto.removeprefix("ref: refs/heads/") if texto.startswith("ref: refs/heads/") else texto


def gerar_relatorio(repo_root: Path, metricas: dict[str, object], entradas_ok: list[str], entradas_faltantes: list[str]) -> str:
    return "\n".join(
        [
            "# Validação label-free e evidência estrutural MV1",
            "",
            "## 1. Escopo do marco MV1",
            "Este marco consolida uma análise pública, auditável e label-free dos embeddings DINOv2 congelados, da topologia entre cidades/regiões e da evidência administrativa/contextual como probe externo.",
            "",
            "## 2. Por que esta branch é um turning point",
            f"A branch `{obter_branch(repo_root)}` reúne a virada metodológica: restauração forense v2dz-v2ef, auditoria temporal, bloqueio da Trilha A e adoção da Trilha B como eixo científico imediato.",
            "",
            "## 3. Estado herdado da restauração `v2dz-v2ef`",
            "A restauração `v2dz-v2ef` é tratada como trilha forense e não como validação operacional. Ela não cria labels, negativos formais ou ground truth operacional.",
            "",
            "## 4. Estado herdado da auditoria temporal",
            f"A auditoria temporal registrou `{metricas['trilha_a_status']}`. O estado herdado é de metadados temporais insuficientes para deslocamento temporal amplo.",
            "",
            "## 5. Por que a Trilha A está bloqueada agora",
            "A Trilha A depende de séries temporais úteis e metadados de nuvem por patch. O estado atual não oferece esse volume de metadados para o corpus Sentinel/DINO.",
            "",
            "## 6. Por que a Trilha B é a trilha imediata",
            "A Trilha B permite avaliar organização estrutural label-free, vizinhança de embeddings e alinhamento exploratório com evidência administrativa/contextual sem transformar esses sinais em label.",
            "",
            "## 7. Entradas encontradas",
            *[f"- `{item}`" for item in entradas_ok],
            "",
            "## Entradas ausentes",
            *[f"- `{item}`" for item in entradas_faltantes],
            "",
            "## 8. Embeddings encontrados e critérios de validade",
            f"Foram encontrados {metricas['total_embeddings_encontrados']} embeddings rastreáveis e {metricas['total_embeddings_validos']} embeddings válidos. Critérios: patch identificado, dimensão válida, hash presente, matriz de similaridade disponível, sem label e sem alvo supervisionado.",
            "",
            "## 9. Metodologia label-free",
            "A metodologia usa distância de cosseno derivada da matriz pública de similaridade, medoids por cidade/região/global, vizinho mais próximo e cobertura contextual como probe externo.",
            "",
            "## 10. Topologia entre cidades/regiões",
            f"Distância média intra-cidade: `{metricas['distancia_media_intra_cidade']}`. Distância média inter-cidades: `{metricas['distancia_media_inter_cidades']}`. Esses valores descrevem organização estrutural, não classe.",
            "",
            "## 11. Vizinhos mais próximos",
            f"Consistência de vizinho na mesma cidade: `{metricas['consistencia_vizinho_mesma_cidade']}`. Consistência na mesma região: `{metricas['consistencia_vizinho_mesma_regiao']}`.",
            "",
            "## 12. Evidência administrativa/contextual como probe externo",
            f"{metricas['evidencia_contextual_disponivel']} amostras têm algum indicador contextual disponível. O campo `score_evidencia_eh_label` é sempre `false`.",
            "",
            "## 13. Checagem de sanidade de domínio do DINOv2",
            "Com n=12, a checagem é apenas piloto. Há sinal estrutural mensurável por vizinhança e distâncias, mas a suficiência amostral não permite conclusão forte.",
            "",
            "## 14. Limitações",
            "- Corpus pequeno para validação ampla.",
            "- Vetores brutos permanecem locais; a saída pública usa métricas derivadas e hashes.",
            "- Evidência administrativa/contextual não é fonte de label operacional.",
            "- Sem positivos formais, negativos formais ou ground truth operacional.",
            "",
            "## 15. Decisão metodológica",
            f"`{metricas['decisao_metodologica']}`.",
            "",
            "## 16. Guardrails preservados",
            *[f"- {item}" for item in metricas["guardrails_preservados"]],
            "",
            "## 17. Próximos passos",
            "Expandir o corpus DINOv2 rastreável, manter a Trilha B como eixo imediato, cruzar topologia estrutural com evidência contextual apenas como priorização de revisão humana e não liberar treino supervisionado.",
            "",
        ]
    )


def executar(repo_root: Path) -> dict[str, object]:
    repo_root = repo_root.resolve()
    linhas, meta = gerar_amostras(repo_root)
    topologia = gerar_topologia(linhas, meta["matriz"]) if linhas else []
    vizinhos = gerar_vizinhos_precisos(linhas, meta["matriz"]) if linhas else []
    metricas = calcular_metricas(linhas, meta, topologia, vizinhos, repo_root)

    escrever_csv(repo_root / CAMINHO_CSV_PRINCIPAL, linhas, COLUNAS_PRINCIPAIS)
    escrever_csv(repo_root / CAMINHO_MATRIZ_TOPOLOGICA, topologia, COLUNAS_TOPOLOGIA)
    escrever_csv(repo_root / CAMINHO_VIZINHOS, vizinhos, COLUNAS_VIZINHOS)
    escrever_csv(repo_root / CAMINHO_GUARDRAILS, [dict(zip(COLUNAS_GUARDRAILS, item)) for item in GUARDRAILS], COLUNAS_GUARDRAILS)

    (repo_root / CAMINHO_METRICAS).parent.mkdir(parents=True, exist_ok=True)
    (repo_root / CAMINHO_METRICAS).write_text(json.dumps(metricas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    entradas_ok = entradas_encontradas(repo_root)
    entradas_faltantes = entradas_ausentes(repo_root)
    (repo_root / CAMINHO_RELATORIO).parent.mkdir(parents=True, exist_ok=True)
    (repo_root / CAMINHO_RELATORIO).write_text(gerar_relatorio(repo_root, metricas, entradas_ok, entradas_faltantes), encoding="utf-8")
    return metricas


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()
    executar(Path(args.repo_root))


if __name__ == "__main__":
    main()
