from __future__ import annotations

import csv
import json
from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable


CSV_COLUNAS = [
    "id_patch_canonico",
    "regiao",
    "total_assets",
    "total_assets_limpos",
    "total_datas_aquisicao",
    "datas_aquisicao",
    "cobertura_nuvens_minima",
    "cobertura_nuvens_maxima",
    "cobertura_nuvens_media",
    "cobertura_janelas_temporais",
    "tem_1_data",
    "tem_2_datas",
    "tem_3_ou_mais_datas",
    "elegivel_deslocamento_temporal",
    "trilha_recomendada",
    "motivo_bloqueio",
]

ARQUIVOS_ENTRADA = [
    Path("manifests/dino_inputs/revp_v1fu_dino_sentinel_input_manifest/dino_sentinel_input_manifest_v1fu.csv"),
    Path("outputs_public/tables/table_dino_embedding_inventory.csv"),
    Path("datasets/protocolo_c/v2ap_sentinel_asset_inventory.csv"),
    Path("datasets/protocolo_c/v2aa_patch_date_candidate_consolidation.csv"),
    Path("datasets/protocolo_c/v2aa_sentinel_filename_date_extraction.csv"),
    Path("datasets/protocolo_c/v2ag_sentinel_date_linkability_audit.csv"),
    Path("datasets/protocolo_c/v2ag_event_patch_temporal_preview.csv"),
    Path("datasets/official_anchor_sentinel_patch_registry.csv"),
    Path("datasets/official_anchor_sentinel_patch_quality_registry.csv"),
    Path("datasets/event_patch_linkage_registry.csv"),
]

CAMINHO_TABELA = Path("outputs_public/tables/revp_auditoria_prontidao_temporal_assets_mv1.csv")
CAMINHO_RELATORIO = Path("outputs_public/execution_reports/revp_auditoria_prontidao_temporal_assets_mv1.md")
CAMINHO_METRICAS = Path("outputs_public/metrics/revp_auditoria_prontidao_temporal_assets_mv1.json")

TRILHA_A = "TRILHA_A_DESLOCAMENTO_TEMPORAL_EMBEDDINGS"
TRILHA_A_PILOTO = "TRILHA_A_PILOTO_E_TRILHA_B_PRINCIPAL"
TRILHA_B = "TRILHA_B_TOPOLOGIA_ENTRE_CIDADES_AEC_LABEL_FREE"
TRILHA_B_METADADOS = "METADADOS_TEMPORAIS_INSUFICIENTES_TRILHA_B_RECOMENDADA"

GUARDRAILS = [
    "sem_treino_de_modelo",
    "sem_criacao_de_labels",
    "sem_negativos_formais",
    "sem_promocao_de_ground_truth",
    "sem_tabela_multimodal_de_atributos",
    "DINOv2_apenas_encoder_congelado_exploratorio",
    "analise_apenas_metadados",
]


@dataclass
class PatchAudit:
    id_patch: str
    regioes: set[str] = field(default_factory=set)
    assets: set[str] = field(default_factory=set)
    assets_limpos: set[str] = field(default_factory=set)
    datas: set[str] = field(default_factory=set)
    nuvens: list[float] = field(default_factory=list)
    fontes: set[str] = field(default_factory=set)

    def adicionar(
        self,
        *,
        fonte: str,
        asset_id: str,
        regiao: str = "",
        data: str = "",
        nuvem: str = "",
        asset_temporal_limpo: bool = False,
    ) -> None:
        fonte_curta = fonte.replace("\\", "/")
        asset = asset_id.strip() or f"{fonte_curta}:{len(self.assets) + 1}"
        self.assets.add(f"{fonte_curta}:{asset}")
        self.fontes.add(fonte_curta)
        if regiao.strip():
            self.regioes.add(normalizar_regiao(regiao))
        if data.strip():
            self.datas.add(data.strip())
        valor_nuvem = parse_float(nuvem)
        if valor_nuvem is not None:
            self.nuvens.append(valor_nuvem)
        if asset_temporal_limpo:
            self.assets_limpos.add(f"{fonte_curta}:{asset}")


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Audita prontidao temporal de assets Sentinel/DINO para MV1.")
    parser.add_argument("--repo-root", default=".", help="Raiz do repositorio REV-P.")
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


def normalizar_regiao(valor: str) -> str:
    texto = valor.strip()
    mapa = {
        "CUR": "Curitiba",
        "CURITIBA": "Curitiba",
        "PET": "Petrópolis",
        "PETROPOLIS": "Petrópolis",
        "PETRÓPOLIS": "Petrópolis",
        "REC": "Recife",
        "RECIFE": "Recife",
        "UNSPECIFIED": "desconhecido",
        "UNKNOWN": "desconhecido",
    }
    return mapa.get(texto.upper(), texto or "desconhecido")


def parse_float(valor: str) -> float | None:
    if valor is None:
        return None
    texto = str(valor).strip().replace(",", ".")
    if not texto:
        return None
    try:
        return float(texto)
    except ValueError:
        return None


def dividir_datas(valor: str) -> list[str]:
    if not valor:
        return []
    partes = []
    for bruto in str(valor).replace(";", "|").split("|"):
        texto = bruto.strip()
        if texto:
            partes.append(texto)
    return partes


def escolher_id(row: dict[str, str], campos: Iterable[str]) -> str:
    for campo in campos:
        valor = row.get(campo, "").strip()
        if valor:
            return valor
    return ""


def escolher_valor(row: dict[str, str], campos: Iterable[str]) -> str:
    for campo in campos:
        valor = row.get(campo, "").strip()
        if valor:
            return valor
    return ""


def add_patch(
    patches: dict[str, PatchAudit],
    *,
    id_patch: str,
    fonte: str,
    asset_id: str,
    regiao: str = "",
    data: str = "",
    nuvem: str = "",
    asset_temporal_limpo: bool = False,
) -> None:
    patch_id = id_patch.strip()
    if not patch_id:
        return
    if patch_id not in patches:
        patches[patch_id] = PatchAudit(patch_id)
    patches[patch_id].adicionar(
        fonte=fonte,
        asset_id=asset_id,
        regiao=regiao,
        data=data,
        nuvem=nuvem,
        asset_temporal_limpo=asset_temporal_limpo,
    )


def coletar_dados(repo_root: Path) -> tuple[dict[str, PatchAudit], dict[str, object]]:
    patches: dict[str, PatchAudit] = {}
    arquivos_encontrados: list[str] = []
    arquivos_ausentes: list[str] = []
    campos_usados: dict[str, list[str]] = {}
    campos_ausentes: dict[str, list[str]] = {}
    dino_patch_ids: set[str] = set()
    fontes_com_nuvem: set[str] = set()

    for rel in ARQUIVOS_ENTRADA:
        path = repo_root / rel
        if path.exists():
            arquivos_encontrados.append(rel.as_posix())
        else:
            arquivos_ausentes.append(rel.as_posix())

    dino_manifest = repo_root / ARQUIVOS_ENTRADA[0]
    for row in ler_csv(dino_manifest):
        patch_id = escolher_id(row, ["canonical_patch_id", "patch_id"])
        if not patch_id:
            continue
        dino_patch_ids.add(patch_id)
        add_patch(
            patches,
            id_patch=patch_id,
            fonte=ARQUIVOS_ENTRADA[0].as_posix(),
            asset_id=escolher_valor(row, ["dino_input_id", "source_asset_id"]),
            regiao=row.get("region", ""),
        )
    campos_usados[ARQUIVOS_ENTRADA[0].as_posix()] = ["canonical_patch_id", "region", "dino_input_id", "source_asset_id"]
    campos_ausentes[ARQUIVOS_ENTRADA[0].as_posix()] = ["data_aquisicao", "cobertura_nuvens"]

    inventario_dino = repo_root / ARQUIVOS_ENTRADA[1]
    for row in ler_csv(inventario_dino):
        add_patch(
            patches,
            id_patch=escolher_id(row, ["patch_id"]),
            fonte=ARQUIVOS_ENTRADA[1].as_posix(),
            asset_id=escolher_valor(row, ["dino_input_id", "vector_sha256"]),
            regiao=row.get("region", ""),
        )
    campos_usados[ARQUIVOS_ENTRADA[1].as_posix()] = ["patch_id", "region", "dino_input_id"]
    campos_ausentes[ARQUIVOS_ENTRADA[1].as_posix()] = ["data_aquisicao", "cobertura_nuvens"]

    inventario_assets = repo_root / ARQUIVOS_ENTRADA[2]
    for row in ler_csv(inventario_assets):
        data = row.get("date_detected", "")
        seguro = row.get("safe_to_use_as_crosswalk_evidence", "").strip().lower() == "true"
        add_patch(
            patches,
            id_patch=row.get("patch_id_detected", ""),
            fonte=ARQUIVOS_ENTRADA[2].as_posix(),
            asset_id=row.get("asset_id", ""),
            regiao=row.get("region_detected", ""),
            data=data,
            asset_temporal_limpo=bool(data.strip() and seguro),
        )
    campos_usados[ARQUIVOS_ENTRADA[2].as_posix()] = ["patch_id_detected", "region_detected", "date_detected", "safe_to_use_as_crosswalk_evidence"]
    campos_ausentes[ARQUIVOS_ENTRADA[2].as_posix()] = ["cobertura_nuvens"]

    consolidacao_datas = repo_root / ARQUIVOS_ENTRADA[3]
    for row in ler_csv(consolidacao_datas):
        datas = dividir_datas(row.get("candidate_dates", "")) or dividir_datas(row.get("selected_sentinel_date", ""))
        if not datas:
            add_patch(
                patches,
                id_patch=row.get("patch_id", ""),
                fonte=ARQUIVOS_ENTRADA[3].as_posix(),
                asset_id=row.get("patch_date_id", ""),
                regiao=row.get("region", ""),
            )
            continue
        for data in datas:
            recuperada = row.get("sentinel_date_recovered", "").strip().lower() == "true"
            conflito = row.get("conflict_status", "").strip().upper() == "CONFLICT"
            add_patch(
                patches,
                id_patch=row.get("patch_id", ""),
                fonte=ARQUIVOS_ENTRADA[3].as_posix(),
                asset_id=f"{row.get('patch_date_id', '')}:{data}",
                regiao=row.get("region", ""),
                data=data,
                asset_temporal_limpo=recuperada and not conflito,
            )
    campos_usados[ARQUIVOS_ENTRADA[3].as_posix()] = ["patch_id", "region", "candidate_dates", "selected_sentinel_date", "sentinel_date_recovered"]
    campos_ausentes[ARQUIVOS_ENTRADA[3].as_posix()] = ["cobertura_nuvens"]

    extracao_nome = repo_root / ARQUIVOS_ENTRADA[4]
    for row in ler_csv(extracao_nome):
        data = row.get("extracted_date", "")
        status = row.get("extraction_status", "").strip().upper()
        add_patch(
            patches,
            id_patch=row.get("patch_id", ""),
            fonte=ARQUIVOS_ENTRADA[4].as_posix(),
            asset_id=row.get("filename_extract_id", ""),
            data=data,
            asset_temporal_limpo=bool(data.strip() and status in {"EXTRACTED", "DATE_EXTRACTED", "PASS"}),
        )
    campos_usados[ARQUIVOS_ENTRADA[4].as_posix()] = ["patch_id", "extracted_date", "extraction_status"]
    campos_ausentes[ARQUIVOS_ENTRADA[4].as_posix()] = ["regiao", "cobertura_nuvens"]

    linkability = repo_root / ARQUIVOS_ENTRADA[5]
    for row in ler_csv(linkability):
        data = row.get("recovered_date", "")
        pode = row.get("can_link_sentinel_date", "").strip().lower() == "true"
        add_patch(
            patches,
            id_patch=row.get("patch_id", ""),
            fonte=ARQUIVOS_ENTRADA[5].as_posix(),
            asset_id=row.get("linkability_audit_id", ""),
            data=data,
            asset_temporal_limpo=bool(data.strip() and pode),
        )
    campos_usados[ARQUIVOS_ENTRADA[5].as_posix()] = ["patch_id", "recovered_date", "can_link_sentinel_date"]
    campos_ausentes[ARQUIVOS_ENTRADA[5].as_posix()] = ["regiao", "cobertura_nuvens"]

    preview = repo_root / ARQUIVOS_ENTRADA[6]
    for row in ler_csv(preview):
        data = row.get("preview_sentinel_date", "")
        status = row.get("preview_status", "").strip().upper()
        add_patch(
            patches,
            id_patch=row.get("patch_id", ""),
            fonte=ARQUIVOS_ENTRADA[6].as_posix(),
            asset_id=row.get("temporal_preview_id", ""),
            data=data,
            asset_temporal_limpo=bool(data.strip() and "BLOCK" not in status),
        )
    campos_usados[ARQUIVOS_ENTRADA[6].as_posix()] = ["patch_id", "preview_sentinel_date", "preview_status"]
    campos_ausentes[ARQUIVOS_ENTRADA[6].as_posix()] = ["regiao", "cobertura_nuvens"]

    registro_oficial = repo_root / ARQUIVOS_ENTRADA[7]
    for row in ler_csv(registro_oficial):
        nuvem = row.get("cloud_cover_metadata", "")
        if nuvem.strip():
            fontes_com_nuvem.add(ARQUIVOS_ENTRADA[7].as_posix())
        add_patch(
            patches,
            id_patch=row.get("reference_patch_id", ""),
            fonte=ARQUIVOS_ENTRADA[7].as_posix(),
            asset_id=escolher_valor(row, ["reference_patch_id", "scene_id_sanitized"]),
            regiao=row.get("region", ""),
            data=row.get("scene_date", ""),
            nuvem=nuvem,
            asset_temporal_limpo=bool(row.get("scene_date", "").strip() and nuvem.strip()),
        )
    campos_usados[ARQUIVOS_ENTRADA[7].as_posix()] = ["reference_patch_id", "region", "scene_date", "cloud_cover_metadata"]

    qualidade_oficial = repo_root / ARQUIVOS_ENTRADA[8]
    for row in ler_csv(qualidade_oficial):
        nuvem = escolher_valor(row, ["local_cloud_fraction", "cloud_metadata_global"])
        if nuvem.strip():
            fontes_com_nuvem.add(ARQUIVOS_ENTRADA[8].as_posix())
        add_patch(
            patches,
            id_patch=row.get("reference_patch_id", ""),
            fonte=ARQUIVOS_ENTRADA[8].as_posix(),
            asset_id=escolher_valor(row, ["quality_id", "scene_id_sanitized"]),
            data=row.get("scene_date", ""),
            nuvem=nuvem,
            asset_temporal_limpo=bool(row.get("scene_date", "").strip() and nuvem.strip()),
        )
    campos_usados[ARQUIVOS_ENTRADA[8].as_posix()] = ["reference_patch_id", "scene_date", "local_cloud_fraction", "cloud_metadata_global"]

    linkage = repo_root / ARQUIVOS_ENTRADA[9]
    for row in ler_csv(linkage):
        patch_id = row.get("patch_candidate_id", "")
        for campo in ["pre_scene_date", "post_scene_date"]:
            data = row.get(campo, "")
            if data.strip():
                add_patch(
                    patches,
                    id_patch=patch_id,
                    fonte=ARQUIVOS_ENTRADA[9].as_posix(),
                    asset_id=f"{row.get('linkage_id', '')}:{campo}",
                    data=data,
                    asset_temporal_limpo=True,
                )
    campos_usados[ARQUIVOS_ENTRADA[9].as_posix()] = ["patch_candidate_id", "pre_scene_date", "post_scene_date"]
    campos_ausentes[ARQUIVOS_ENTRADA[9].as_posix()] = ["cobertura_nuvens"]

    meta = {
        "arquivos_encontrados": arquivos_encontrados,
        "arquivos_ausentes": arquivos_ausentes,
        "campos_usados": campos_usados,
        "campos_ausentes": campos_ausentes,
        "dino_patch_ids": dino_patch_ids,
        "fontes_com_nuvem": fontes_com_nuvem,
    }
    return patches, meta


def classificar_janela(total_datas: int) -> str:
    if total_datas <= 0:
        return "sem_data"
    if total_datas == 1:
        return "1_data"
    if total_datas == 2:
        return "2_datas"
    return "3_ou_mais_datas"


def patch_temporalmente_elegivel(patch: PatchAudit) -> bool:
    return len(patch.datas) >= 3 and bool(patch.nuvens)


def motivo_bloqueio(patch: PatchAudit) -> str:
    motivos = []
    if not patch.datas:
        motivos.append("METADADOS_TEMPORAIS_AUSENTES")
    elif len(patch.datas) < 3:
        motivos.append("MENOS_DE_3_DATAS_LIMPAS")
    if not patch.nuvens:
        motivos.append("METADADOS_NUVEM_AUSENTES")
    return ";".join(motivos) if motivos else "SEM_BLOQUEIO_TECNICO_NESTA_AUDITORIA"


def decidir_global(patches: dict[str, PatchAudit], dino_patch_ids: set[str], fontes_com_nuvem: set[str]) -> str:
    elegiveis = sum(1 for patch in patches.values() if patch_temporalmente_elegivel(patch))
    dino_com_data = sum(1 for pid in dino_patch_ids if pid in patches and patches[pid].datas)
    dino_com_nuvem = sum(1 for pid in dino_patch_ids if pid in patches and patches[pid].nuvens)

    if not patches or not fontes_com_nuvem:
        return TRILHA_B_METADADOS
    if dino_patch_ids and (dino_com_data < 20 or dino_com_nuvem < 20):
        return TRILHA_B_METADADOS
    if elegiveis >= 30:
        return TRILHA_A
    if 20 <= elegiveis <= 29:
        return TRILHA_A_PILOTO
    return TRILHA_B


def formatar_numero(valor: float) -> str:
    return f"{valor:.6f}".rstrip("0").rstrip(".")


def linhas_csv(patches: dict[str, PatchAudit], decisao_global: str) -> list[dict[str, str]]:
    linhas = []
    for patch_id in sorted(patches):
        patch = patches[patch_id]
        datas = sorted(patch.datas)
        nuvens = patch.nuvens
        total_datas = len(datas)
        linhas.append(
            {
                "id_patch_canonico": patch_id,
                "regiao": ";".join(sorted(patch.regioes)) if patch.regioes else "desconhecido",
                "total_assets": str(len(patch.assets)),
                "total_assets_limpos": str(len(patch.assets_limpos)),
                "total_datas_aquisicao": str(total_datas),
                "datas_aquisicao": ";".join(datas) if datas else "ausente",
                "cobertura_nuvens_minima": formatar_numero(min(nuvens)) if nuvens else "ausente",
                "cobertura_nuvens_maxima": formatar_numero(max(nuvens)) if nuvens else "ausente",
                "cobertura_nuvens_media": formatar_numero(mean(nuvens)) if nuvens else "ausente",
                "cobertura_janelas_temporais": classificar_janela(total_datas),
                "tem_1_data": "sim" if total_datas == 1 else "nao",
                "tem_2_datas": "sim" if total_datas == 2 else "nao",
                "tem_3_ou_mais_datas": "sim" if total_datas >= 3 else "nao",
                "elegivel_deslocamento_temporal": "sim" if patch_temporalmente_elegivel(patch) else "nao",
                "trilha_recomendada": decisao_global,
                "motivo_bloqueio": motivo_bloqueio(patch),
            }
        )
    return linhas


def metricas(linhas: list[dict[str, str]], decisao_global: str) -> dict[str, object]:
    contagem_regiao = Counter(row["regiao"] for row in linhas)
    total_assets = sum(int(row["total_assets"]) for row in linhas)
    bloqueios = Counter(
        motivo
        for row in linhas
        for motivo in row["motivo_bloqueio"].split(";")
        if motivo and motivo != "SEM_BLOQUEIO_TECNICO_NESTA_AUDITORIA"
    )
    return {
        "total_patches": len(linhas),
        "total_assets": total_assets,
        "contagem_por_regiao": dict(sorted(contagem_regiao.items())),
        "patches_com_1_data": sum(1 for row in linhas if row["tem_1_data"] == "sim"),
        "patches_com_2_datas": sum(1 for row in linhas if row["tem_2_datas"] == "sim"),
        "patches_com_3_ou_mais_datas": sum(1 for row in linhas if row["tem_3_ou_mais_datas"] == "sim"),
        "patches_elegiveis_deslocamento_temporal": sum(1 for row in linhas if row["elegivel_deslocamento_temporal"] == "sim"),
        "decisao_global": decisao_global,
        "bloqueios_principais": dict(bloqueios.most_common()),
        "guardrails_preservados": GUARDRAILS,
    }


def relatorio_markdown(repo_root: Path, linhas: list[dict[str, str]], meta: dict[str, object], metricas_json: dict[str, object]) -> str:
    branch = obter_branch(repo_root)
    contagem_regiao = metricas_json["contagem_por_regiao"]
    usados = meta["campos_usados"]
    ausentes = meta["campos_ausentes"]

    return "\n".join(
        [
            "# Auditoria de prontidão temporal de assets MV1",
            "",
            "## Escopo",
            "Auditoria apenas de metadados da prontidão temporal de assets Sentinel/DINO para decidir a próxima trilha MV1: deslocamento temporal label-free de embeddings ou correlação topológica/administrativa entre cidades.",
            "",
            "## Branch",
            str(branch),
            "",
            "## Arquivos de entrada encontrados",
            *[f"- `{item}`" for item in meta["arquivos_encontrados"]],
            "",
            "## Arquivos de entrada ausentes",
            *[f"- `{item}`" for item in meta["arquivos_ausentes"]],
            "",
            "## Campos usados",
            *[f"- `{fonte}`: {', '.join(campos)}" for fonte, campos in sorted(usados.items())],
            "",
            "## Campos ausentes ou insuficientes",
            *[f"- `{fonte}`: {', '.join(campos)}" for fonte, campos in sorted(ausentes.items())],
            "",
            "## Contagens por número de datas",
            f"- 1 data: {metricas_json['patches_com_1_data']}",
            f"- 2 datas: {metricas_json['patches_com_2_datas']}",
            f"- 3 ou mais datas: {metricas_json['patches_com_3_ou_mais_datas']}",
            "",
            "## Contagens por região",
            *[f"- {regiao}: {total}" for regiao, total in sorted(contagem_regiao.items())],
            "",
            "## Patches elegíveis",
            f"- Elegíveis para deslocamento temporal: {metricas_json['patches_elegiveis_deslocamento_temporal']}",
            "",
            "## Decisão A/B",
            f"- Decisão global: `{metricas_json['decisao_global']}`",
            "- Critério aplicado: a trilha A exige volume suficiente de patches com 3 ou mais datas limpas e metadados de nuvem rastreáveis; quando o universo Sentinel/DINO principal não possui metadados temporais/de nuvem suficientes, a trilha B permanece recomendada.",
            "",
            "## Limitações",
            "- Datas ausentes foram registradas como `ausente`; regiões não documentadas foram registradas como `desconhecido`.",
            "- A auditoria não abre rasters, não lê pixels, não baixa assets e não calcula embeddings.",
            "- Metadados temporais e de nuvem aparecem de forma parcial em registros estruturados, mas não fecham prontidão temporal ampla para MV1.",
            "",
            "## Guardrails preservados",
            *[f"- {item}" for item in GUARDRAILS],
            "",
        ]
    )


def obter_branch(repo_root: Path) -> str:
    head = repo_root / ".git" / "HEAD"
    if not head.exists():
        return "desconhecido"
    texto = head.read_text(encoding="utf-8", errors="replace").strip()
    if texto.startswith("ref: refs/heads/"):
        return texto.removeprefix("ref: refs/heads/")
    return texto or "desconhecido"


def executar(repo_root: Path) -> dict[str, object]:
    repo_root = repo_root.resolve()
    patches, meta = coletar_dados(repo_root)
    decisao = decidir_global(patches, set(meta["dino_patch_ids"]), set(meta["fontes_com_nuvem"]))
    linhas = linhas_csv(patches, decisao)
    metricas_json = metricas(linhas, decisao)
    metricas_json["gerado_em_utc"] = datetime.now(timezone.utc).isoformat()

    escrever_csv(repo_root / CAMINHO_TABELA, linhas, CSV_COLUNAS)

    (repo_root / CAMINHO_METRICAS).parent.mkdir(parents=True, exist_ok=True)
    (repo_root / CAMINHO_METRICAS).write_text(
        json.dumps(metricas_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    (repo_root / CAMINHO_RELATORIO).parent.mkdir(parents=True, exist_ok=True)
    (repo_root / CAMINHO_RELATORIO).write_text(
        relatorio_markdown(repo_root, linhas, meta, metricas_json),
        encoding="utf-8",
    )

    return metricas_json


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()
    executar(Path(args.repo_root))


if __name__ == "__main__":
    main()
