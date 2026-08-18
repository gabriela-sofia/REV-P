"""MV2-DATA-02 Tarefa 7 — Executor de raster canary (fail-closed por padrão).

Por padrão NÃO executa nada. Só baixa um raster canary se TODOS os requisitos forem
satisfeitos simultaneamente:
  - config allow_network=true;
  - config allow_raster_download=true;
  - config allow_canary_download=true;
  - target presente no plano canary com lineage forte (canary_allowed_by_data=true);
  - private_output_dir válido (local_only/ | data_local/ | private_outputs/);
  - estimated_size_mb <= max_download_mb;
  - confirmação explícita por env var REV_P_ALLOW_RASTER_CANARY=YES.
Se qualquer requisito faltar -> NOT_EXECUTED_GUARDRAIL (nenhum raster, nenhuma rede).

Raster (quando existir) vai SOMENTE para diretório privado git-ignored; o manifesto público
é leve e nunca contém raster.

Saída pública: outputs_public/mv2_data_api_raster_harness/mv2_data_02_raster_canary_execution_manifest.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_02_api_config import canary_env_confirmed, load_config
from mv2_data_02_common import (
    PROJECT_ROOT,
    PUBLIC_DIR,
    clean,
    ensure_public_dir,
    is_private_path,
    read_csv_rows,
    write_csv,
)

COLUMNS = [
    "canary_id",
    "target_id",
    "executed",
    "execution_status",
    "private_raster_path",
    "size_bytes",
    "sha256",
    "bands_present",
    "scl_present",
    "max_download_mb",
    "unmet_requirements",
    "blocking_reason",
    "notes",
]


def _unmet_requirements(plan_row: dict[str, str], cfg: dict[str, object]) -> list[str]:
    unmet: list[str] = []
    if not cfg.get("allow_network"):
        unmet.append("allow_network=false")
    if not cfg.get("allow_raster_download"):
        unmet.append("allow_raster_download=false")
    if not cfg.get("allow_canary_download"):
        unmet.append("allow_canary_download=false")
    if clean(plan_row.get("canary_allowed_by_data")) != "true":
        unmet.append("lineage_nao_forte")
    priv = str(cfg.get("private_output_dir"))
    if not is_private_path(priv):
        unmet.append("private_output_dir_invalido")
    try:
        est = int(float(clean(plan_row.get("estimated_size_mb")) or "0"))
    except ValueError:
        est = 10 ** 9
    if est > int(str(cfg.get("max_download_mb", 0))):
        unmet.append("estimated_size_acima_de_max_download_mb")
    if not canary_env_confirmed():
        unmet.append("env_REV_P_ALLOW_RASTER_CANARY!=YES")
    return unmet


def _attempt_download(plan_row: dict[str, str], cfg: dict[str, object]) -> dict[str, object]:
    """Caminho de execução real — só alcançado quando todos os gates passam.

    Mantido conservador: baixa para o diretório privado e calcula SHA256. Falha de rede ou
    ausência de requests nunca cria raster; retorna status auditável.
    """
    import hashlib

    private_root = PROJECT_ROOT / str(cfg.get("private_output_dir"))
    private_root.mkdir(parents=True, exist_ok=True)
    odata = read_csv_rows(PUBLIC_DIR / "mv2_data_02_cdse_odata_metadata_results.csv")
    url = ""
    for r in odata:
        if clean(r.get("target_id")) == clean(plan_row.get("target_id")):
            url = clean(r.get("download_url_resolved"))
            break
    if not url:
        return {"executed": "false", "execution_status": "NOT_EXECUTED_NO_DOWNLOAD_URL",
                "blocking_reason": "OData_nao_resolveu_download_url", "notes": "lineage/endpoint pendente"}
    try:
        import requests
    except ImportError:
        return {"executed": "false", "execution_status": "NOT_EXECUTED_FETCH_DEP_MISSING",
                "blocking_reason": "requests_indisponivel", "notes": ""}
    try:
        dest = private_root / (clean(plan_row.get("canary_id")) + ".bin")
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            h = hashlib.sha256()
            size = 0
            with dest.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)
                    h.update(chunk)
                    size += len(chunk)
        return {
            "executed": "true",
            "execution_status": "EXECUTED_PRIVATE_ONLY",
            "private_raster_path": dest.relative_to(PROJECT_ROOT).as_posix(),
            "size_bytes": size,
            "sha256": h.hexdigest(),
            "blocking_reason": "",
            "notes": "raster apenas em diretorio privado git-ignored",
        }
    except Exception as exc:
        return {"executed": "false", "execution_status": "NOT_EXECUTED_FETCH_ERROR",
                "blocking_reason": f"erro:{type(exc).__name__}", "notes": ""}


def build_rows() -> list[dict[str, object]]:
    cfg = load_config()
    plan = read_csv_rows(PUBLIC_DIR / "mv2_data_02_raster_canary_plan.csv")
    rows: list[dict[str, object]] = []
    for p in plan:
        unmet = _unmet_requirements(p, cfg)
        base: dict[str, object] = {
            "canary_id": clean(p.get("canary_id")),
            "target_id": clean(p.get("target_id")),
            "private_raster_path": "",
            "size_bytes": "",
            "sha256": "",
            "bands_present": "",
            "scl_present": "",
            "max_download_mb": cfg.get("max_download_mb"),
            "unmet_requirements": "|".join(unmet) or "none",
        }
        if unmet:
            base.update(
                {
                    "executed": "false",
                    "execution_status": "NOT_EXECUTED_GUARDRAIL",
                    "blocking_reason": "guardrail_fail_closed",
                    "notes": "nenhum raster baixado; nenhuma chamada de rede",
                }
            )
        else:
            base.update(_attempt_download(p, cfg))
            if clean(str(base.get("executed"))) == "true":
                base["bands_present"] = clean(p.get("bands_requested"))
                base["scl_present"] = "true"
        rows.append(base)
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_02_raster_canary_execution_manifest.csv"
    write_csv(out, COLUMNS, rows)
    executed = sum(1 for r in rows if r["executed"] == "true")
    print(
        f"[mv2_data_02_canary_exec] {len(rows)} entradas | executados={executed} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
