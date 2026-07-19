"""MV2-DATA-04 Tarefa 1 — Preflight de config/env para o corpus probe.

Lê a config local privada (se existir, via DATA-03), valida flags, detecta env vars
presentes SEM imprimir valores e decide o modo permitido. Neste marco, raster/download
estão SEMPRE bloqueados; metadata call só pode ocorrer se a config permitir. Sem config,
tudo fica NOT_EXECUTED_CONFIG_BLOCKED.

Saída: outputs_public/mv2_data_corpus_metadata_probe/mv2_data_04_config_env_preflight.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_04_common import (
    CREDENTIAL_ENV_VARS,
    ENV_ALLOW_METADATA_CALLS,
    ENV_ALLOW_NETWORK,
    OPTIONAL_LIBS,
    PROJECT_ROOT,
    PUBLIC_DIR,
    effective_flags,
    ensure_public_dir,
    env_present,
    lib_available,
    write_csv,
)

COLUMNS = [
    "check_id",
    "check_name",
    "status",
    "configured",
    "enabled",
    "secret_present",
    "secret_value_logged",
    "blocking_reason",
    "notes",
]


def build_rows() -> list[dict[str, object]]:
    flags = effective_flags()
    found = bool(flags["config_found"])
    rows: list[dict[str, object]] = []
    n = 0

    def add(name, status, configured="", enabled="", secret_present="", blocking="", notes=""):
        nonlocal n
        n += 1
        rows.append({
            "check_id": f"MV2_DATA_04_CHK_{n:02d}",
            "check_name": name,
            "status": status,
            "configured": configured,
            "enabled": enabled,
            "secret_present": secret_present,
            "secret_value_logged": "false",  # NUNCA loga valor de segredo
            "blocking_reason": blocking,
            "notes": notes,
        })

    add("config_local_file", "FOUND" if found else "CONFIG_MISSING",
        configured="true" if found else "false",
        blocking="" if found else "config_local_ausente_metadata_bloqueada",
        notes=str(flags["config_path"]) or "sem config local privada")

    add("allow_network", "ENABLED" if flags["allow_network"] else "DISABLED",
        enabled="true" if flags["allow_network"] else "false",
        blocking="" if flags["allow_network"] else "rede_desabilitada_fail_closed")
    add("allow_metadata_calls", "ENABLED" if flags["allow_metadata_calls"] else "DISABLED",
        enabled="true" if flags["allow_metadata_calls"] else "false",
        blocking="" if flags["allow_metadata_calls"] else "metadata_calls_desabilitado")

    # Raster/download SEMPRE bloqueados neste marco (independe de config).
    add("allow_raster_download", "FORCED_DISABLED", enabled="false",
        blocking="DATA-04_e_metadata_only_raster_sempre_bloqueado",
        notes="marco metadata-only; nenhum download/raster")
    add("allow_crop", "FORCED_DISABLED", enabled="false",
        blocking="DATA-04_nao_cria_crop", notes="nenhum crop neste marco")

    for key in ["gee_enabled", "cdse_stac_enabled", "cdse_odata_enabled"]:
        add(key, "ENABLED" if flags[key] else "DISABLED",
            configured="true" if found else "false",
            enabled="true" if flags[key] else "false")

    for var in CREDENTIAL_ENV_VARS:
        present = env_present(var)
        add(f"env_{var}", "PRESENT" if present else "ABSENT",
            secret_present="true" if present else "false",
            notes="presenca detectada sem ler valor")
    for var in [ENV_ALLOW_NETWORK, ENV_ALLOW_METADATA_CALLS]:
        add(f"env_{var}", "PRESENT" if env_present(var) else "ABSENT",
            notes="flag de permissao; so vale com config local presente")

    for lib in OPTIONAL_LIBS:
        ok = lib_available(lib)
        add(f"lib_{lib}", "AVAILABLE" if ok else "MISSING",
            configured="true" if ok else "false",
            notes="" if ok else "biblioteca opcional ausente")

    # Decisão de modo.
    if not found:
        mode, reason = "CONFIG_MISSING", "sem_config_local_metadata_bloqueada"
    elif flags["allow_network"] and flags["allow_metadata_calls"]:
        mode, reason = "METADATA_ALLOWED_IF_TEMPORAL_WINDOW", ""
    else:
        mode, reason = "NOT_EXECUTED_CONFIG_BLOCKED", "config_presente_mas_flags_nao_habilitam"
    add("mode_decision", mode, blocking=reason,
        notes="metadata só executa COM janela temporal por target")

    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_04_config_env_preflight.csv"
    write_csv(out, COLUMNS, rows)
    flags = effective_flags()
    print(
        f"[mv2_data_04_preflight] config_found={flags['config_found']} "
        f"allow_network={flags['allow_network']} allow_metadata={flags['allow_metadata_calls']} "
        f"(raster/download sempre bloqueados) | saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
