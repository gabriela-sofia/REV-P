"""MV2-DATA-03 Tarefa 1 — Preflight de config/env.

Lê a config local privada (se existir), valida flags, detecta env vars presentes SEM
imprimir valores, detecta bibliotecas opcionais e o diretório privado, e decide o modo
permitido. Se a config local não existir, marca CONFIG_MISSING e gera um runbook.

Saída: outputs_public/mv2_data_live_api_canary/mv2_data_03_config_env_preflight.csv
Runbook (quando CONFIG_MISSING): .../MV2_DATA_03_CONFIG_RUNBOOK.md
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_03_common import (
    CREDENTIAL_ENV_VARS,
    ENV_ALLOW_METADATA_CALLS,
    ENV_ALLOW_NETWORK,
    ENV_ALLOW_RASTER_CANARY,
    ENV_ALLOW_RASTER_DOWNLOAD,
    LOCAL_CONFIG_RELPATHS,
    OPTIONAL_LIBS,
    PROJECT_ROOT,
    PUBLIC_DIR,
    effective_flags,
    ensure_public_dir,
    env_present,
    is_private_path,
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
        rows.append(
            {
                "check_id": f"MV2_DATA_03_CHK_{n:02d}",
                "check_name": name,
                "status": status,
                "configured": configured,
                "enabled": enabled,
                "secret_present": secret_present,
                "secret_value_logged": "false",  # NUNCA loga valor de segredo
                "blocking_reason": blocking,
                "notes": notes,
            }
        )

    # Config local privada.
    add(
        "config_local_file",
        "FOUND" if found else "CONFIG_MISSING",
        configured="true" if found else "false",
        blocking="" if found else "config_local_ausente_nenhuma_rede_permitida",
        notes=str(flags["config_path"]) or "procure em: " + " | ".join(LOCAL_CONFIG_RELPATHS),
    )

    # Flags efetivas.
    add("allow_network", "ENABLED" if flags["allow_network"] else "DISABLED",
        configured="true" if found else "false",
        enabled="true" if flags["allow_network"] else "false",
        blocking="" if flags["allow_network"] else "rede_desabilitada_fail_closed")
    add("allow_metadata_calls", "ENABLED" if flags["allow_metadata_calls"] else "DISABLED",
        enabled="true" if flags["allow_metadata_calls"] else "false",
        blocking="" if flags["allow_metadata_calls"] else "metadata_calls_desabilitado")
    add("allow_raster_download", "ENABLED" if flags["allow_raster_download"] else "DISABLED",
        enabled="true" if flags["allow_raster_download"] else "false",
        blocking="" if flags["allow_raster_download"] else "raster_download_desabilitado")
    add("allow_canary", "ENABLED" if flags["allow_canary"] else "DISABLED",
        enabled="true" if flags["allow_canary"] else "false",
        blocking="" if flags["allow_canary"] else f"{ENV_ALLOW_RASTER_CANARY}!=YES")

    # Provider enable flags da config.
    for key in ["gee_enabled", "cdse_stac_enabled", "cdse_odata_enabled"]:
        add(key, "ENABLED" if flags[key] else "DISABLED",
            configured="true" if found else "false",
            enabled="true" if flags[key] else "false")

    # Env vars de credencial (presença, NUNCA valor).
    for var in CREDENTIAL_ENV_VARS:
        present = env_present(var)
        add(f"env_{var}", "PRESENT" if present else "ABSENT",
            secret_present="true" if present else "false",
            notes="presenca detectada sem ler valor")

    # Env vars de permissão (presença/efeito, sem valor sensível).
    for var in [ENV_ALLOW_NETWORK, ENV_ALLOW_METADATA_CALLS, ENV_ALLOW_RASTER_DOWNLOAD, ENV_ALLOW_RASTER_CANARY]:
        add(f"env_{var}", "PRESENT" if env_present(var) else "ABSENT",
            notes="flag de permissao; so vale com config local presente")

    # Bibliotecas opcionais.
    for lib in OPTIONAL_LIBS:
        ok = lib_available(lib)
        add(f"lib_{lib}", "AVAILABLE" if ok else "MISSING",
            configured="true" if ok else "false",
            notes="" if ok else "biblioteca opcional ausente")

    # Diretório privado.
    priv = str(flags["private_output_dir"])
    add("private_output_dir", "VALID" if is_private_path(priv) else "INVALID",
        configured="true",
        blocking="" if is_private_path(priv) else "private_dir_fora_de_local_only/data_local/private_outputs",
        notes=priv)

    # Decisão de modo.
    if not found:
        mode = "CONFIG_MISSING"
        reason = "sem_config_local_nenhuma_rede"
    elif flags["allow_network"] and flags["allow_metadata_calls"]:
        mode = "LIVE_METADATA_ALLOWED"
        reason = ""
    else:
        mode = "FAIL_CLOSED"
        reason = "config_presente_mas_flags_nao_habilitam_rede"
    add("mode_decision", mode, blocking=reason,
        notes="canary so com allow_canary + consenso forte")

    return rows


def _write_runbook() -> None:
    out = PUBLIC_DIR / "MV2_DATA_03_CONFIG_RUNBOOK.md"
    out.write_text(
        "# MV2-DATA-03 — Runbook de configuração (CONFIG_MISSING)\n\n"
        "Nenhuma config local privada foi encontrada, então **nenhuma chamada de rede** foi\n"
        "executada (fail-closed). Para habilitar o probe de metadados e o canary privado:\n\n"
        "## 1. Criar config local privada (git-ignored)\n\n"
        "Crie um destes arquivos (NÃO versionar, NÃO colocar segredo):\n\n"
        + "".join(f"- `{p}`\n" for p in LOCAL_CONFIG_RELPATHS)
        + "\nConteúdo (apenas flags + nome do diretório privado):\n\n"
        "```json\n"
        "{\n"
        '  "gee_enabled": false,\n'
        '  "cdse_stac_enabled": true,\n'
        '  "cdse_odata_enabled": true,\n'
        '  "allow_network": true,\n'
        '  "allow_metadata_calls": true,\n'
        '  "allow_raster_download": false,\n'
        '  "allow_canary_download": false,\n'
        '  "max_download_mb": 50,\n'
        '  "private_output_dir": "local_only/mv2_data_live_api_canary"\n'
        "}\n"
        "```\n\n"
        "## 2. Credenciais SÓ por variável de ambiente (nunca em arquivo)\n\n"
        "- `CDSE_TOKEN` ou `CDSE_USERNAME` + `CDSE_PASSWORD` (CDSE STAC/OData)\n"
        "- `GEE_PROJECT` + `GOOGLE_APPLICATION_CREDENTIALS` (Earth Engine)\n\n"
        "## 3. Permissões opcionais por env var\n\n"
        "- `REV_P_ALLOW_NETWORK=1`, `REV_P_ALLOW_METADATA_CALLS=1`\n"
        "- `REV_P_ALLOW_RASTER_DOWNLOAD=1`\n"
        "- `REV_P_ALLOW_RASTER_CANARY=YES` (confirmação explícita do canary)\n\n"
        "Mesmo com tudo habilitado: no máximo **1** raster canary privado, só para a âncora\n"
        "oficial, raster só em diretório privado, e **o Dia 10 do corpus continua bloqueado**.\n",
        encoding="utf-8",
    )


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_03_config_env_preflight.csv"
    write_csv(out, COLUMNS, rows)
    flags = effective_flags()
    if not flags["config_found"]:
        _write_runbook()
    print(
        f"[mv2_data_03_preflight] config_found={flags['config_found']} "
        f"allow_network={flags['allow_network']} allow_metadata={flags['allow_metadata_calls']} "
        f"allow_canary={flags['allow_canary']} | saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
