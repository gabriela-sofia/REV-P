"""
Diagnostico do conda no Windows -- por que o solve trava.

Contexto (2026-08-06): duas tentativas de `conda install` do stack geoespacial
no ambiente revp-susc ficaram presas por 7,5h e 9,7h sem escrever um unico byte
no ambiente e sem crescer o cache de pacotes. O mesmo conjunto instalou por pip
em minutos, o que descarta problema de rede geral e aponta para o conda
especificamente.

Este script NAO roda nenhum solve e NAO instala nada. Ele so coleta evidencia,
com timeout explicito em toda chamada, para separar tres hipoteses:

  H1: conda-libmamba-solver ausente -> `--solver=libmamba` foi ignorado e o
      solver classico rodou (o classico realmente trava nesse stack em win-64).
  H2: cache de repodata corrompido -> conda espera indefinidamente por um
      arquivo que nunca completa.
  H3: canais do conda (conda.anaconda.org / repo.anaconda.com) inacessiveis ou
      lentissimos, enquanto pypi.org responde normal.

Uso:
    C:\\Users\\gabriela\\miniconda3\\envs\\revp-susc\\python.exe scripts\\ambiente\\diag_conda_win.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

CONDA_ROOT = Path(r"C:\Users\gabriela\miniconda3")
TIMEOUT_CMD = 60
TIMEOUT_NET = 45

PROBES = [
    ("pypi", "https://pypi.org/simple/geopandas/"),
    ("anaconda_org_cf", "https://conda.anaconda.org/conda-forge/win-64/repodata.json"),
    ("anaconda_org_noarch", "https://conda.anaconda.org/conda-forge/noarch/repodata.json"),
    ("repo_anaconda_main", "https://repo.anaconda.com/pkgs/main/win-64/repodata.json"),
]


def run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TIMEOUT_CMD, shell=False
        )
        return (out.stdout or out.stderr).strip()
    except subprocess.TimeoutExpired:
        return f"TIMEOUT_{TIMEOUT_CMD}s"
    except Exception as exc:  # noqa: BLE001
        return f"ERRO:{type(exc).__name__}"


def probe(url: str) -> str:
    """Baixa so o primeiro chunk -- repodata.json do conda-forge tem centenas de MB."""
    import requests

    t0 = time.time()
    try:
        with requests.get(url, stream=True, timeout=TIMEOUT_NET) as r:
            code = r.status_code
            size = r.headers.get("Content-Length", "?")
            primeiro = next(r.iter_content(chunk_size=65536), b"")
            dt = time.time() - t0
            kbs = (len(primeiro) / 1024) / dt if dt > 0 else 0
            return f"HTTP={code} bytes_declarados={size} 1o_chunk={len(primeiro)}B em {dt:.1f}s (~{kbs:.0f}KB/s)"
    except Exception as exc:  # noqa: BLE001
        return f"FALHOU apos {time.time()-t0:.1f}s: {type(exc).__name__}"


def main() -> int:
    conda_exe = str(CONDA_ROOT / "Scripts" / "conda.exe")

    print("===DIAG_CONDA===")

    # --- H1: solver ---
    print("CONDA_VER=" + run([conda_exe, "--version"]).replace("\n", " "))
    lista = run([conda_exe, "list", "-n", "base", "--json", "conda-libmamba-solver"])
    tem_solver = "conda-libmamba-solver" in lista and "[]" not in lista[:5]
    print(f"LIBMAMBA_INSTALADO={tem_solver}")
    try:
        import importlib.util as u
        print(f"LIBMAMBA_IMPORTAVEL={u.find_spec('conda_libmamba_solver') is not None}")
    except Exception:  # noqa: BLE001
        print("LIBMAMBA_IMPORTAVEL=?")
    print("MAMBA_BIN=" + str(shutil.which("mamba")))
    print("MICROMAMBA_BIN=" + str(shutil.which("micromamba")))

    # --- H2: cache de repodata ---
    cache = CONDA_ROOT / "pkgs" / "cache"
    if cache.exists():
        arquivos = sorted(cache.glob("*.json"), key=lambda p: p.stat().st_size, reverse=True)
        print(f"REPODATA_CACHE_N={len(arquivos)}")
        for p in arquivos[:5]:
            mb = p.stat().st_size / 1024 / 1024
            idade_h = (time.time() - p.stat().st_mtime) / 3600
            valido = "?"
            try:
                json.loads(p.read_text(encoding="utf-8"))
                valido = "JSON_OK"
            except Exception:  # noqa: BLE001
                valido = "JSON_CORROMPIDO"
            print(f"  CACHE {p.name[:20]} {mb:.1f}MB idade={idade_h:.1f}h {valido}")
    else:
        print("REPODATA_CACHE_N=0 (pasta ausente)")

    # --- config ---
    condarc = Path(os.path.expanduser("~")) / ".condarc"
    print(f"CONDARC_EXISTE={condarc.exists()}")
    if condarc.exists():
        for linha in condarc.read_text(encoding="utf-8").splitlines():
            if linha.strip():
                print("  RC " + linha.strip()[:70])

    # --- H3: rede ---
    print("---REDE---")
    for nome, url in PROBES:
        print(f"{nome}: {probe(url)}")

    total, usado, livre = shutil.disk_usage(CONDA_ROOT.anchor)
    print(f"DISCO_LIVRE_GB={livre/1024**3:.0f}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
