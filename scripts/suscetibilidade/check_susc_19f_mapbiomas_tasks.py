"""Atualiza o status das tasks GEE do SUSC-19F e registra no output publico.

A execucao de 19F usa getInfo sincrono (sem export assincrono no Drive), entao as
"tasks" sao os chunks de getInfo ja concluidos e registrados em
mapbiomas_tasks.csv (local_runs). Este script le esse arquivo, atualiza o status e
escreve status_tasks_mapbiomas_19f.csv no output publico. Se nao houver tasks locais,
registra estado "nao_iniciado" sem inventar nada.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import susc_19f_mapbiomas_common as s19f  # noqa: E402
from susc_io import ensure_dir, read_csv, write_csv  # noqa: E402


def main() -> int:
    ensure_dir(s19f.OUT)
    if s19f.LOCAL_TASKS.exists():
        rows = read_csv(s19f.LOCAL_TASKS)
        completed = sum(1 for r in rows if r.get("status") == "completed")
        total = len(rows)
        print(f"tasks 19F: {completed}/{total} concluidas (getInfo sincrono)")
    else:
        rows = [{
            "task_id": s19f.NA, "task_type": "synchronous_getInfo", "status": "nao_iniciado",
            "patches_processados": "0", "observacao": "nenhuma task local; execute run_susc_19f_mapbiomas_gee.py",
        }]
        print("tasks 19F: nenhuma task local registrada")
    write_csv(s19f.STATUS_TASKS, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
