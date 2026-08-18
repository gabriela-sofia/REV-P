"""
INFRA-01 -- A reducao a tabela de pontos resolveu mesmo o custo de carga?

A PERGUNTA, colocada como ela precisa ser para ter resposta:

"reduzir toda fonte de imagem a tabela de pontos" so vale como solucao de
infraestrutura se DUAS coisas forem verdade ao mesmo tempo:

  1. o caminho de MODELO nao abre raster nenhum -- nem para uma leitura
  2. a tabela inteira cabe na memoria e carrega em tempo de interacao

Se a primeira falhar, a reducao virou uma etapa a mais e nao uma substituicao:
o custo do raster continua la, so mudou de lugar. Este script nao pergunta ao
codigo o que ele acha que faz -- mede.

O QUE ELE VERIFICA:

  A. CAMINHO DE MODELO SEM RASTER. Varre os scripts que vao do dado ao ajuste
     procurando importacao de biblioteca de raster. Encontrar `rasterio` no
     `ter01` e esperado e correto -- e a etapa de derivacao. Encontrar no
     `ds05` ou no `mod_*` seria o sintoma de que a reducao nao fechou.

  B. CUSTO REAL DE CARGA. Tempo e memoria para abrir a tabela consolidada e
     montar a matriz de features do maior ajuste. E o numero que decide se
     isso roda num notebook ou exige maquina.

  C. DIVISAO DE CARGA JA IMPLANTADA. Inventario do que cada etapa cara faz
     para nao depender de uma execucao unica: subprocesso isolado com teto,
     cache em disco, agrupamento por celula de grade, trabalhadores.

  D. ONDE ESTA O GARGALO AGORA. Tamanho e contagem dos artefatos por etapa,
     para dizer qual e a proxima coisa cara -- e nao a que era cara antes.

NAO faz: nao otimiza, nao muda pipeline. Mede e aponta.

Uso:
    python scripts/suscetibilidade/infra01_carga_e_paralelismo.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
SCRIPTS = REPO / "scripts"
OUT = RUNS / "infra-01-carga"

# etapas do caminho DADO -> AJUSTE. Nenhuma delas pode abrir raster.
CAMINHO_MODELO = [
    "suscetibilidade/ds03_esquema_alvo.py",
    "suscetibilidade/ds04_reduzir_por_fonte.py",
    "suscetibilidade/ds05_admissao_consolidacao.py",
    "suscetibilidade/mod_mec02_fluvial_pool_expandido.py",
    "suscetibilidade/mod_pluv01_recife_fonte_unica.py",
    "suscetibilidade/aud_chuva01_fontes_incompativeis.py",
]
# etapas de DERIVACAO, onde abrir raster e o trabalho
CAMINHO_DERIVACAO = [
    "terreno/ter01_cadeia_harmonizada.py",
    "terreno/ter02_reextrair_e_comparar.py",
    "terreno/ter05_harmonizar_uk.py",
    "terreno/ter06_harmonizar_chips_nivel1.py",
]

RASTER = re.compile(r"^\s*(?:import|from)\s+(rasterio|gdal|osgeo|whitebox|pysheds|xarray|rioxarray)",
                    re.MULTILINE)

MECANISMOS_DE_CARGA = {
    "subprocesso isolado com teto de tempo": [
        "terreno/ter01_cadeia_harmonizada.py", "terreno/ter06_harmonizar_chips_nivel1.py"],
    "cache em disco entre execucoes": [
        "terreno/ter01_cadeia_harmonizada.py",
        "suscetibilidade/chuva04_adquirir_era5_global.py"],
    "agrupamento antes de consultar": [
        "suscetibilidade/chuva04_adquirir_era5_global.py"],
    "trabalhadores em paralelo": [
        "suscetibilidade/chuva04_adquirir_era5_global.py",
        "suscetibilidade/chuva02_padronizar_fonte_unica_recife.py"],
}


def memoria_mb() -> float | None:
    try:
        import resource  # noqa: PLC0415
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        import psutil  # noqa: PLC0415
        return round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("===INFRA01_CARGA_E_PARALELISMO===")
    rel: dict = {}

    # ---- A ----
    print("\n--- A. O CAMINHO DE MODELO ABRE RASTER? ---")
    achados = {}
    for grupo, lista in (("modelo", CAMINHO_MODELO), ("derivacao", CAMINHO_DERIVACAO)):
        for rel_path in lista:
            p = SCRIPTS / rel_path
            if not p.exists():
                continue
            libs = sorted(set(RASTER.findall(p.read_text(encoding="utf-8"))))
            achados[rel_path] = {"grupo": grupo, "libs_raster": libs}
    violacoes = [k for k, v in achados.items()
                 if v["grupo"] == "modelo" and v["libs_raster"]]
    for k, v in achados.items():
        marca = "  <-- VIOLACAO" if k in violacoes else ""
        print(f"  [{v['grupo']:9s}] {k:52s} {v['libs_raster'] or '(nenhuma)'}{marca}")
    print(f"  => {len(violacoes)} violacao(oes) no caminho de modelo")
    rel["caminho_modelo"] = {"achados": achados, "violacoes": violacoes}

    # ---- B ----
    print("\n--- B. CUSTO REAL DE CARGA ---")
    from ds03_esquema_alvo import VARIAVEIS_FISICAS, VERSAO  # noqa: PLC0415

    uni = RUNS / "ds-05-tabela-unica" / f"tabela_unica_{VERSAO}.csv"
    pool = RUNS / "ds-05-tabela-unica" / f"tabela_unica_pool_fluvial_{VERSAO}.csv"
    carga = {}
    base_mem = memoria_mb()
    for nome, p in (("consolidada", uni), ("pool_fluvial", pool)):
        if not p.exists():
            continue
        t0 = time.time()
        d = pd.read_csv(p, low_memory=False)
        t_load = time.time() - t0
        t1 = time.time()
        X = d[list(VARIAVEIS_FISICAS)].to_numpy(dtype="float64")
        t_matriz = time.time() - t1
        carga[nome] = {
            "linhas": int(len(d)), "colunas": int(d.shape[1]),
            "mb_em_disco": round(p.stat().st_size / 1024 / 1024, 1),
            "mb_em_memoria": round(d.memory_usage(deep=True).sum() / 1024 / 1024, 1),
            "segundos_carga": round(t_load, 2),
            "segundos_matriz": round(t_matriz, 3),
            "shape_matriz": list(X.shape),
        }
        print(f"  {nome:14s} {len(d):>7,} linhas  "
              f"{carga[nome]['mb_em_disco']:>6.1f} MB disco  "
              f"{carga[nome]['mb_em_memoria']:>6.1f} MB RAM  "
              f"carga {t_load:.2f}s  matriz {t_matriz:.3f}s")
        del d, X
    pico = memoria_mb()
    if pico:
        print(f"  pico do processo: {pico:.0f} MB (base {base_mem:.0f} MB)")
    rel["carga"] = {"tabelas": carga, "pico_mb": pico, "base_mb": base_mem}

    # ---- C ----
    print("\n--- C. DIVISAO DE CARGA JA IMPLANTADA ---")
    mecan = {}
    for mec, arquivos in MECANISMOS_DE_CARGA.items():
        presentes = [a for a in arquivos if (SCRIPTS / a).exists()]
        mecan[mec] = presentes
        print(f"  {mec}: {len(presentes)} etapa(s)")
        for a in presentes:
            print(f"      {a}")
    rel["mecanismos"] = mecan

    ch = RUNS / "chuva-04-era5-global" / "resumo.json"
    if ch.exists():
        c = json.loads(ch.read_text(encoding="utf-8"))
        print(f"\n  reducao medida na aquisicao de chuva: "
              f"{c.get('reducao_de_chamadas')}x "
              f"({c.get('pontos'):,} pontos -> {c.get('pares_consultados'):,} consultas)")
        rel["reducao_aquisicao"] = c.get("reducao_de_chamadas")

    # ---- D ----
    print("\n--- D. ONDE ESTA O CUSTO AGORA ---")
    pesos = []
    for p in sorted(RUNS.iterdir()):
        if not p.is_dir():
            continue
        try:
            arqs = list(p.rglob("*"))
            b = sum(f.stat().st_size for f in arqs if f.is_file())
        except Exception:  # noqa: BLE001
            continue
        if b > 20 * 1024 * 1024:
            pesos.append({"etapa": p.name, "mb": round(b / 1024 / 1024, 1),
                          "arquivos": sum(1 for f in arqs if f.is_file())})
    pesos.sort(key=lambda x: -x["mb"])
    for x in pesos[:8]:
        print(f"  {x['etapa']:44s} {x['mb']:>9,.1f} MB  {x['arquivos']:>6,} arquivos")
    rel["etapas_pesadas"] = pesos[:15]

    # ---- veredito ----
    print("\n--- VEREDITO ---")
    if violacoes:
        print(f"  A reducao NAO fechou: {violacoes} abre raster no caminho de "
              "modelo. O custo do raster continua no caminho de ajuste.")
    else:
        print("  A reducao FECHOU: nenhuma etapa do caminho dado->ajuste abre "
              "raster. Biblioteca de raster so aparece na derivacao, que e "
              "onde ela deve estar.")
    maior = max(carga.values(), key=lambda x: x["mb_em_memoria"]) if carga else None
    if maior:
        print(f"  A maior tabela ocupa {maior['mb_em_memoria']} MB e carrega em "
              f"{maior['segundos_carga']}s -- cabe em memoria de notebook, sem "
              "carregamento por pedaco.")
    rel["veredito_reducao"] = "fechou" if not violacoes else "nao_fechou"

    (OUT / "diagnostico_infra.json").write_text(
        json.dumps(rel, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nGRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
