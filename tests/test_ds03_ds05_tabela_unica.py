"""Testes da tabela unica: contrato (ds03), reducao (ds04) e admissao (ds05).

O que estes testes protegem, e por que cada um existe:

o risco desta etapa nao e o script quebrar -- e ele produzir uma tabela
plausivel e errada. Uma coluna com duas fontes dentro, um negativo promovido a
observado, um ponto contado duas vezes: nada disso levanta excecao. Por isso os
testes checam INVARIANTES do dado produzido, e nao so se o codigo roda.

Os artefatos vivem em `local_runs/`, que e git-ignored. Quando faltarem, o
teste marca skip com o comando que os gera, em vez de falhar -- falhar por
ausencia de dado local confundiria quem clonasse o repositorio.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "local_runs"
RED = RUNS / "ds-04-reducao"
UNI = RUNS / "ds-05-tabela-unica"
ESQ = RUNS / "ds-03-esquema"

sys.path.insert(0, str(ROOT / "scripts" / "suscetibilidade"))
from ds03_esquema_alvo import (  # noqa: E402
    CLASSE, CLASSES_TREINO, COLUNAS, DOMINIOS, OBRIGATORIAS,
    VARIAVEIS_TERRENO, VERSAO, contrato,
)


def _exige(p: Path, comando: str) -> None:
    if not p.exists():
        pytest.skip(f"{p.relative_to(ROOT)} ausente -- gere com: {comando}")


@pytest.fixture(scope="module")
def unica() -> pd.DataFrame:
    p = UNI / f"tabela_unica_{VERSAO}.csv"
    _exige(p, "python scripts/suscetibilidade/ds05_admissao_consolidacao.py")
    return pd.read_csv(p, low_memory=False)


# --------------------------------------------------------------------------
# contrato
# --------------------------------------------------------------------------

def test_contrato_nao_tem_coluna_de_valor_sem_procedencia():
    """A regra de ouro do ds03, verificada e nao so escrita no docstring."""
    c = contrato()
    nomes = {x["nome"] for x in c["colunas"]}
    assert {"elevation_m", "slope_deg", "hand_m", "twi_dinf"} <= nomes
    # terreno tem cadeia_terreno; chuva tem fonte_chuva
    assert "cadeia_terreno" in nomes
    assert "fonte_chuva" in nomes
    assert set(c["variaveis_terreno"]) <= set(c["variaveis_fisicas"])


def test_contrato_gerado_bate_com_o_modulo():
    p = ESQ / f"esquema_alvo_{VERSAO}.json"
    _exige(p, "python scripts/suscetibilidade/ds03_esquema_alvo.py")
    gravado = json.loads(p.read_text(encoding="utf-8"))
    assert gravado == contrato(), (
        "o contrato gravado divergiu do modulo: rode o ds03 de novo")


def test_esquema_alvo_nao_contem_dado():
    p = ESQ / f"esquema_alvo_{VERSAO}.csv"
    _exige(p, "python scripts/suscetibilidade/ds03_esquema_alvo.py")
    linhas = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(linhas) == 1, "o arquivo de contrato ganhou linha de dado"
    assert linhas[0].split(",") == list(COLUNAS)


# --------------------------------------------------------------------------
# reducao
# --------------------------------------------------------------------------

def test_toda_fonte_reduzida_respeita_o_contrato():
    _exige(RED, "python scripts/suscetibilidade/ds04_reduzir_por_fonte.py")
    achou = False
    for p in sorted(RED.glob("*.csv")):
        d = pd.read_csv(p, low_memory=False)
        achou = True
        assert list(d.columns) == list(COLUNAS), f"{p.name} fora do contrato"
        for col, dom in DOMINIOS.items():
            fora = set(d.loc[d[col].notna(), col].unique()) - set(dom)
            assert not fora, f"{p.name}: {col} tem valor fora do dominio: {fora}"
        fora_cls = set(d["classe"].dropna().astype(int).unique()) - set(CLASSE)
        assert not fora_cls, f"{p.name}: classe fora do dominio: {fora_cls}"
    assert achou, "nenhuma fonte reduzida encontrada"


def test_curitiba_nao_entra_como_negativo_observado():
    """Bloco 3 do checklist: o negativo de Curitiba e ausencia de registro.

    O campo de origem diz que houve chamado no 156 e o assunto nao era
    hidrologico. Isso nao e "a area foi analisada e nao havia inundacao".
    """
    p = RED / "curitiba.csv"
    _exige(p, "python scripts/suscetibilidade/ds04_reduzir_por_fonte.py --fonte curitiba")
    d = pd.read_csv(p, low_memory=False)
    neg = d[d["classe"] == 0]
    assert len(neg) > 0
    assert set(neg["nivel_negativo"].unique()) == {"ausencia"}


def test_recife_entra_em_cadeia_nativa_e_curitiba_em_wbt30():
    """A resolucao segue o mecanismo, nao a regiao (decisao de 12/08/2026)."""
    for fonte, esperado in (("recife", "nativa_10m"), ("curitiba", "wbt30")):
        p = RED / f"{fonte}.csv"
        _exige(p, "python scripts/suscetibilidade/ds04_reduzir_por_fonte.py")
        d = pd.read_csv(p, low_memory=False)
        assert set(d["cadeia_terreno"].unique()) == {esperado}
    assert set(pd.read_csv(RED / "recife.csv")["mecanismo"].unique()) == {"PLUVIAL_URBANO"}


def test_nenhuma_variavel_fisica_foi_imputada_com_zero():
    """Ausencia e nulo declarado. Zero em HAND ou chuva seria um valor real."""
    _exige(RED, "python scripts/suscetibilidade/ds04_reduzir_por_fonte.py")
    for p in sorted(RED.glob("*.csv")):
        d = pd.read_csv(p, low_memory=False)
        for v in ("hand_m", "rain_max_24h", "rain_decay_index"):
            # zero e valor legitimo isolado; o que denuncia imputacao e a
            # coluna INTEIRA em zero, ou zero exatamente onde a fonte nao tem
            # a variavel.
            if d[v].notna().any():
                assert not (d[v].fillna(0) == 0).all(), f"{p.name}: {v} toda zero"


# --------------------------------------------------------------------------
# admissao e consolidacao
# --------------------------------------------------------------------------

def test_ponto_id_e_unico_na_tabela_consolidada(unica):
    assert unica["ponto_id"].is_unique, (
        "ponto_id repetido: dois pontos com o mesmo identificador seriam "
        "contados duas vezes em qualquer agregacao")


def test_nenhum_ponto_existe_em_duas_fontes(unica):
    """Passo 1.4 do checklist. Coincidencia geografica, nao so de id."""
    g = unica.copy()
    g["celula"] = (
        (g["lat"] / 1e-4).round().astype("Int64").astype(str) + "_"
        + (g["lon"] / 1e-4).round().astype("Int64").astype(str))
    cruzadas = g.groupby("celula")["fonte"].nunique()
    assert int((cruzadas > 1).sum()) == 0, (
        f"{int((cruzadas > 1).sum())} celulas de ~11 m aparecem em mais de "
        "uma fonte")


def test_toda_rejeicao_tem_motivo_nomeado(unica):
    """Passo 1.3: 'o restante fica como proveniencia, com o motivo nomeado'."""
    rej = unica[~unica["admitido"]]
    assert len(rej) > 0, "nenhuma rejeicao: o filtro de admissao nao rodou?"
    assert rej["motivo_rejeicao"].notna().all()
    assert (rej["motivo_rejeicao"].astype(str).str.strip() != "").all()
    adm = unica[unica["admitido"]]
    assert (adm["motivo_rejeicao"].fillna("") == "").all(), (
        "linha admitida com motivo de rejeicao preenchido")


def test_a_conta_fecha_por_fonte(unica):
    """entrada = admitido + rejeitado, em cada fonte. E o relatorio de E2."""
    p = UNI / "relatorio_contagem_por_fonte.csv"
    _exige(p, "python scripts/suscetibilidade/ds05_admissao_consolidacao.py")
    t = pd.read_csv(p)
    real = unica.groupby("fonte").agg(
        entrada=("ponto_id", "size"), admitido=("admitido", "sum"))
    for _, r in t.iterrows():
        assert int(r["entrada"]) == int(real.loc[r["fonte"], "entrada"])
        assert int(r["admitido"]) == int(real.loc[r["fonte"], "admitido"])
        assert int(r["admitido"]) <= int(r["entrada"])


def test_admitido_satisfaz_os_tres_criterios(unica):
    adm = unica[unica["admitido"]]
    assert adm["aoi"].notna().all()
    assert adm["grupo_cv"].notna().all()
    assert adm["lat"].notna().all() and adm["lon"].notna().all()
    assert not (adm["cadeia_terreno"].isin(["global", "ausente"])).any(), (
        "cadeia global admitida: e outro instrumento, nao outra resolucao")
    assert adm[list(VARIAVEIS_TERRENO)].notna().all(axis=1).all()
    assert adm["classe"].isin(CLASSES_TREINO).all()


def test_pool_fluvial_e_uma_cadeia_so_e_um_mecanismo_so(unica):
    pool = unica[unica["elegivel_pool_fluvial"]]
    assert len(pool) > 0
    assert set(pool["cadeia_terreno"].unique()) == {"wbt30"}
    assert set(pool["mecanismo"].unique()) == {"FLUVIAL_ENXURRADA"}
    assert "recife" not in set(pool["fonte"].unique()), (
        "Recife e pluvial urbano; entrar no pool fluvial desfaria a separacao "
        "por mecanismo")


def test_ufo_nunca_entra_num_modelo_por_mecanismo(unica):
    """A propria base declara drivers pluvial, fluvial e mare sem separacao."""
    ufo = unica[unica["fonte"] == "ufo"]
    if ufo.empty:
        pytest.skip("fonte ufo nao reduzida neste ambiente")
    assert set(ufo["mecanismo"].unique()) == {"MISTO_NAO_SEPARAVEL"}
    assert int(ufo["elegivel_pool_fluvial"].sum()) == 0


def test_todo_valor_de_chuva_tem_fonte_declarada(unica):
    """A coluna pode misturar produtos; o que nao pode e misturar em silencio.

    A direcao que importa e valor -> fonte. O contrario nao vale como regra:
    Recife tem 9 pontos sem valor de chuva e com `rain_data_source` gravado,
    porque a fonte estava escolhida e a serie e que faltou naquele dia.
    """
    com_chuva = unica[unica["rain_max_24h"].notna()
                      | unica["rain_decay_index"].notna()]
    if com_chuva.empty:
        pytest.skip("nenhuma linha com chuva")
    assert (com_chuva["fonte_chuva"] != "ausente").all(), (
        "valor de chuva sem fonte declarada")


def test_recife_continua_registrado_com_mistura_de_fonte_de_chuva(unica):
    """Guarda de regressao sobre o achado do aud_chuva01.

    Recife carrega CHIRPS v2 e ERA5-Land na mesma coluna, e a fonte esta
    associada ao rotulo. Se um dia a mistura for resolvida, este teste falha e
    obriga a atualizar a auditoria junto -- que e o comportamento desejado.
    """
    rec = unica[(unica["fonte"] == "recife") & unica["rain_max_24h"].notna()]
    if rec.empty:
        pytest.skip("recife nao reduzido neste ambiente")
    fontes = set(rec["fonte_chuva"].unique())
    assert len(fontes) > 1, (
        "a mistura de fonte de chuva em Recife sumiu: atualize "
        "aud_chuva01 e a documentacao antes de relaxar este teste")

    aud = RUNS / "aud-chuva-01" / "auditoria_fonte_de_chuva.json"
    _exige(aud, "python scripts/suscetibilidade/aud_chuva01_fontes_incompativeis.py")
    m = json.loads(aud.read_text(encoding="utf-8"))
    assert "recife" in m["fontes_com_mistura"]


def test_manifesto_registra_hash_do_consolidado():
    p = UNI / f"manifesto_{VERSAO}.json"
    _exige(p, "python scripts/suscetibilidade/ds05_admissao_consolidacao.py")
    m = json.loads(p.read_text(encoding="utf-8"))
    alvo = f"tabela_unica_{VERSAO}.csv"
    assert alvo in m["arquivos"]
    assert len(m["arquivos"][alvo]["sha256"]) == 64
    assert m["admitidos"] <= m["entrada"]
    assert m["pool_fluvial"] <= m["admitidos"]


def test_consolidacao_e_deterministica():
    """Rodar de novo com a mesma entrada tem de dar o mesmo hash."""
    p = UNI / f"manifesto_{VERSAO}.json"
    _exige(p, "python scripts/suscetibilidade/ds05_admissao_consolidacao.py")
    alvo = f"tabela_unica_{VERSAO}.csv"
    antes = json.loads(p.read_text(encoding="utf-8"))["arquivos"][alvo]["sha256"]
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/suscetibilidade/ds05_admissao_consolidacao.py")],
        cwd=ROOT, check=True, capture_output=True)
    depois = json.loads(p.read_text(encoding="utf-8"))["arquivos"][alvo]["sha256"]
    assert antes == depois


def test_nada_de_local_runs_esta_staged():
    staged = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True)
    assert "local_runs/" not in staged
