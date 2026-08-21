"""Diagnostico exploratorio: EBM (Explainable Boosting Machine) sobre o dataset
real v12 de Recife (SUSC-20C/20D), com as MESMAS 6 features causais do Firth e
comparado contra o LOO-AUC ja publicado (0,6781; n=278, 154 pos / 124 neg).

Papel deste script na linhagem do projeto
------------------------------------------
Mesmo papel que o GBM teve em Curitiba (SUSC-20U): um DIAGNOSTICO, nao uma
proposta de rota primaria. Nenhuma feature nova, nenhuma variavel orbital,
nenhum dado derivado do rotulo -- so as 6 features fisico-hidrologicas de
sempre. A rota primaria continua sendo o Firth interpretavel; isso so
responde "existe nao-linearidade real capturavel por um modelo ainda
interpretavel (shape functions), nas mesmas features?".

O que este script NAO faz
--------------------------
Nao promove EBM a rota primaria. Nao adiciona feature nova. Nao decide
monotonicidade (isso e outra rodada, so depois de olhar as curvas cruas
geradas aqui). Nao faz grid search de hiperparametro nesta primeira rodada
(config default, mesmo criterio do SUSC-20U pro GBM).

Como este script se comporta se algo faltar
---------------------------------------------
Ele NUNCA quebra com um traceback cru sem contexto. Cada etapa atualiza um
relatorio incremental; se qualquer coisa falhar (dataset nao encontrado,
coluna faltando, erro de biblioteca), o relatorio completo -- com a etapa
exata onde parou e o motivo -- e impresso no final e salvo em
local_runs/ebm_diagnostico_recife_v12/relatorio_execucao.json. Isso existe
pra evitar rodadas de ida-e-volta: cole o relatorio inteiro de volta na
conversa, nao so a mensagem de erro.

Saida (quando roda ate o fim)
-------------------------------
local_runs/ebm_diagnostico_recife_v12/
    relatorio_execucao.json       -- relatorio completo, sempre gerado
    shape_functions/<feature>.csv -- a curva aprendida por feature
    plots/<feature>.png           -- grafico de cada curva
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
sys.path.insert(0, str(ROOT / "scripts" / "suscetibilidade"))

FEATURES_CAUSAIS = [
    "slope_deg",
    "hand_m_dinf",
    "twi_dinf",
    "rain_decay_index_api_chirps",
    "rain_peak_residual_orthogonalized",
    "elevation_m",
]

FIRTH_LOO_AUC_PUBLICADO = 0.6781  # SUSC-20C/v12, n=278, 154 pos / 124 neg

CANDIDATOS_LABEL = [
    "label", "event_label", "is_flood_event", "target", "y",
    "evento", "positivo", "flood_event", "event", "is_positive", "class",
]

SEED = 20260805
SAIDA = ROOT / "local_runs" / "ebm_diagnostico_recife_v12"


CAMINHOS_CONHECIDOS = [
    # Caminho real confirmado em 05/08/2026 -- publicado e versionado no git,
    # nao em local_runs/ (que e efemero e ja nao tinha mais o arquivo).
    "outputs_public/data/linha_causal/susc_20a_aquisicao_eventos_reais_recife/dataset/dataset_eventos_features_v12_final.csv",
    # Caminhos antigos mencionados nos relatorios (local_runs/), mantidos como
    # fallback caso um dia voltem a existir.
    "local_runs/recife_modelo_v12_extracao_final/dataset_v12_final.csv",
    "local_runs/recife_modelo_v12_extracao_final/dataset_eventos_features_v12_final.csv",
]


def localizar_dataset(relatorio: dict) -> Path:
    # 1) tenta os caminhos ja conhecidos dos relatorios do projeto -- rapido,
    # sem varredura de disco.
    for relativo in CAMINHOS_CONHECIDOS:
        candidato = ROOT / relativo
        if candidato.exists():
            relatorio["candidatos_dataset_encontrados"] = [str(candidato)]
            relatorio["dataset_localizado_via"] = "caminho_conhecido"
            return candidato

    # 2) fallback: varredura recursiva, mas SO dentro de local_runs/ (nao
    # outputs_public/, que tem centenas de subpastas de resultado e pode
    # deixar a varredura bem lenta sem necessidade -- o dataset bruto do v12
    # vive em local_runs por convencao do projeto, nao em outputs_public).
    relatorio["dataset_localizado_via"] = "varredura_local_runs"
    pasta_local_runs = ROOT / "local_runs"
    candidatos = sorted(set(pasta_local_runs.glob("**/*v12*final*.csv"))) if pasta_local_runs.exists() else []
    relatorio["candidatos_dataset_encontrados"] = [str(c) for c in candidatos]
    if not candidatos:
        raise RuntimeError(
            "Nao achei o dataset v12 nos caminhos conhecidos nem numa varredura "
            "de local_runs/. Rode `Get-ChildItem -Recurse -Filter *v12*.csv` "
            "no PowerShell dentro de REV-P (pode incluir outputs_public/ tambem) "
            "e me diga o caminho real -- eu adiciono em CAMINHOS_CONHECIDOS."
        )
    return candidatos[0]


def identificar_coluna_label(df: pd.DataFrame, relatorio: dict) -> str:
    relatorio["colunas_disponiveis_no_dataset"] = list(df.columns)
    for candidato in CANDIDATOS_LABEL:
        if candidato in df.columns:
            return candidato
    raise RuntimeError(
        "Nao identifiquei a coluna de rotulo automaticamente entre os "
        f"candidatos testados {CANDIDATOS_LABEL}. Colunas reais do dataset "
        f"estao em relatorio['colunas_disponiveis_no_dataset']. Me diga qual "
        "coluna e o rotulo positivo/negativo que eu ajusto o script."
    )


def main() -> dict:
    relatorio: dict = {
        "timestamp": datetime.now().isoformat(),
        "etapa_atual": "iniciando",
    }

    try:
        relatorio["etapa_atual"] = "verificando ambiente"
        from susc_ambiente_compat_common import (
            aplicar_shim_firthlogist_sklearn,
            verificar_ambiente_treino_susc,
        )
        relatorio["ambiente"] = verificar_ambiente_treino_susc()
        aplicar_shim_firthlogist_sklearn()

        from interpret.glassbox import ExplainableBoostingClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold

        relatorio["etapa_atual"] = "localizando dataset"
        caminho = localizar_dataset(relatorio)
        relatorio["dataset_usado"] = str(caminho)

        relatorio["etapa_atual"] = "carregando e validando dataset"
        df = pd.read_csv(caminho)
        relatorio["dataset_shape"] = list(df.shape)

        faltando = [f for f in FEATURES_CAUSAIS if f not in df.columns]
        if faltando:
            relatorio["colunas_disponiveis_no_dataset"] = list(df.columns)
            raise RuntimeError(
                f"Faltam features esperadas no dataset: {faltando}. "
                f"Colunas disponiveis estao em relatorio['colunas_disponiveis_no_dataset']."
            )

        col_label = identificar_coluna_label(df, relatorio)
        relatorio["coluna_label_identificada"] = col_label

        dados = df[FEATURES_CAUSAIS + [col_label]].dropna()
        n_pos = int(dados[col_label].sum())
        n_neg = int(len(dados) - n_pos)
        relatorio["n_apos_remover_incompletos"] = len(dados)
        relatorio["n_positivos"] = n_pos
        relatorio["n_negativos"] = n_neg

        X = dados[FEATURES_CAUSAIS].to_numpy()
        y = dados[col_label].to_numpy().astype(int)

        (SAIDA / "shape_functions").mkdir(parents=True, exist_ok=True)
        (SAIDA / "plots").mkdir(parents=True, exist_ok=True)

        # --- checagem rapida: 5-fold repetido (feedback rapido antes do LOO) ---
        # n_repeats baixo de proposito: medido em 05/08/2026 que 1 fit de EBM
        # neste dataset (n~269) leva ~16s num sandbox fraco (2 nucleos) -- com
        # n_repeats=20 (100 fits) isso passaria de 25min so nesta etapa. 5
        # repeats (25 fits) da um sinal rapido de verdade; se sua maquina for
        # rapida o suficiente, aumente esse numero depois pra mais precisao.
        relatorio["etapa_atual"] = "5-fold repetido (checagem rapida)"
        t0 = time.time()
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
        aucs_5fold = []
        for treino_idx, teste_idx in rskf.split(X, y):
            modelo = ExplainableBoostingClassifier(random_state=SEED, feature_names=FEATURES_CAUSAIS)
            modelo.fit(X[treino_idx], y[treino_idx])
            pred = modelo.predict_proba(X[teste_idx])[:, 1]
            aucs_5fold.append(roc_auc_score(y[teste_idx], pred))
        relatorio["auc_5fold_repetido_20x_media"] = round(float(np.mean(aucs_5fold)), 4)
        relatorio["auc_5fold_repetido_20x_desvio"] = round(float(np.std(aucs_5fold)), 4)
        relatorio["tempo_5fold_segundos"] = round(time.time() - t0, 1)
        print(f"5-fold repetido (20x): {np.mean(aucs_5fold):.4f} +/- {np.std(aucs_5fold):.4f} "
              f"({relatorio['tempo_5fold_segundos']}s)")

        # --- LOO-AUC, a metrica oficial de comparacao com o Firth publicado ---
        relatorio["etapa_atual"] = f"LOO-CV ({len(X)} fits)"
        print(f"Rodando LOO-CV (n={len(X)} fits -- pode levar alguns minutos)...")
        t0 = time.time()
        loo = LeaveOneOut()
        y_real, y_pred = [], []
        for i, (treino_idx, teste_idx) in enumerate(loo.split(X)):
            modelo = ExplainableBoostingClassifier(random_state=SEED, feature_names=FEATURES_CAUSAIS)
            modelo.fit(X[treino_idx], y[treino_idx])
            y_pred.append(modelo.predict_proba(X[teste_idx])[:, 1][0])
            y_real.append(y[teste_idx][0])
            if i == 9:  # depois de 10 iteracoes, da pra estimar o tempo total
                eta_min = (time.time() - t0) / 10 * len(X) / 60
                print(f"  Estimativa de tempo total do LOO-CV: ~{eta_min:.0f} min "
                      f"(baseado nas 10 primeiras iteracoes). Se for longo demais, "
                      f"Ctrl+C interrompe -- o 5-fold acima ja e um sinal valido sozinho.")
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(X)} ({time.time()-t0:.0f}s)")

        loo_auc = roc_auc_score(y_real, y_pred)
        relatorio["loo_auc_ebm"] = round(float(loo_auc), 4)
        relatorio["loo_auc_firth_publicado"] = FIRTH_LOO_AUC_PUBLICADO
        relatorio["diferenca_ebm_menos_firth"] = round(float(loo_auc) - FIRTH_LOO_AUC_PUBLICADO, 4)
        relatorio["tempo_loo_segundos"] = round(time.time() - t0, 0)
        print(f"LOO-AUC (EBM, config default): {loo_auc:.4f}")
        print(f"LOO-AUC (Firth, publicado): {FIRTH_LOO_AUC_PUBLICADO:.4f}")
        print(f"Diferenca: {relatorio['diferenca_ebm_menos_firth']:+.4f}")

        # --- modelo final no dataset inteiro, so pra extrair as curvas ---
        relatorio["etapa_atual"] = "extraindo shape functions"
        modelo_final = ExplainableBoostingClassifier(random_state=SEED, feature_names=FEATURES_CAUSAIS)
        modelo_final.fit(X, y)
        explicacao = modelo_final.explain_global()

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        resumo_features = []
        for idx, feature in enumerate(FEATURES_CAUSAIS):
            dado_feature = explicacao.data(idx)
            nomes = dado_feature["names"]
            scores = np.asarray(dado_feature["scores"])
            valores_x = nomes[:-1] if len(nomes) > len(scores) else nomes

            curva = pd.DataFrame({"valor_feature": valores_x, "contribuicao_log_odds": scores})
            curva.to_csv(SAIDA / "shape_functions" / f"{feature}.csv", index=False)

            plt.figure(figsize=(6, 4))
            plt.step(valores_x, scores, where="post")
            plt.title(f"EBM - forma aprendida: {feature}")
            plt.xlabel(feature)
            plt.ylabel("contribuicao pro log-odds")
            plt.tight_layout()
            plt.savefig(SAIDA / "plots" / f"{feature}.png", dpi=120)
            plt.close()

            resumo_features.append({
                "feature": feature,
                "contribuicao_min": round(float(scores.min()), 4),
                "contribuicao_max": round(float(scores.max()), 4),
                "amplitude": round(float(scores.max() - scores.min()), 4),
            })
        relatorio["resumo_por_feature"] = resumo_features

        relatorio["limitacoes"] = [
            "Diagnostico, nao rota primaria -- Firth/linear continua sendo a rota oficial.",
            "Config default do EBM, sem grid search de hiperparametro.",
            "Sem restricao de monotonicidade nesta rodada -- ver plots/ antes de decidir.",
            "N pequeno -- LOO-AUC e validacao interna, nao validacao externa.",
        ]
        relatorio["etapa_atual"] = "concluido"

    except Exception as exc:  # noqa: BLE001 - queremos capturar tudo pra reportar
        relatorio["etapa_com_falha"] = relatorio.get("etapa_atual")
        relatorio["erro"] = f"{type(exc).__name__}: {exc}"
        relatorio["traceback"] = traceback.format_exc()

    finally:
        SAIDA.mkdir(parents=True, exist_ok=True)
        caminho_relatorio = SAIDA / "relatorio_execucao.json"
        with open(caminho_relatorio, "w", encoding="utf-8") as fh:
            json.dump(relatorio, fh, indent=2, ensure_ascii=False, default=str)

        print("\n" + "=" * 70)
        print("RELATORIO COMPLETO -- cole isso de volta na conversa")
        print("=" * 70)
        print(json.dumps(relatorio, indent=2, ensure_ascii=False, default=str))
        print(f"\n(tambem salvo em {caminho_relatorio})")

    return relatorio


if __name__ == "__main__":
    main()
