# Reprodutibilidade — SUSC-20 Recife

O script de modelagem original (`pipeline_v12_primary.py`) apontava pra um caminho
local (`local_runs/recife_modelo_v12_extracao_final/...`, via a constante
`LOCAL_RUNS_ROOT = "<local_runs_root>"` — um placeholder, nem um caminho de verdade)
que não existe no repositório — rodar a partir de um clone limpo dava erro de arquivo
não encontrado.

A versão consolidada, no mesmo caminho
`outputs_public/data/linha_causal/susc_20c_modelagem_validacao_estatistica_rigorosa_recife/scripts/pipeline_v12_primary.py`,
substitui esse script: mesma metodologia exata (screening univariado Mann-Whitney,
Firth penalizada multivariada, bootstrap N=1000, AUC preditivo LOO + 5-fold repetido
50x), mas lendo e escrevendo em caminhos que existem de verdade no repo.

**Verificação** (executada nesta sessão, Python 3.10 + scikit-learn 1.5.2 +
firthlogist 0.5.0): rodei o script consolidado contra
`outputs_public/data/linha_causal/susc_20a_aquisicao_eventos_reais_recife/dataset/dataset_eventos_features_v12_final.csv`
(o dataset que já estava publicado) e comparei com os resultados já commitados em
`susc_20c_modelagem_validacao_estatistica_rigorosa_recife/results/`.

Os 5 arquivos de resultado saíram **byte a byte idênticos** aos que já estavam no
repositório (`git status` não reporta nenhuma mudança em `results/` depois da
execução):

| Arquivo | Resultado |
|---|---|
| `primaria_v12_univariate_mannwhitney.csv` | idêntico |
| `primaria_v12_firth_multivariate_coefs.csv` | idêntico |
| `primaria_v12_bootstrap_coefs.csv` | idêntico |
| `primaria_v12_predictive_auc.json` | idêntico |
| `all_reports_v12_primary.json` | idêntico |

Números conferidos contra o README e o relatório v12 master: dataset com 278 linhas
(154 pos / 124 neg), **n=269** efetivamente usados após `dropna` nas 6 features
(145 pos / 124 neg, EPV=20,67), **LOO-AUC=0,6781**, `skf_auc_mean=0,6747`
(std 0,0115), e os 6 coeficientes de Firth — incluindo
`rain_decay_index_api_chirps` coef=0,9896, IC [0,6133; 1,4231], p<0,0001, o único
com IC que não cruza zero junto com `twi_dinf` (0,2786, p=0,0461).

**Ambiente necessário**: Python 3.10, `scikit-learn<1.6` (fixado no
`environment.yml` — versões mais novas removem `_validate_data`, uma API
interna que o `firthlogist` usa, e o fit quebra com `AttributeError`) e
`firthlogist`. O `firthlogist` publica wheels só para Python <3.11, então o
pino de `python=3.10` do `environment.yml` não é preferência, é requisito.

**Escopo desta correção**: só o passo de modelagem (susc_20c). Os 3 scripts
que constroem o dataset a partir de fontes brutas (`fetch_rain_leadA_positives.py`,
`fetch_rain_leadc.py`, `build_v12_dataset.py`, em susc_20b) também apontam pro
mesmo `local_runs/` inexistente, mas dependem de acesso a API externa de chuva —
não dá pra reverificá-los aqui sem essas credenciais. Ficam registrados como
histórico de como o dataset final foi construído, não como algo re-executável
neste ambiente.

**Como rodar**:

```bash
python outputs_public/data/linha_causal/susc_20c_modelagem_validacao_estatistica_rigorosa_recife/scripts/pipeline_v12_primary.py
```
