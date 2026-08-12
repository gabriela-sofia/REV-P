# Documento de Planejamento do Projeto — Entrega 01 (29/08/2026)

Experiência Criativa: Projeto Transformador II — BCC/PUCPR.
Formato IEEEtran `conference`, português, 3 páginas (limite do template).

**Versão atual**: v2 (2026-08-11). A v1 (2026-08-09) está preservada em
`main_v1_2026-08-09.tex` e foi escrita antes de a frente externa (EXT-UK, CEMS,
mod-neg-01, mod-serra-01) produzir resultado. A v2 corrige afirmações que aquela
versão fazia sobre Petrópolis e sobre o negativo brasileiro.

## Arquivos

| Arquivo | Papel |
|---|---|
| `main.tex` | fonte LaTeX v2 (compila em Overleaf sem ajuste; IEEEtran já é padrão lá) |
| `main.pdf` | saída compilada, 3 páginas |
| `main_v1_2026-08-09.tex` | versão anterior, preservada para comparação |
| `fig/fig2_datasets.pdf` | Figura 2, vetorial, gerada a partir dos dados reais do projeto |
| `fig/make_fig2.py` | script que gera a Figura 2 (reprodutível) |
| `NOTA_v1_para_v2.md` | o que mudou, por quê, e o que ficou pendente |

A Figura 1 é TikZ inline no `main.tex` — vetorial, sem arquivo externo.

## Compilar

```
pdflatex main && pdflatex main
```

Para regenerar a Figura 2 (lê artefatos reais do projeto):

```
python fig/make_fig2.py
```

O script resolve os caminhos a partir da própria localização; se o repositório
for movido, defina `REVP_ROOT` e `PROJETO_ROOT` no ambiente.

## Controle de espaço (importante ao editar)

O documento está no limite de 3 páginas. O parâmetro mais sensível é a largura
da Figura 1:

```latex
\resizebox{0.52\textwidth}{!}{ ... }   % linha ~103 do main.tex
```

Valores acima de `0.54` empurram a bibliografia para uma quarta página **no
ambiente de compilação local usado aqui**, que não tem os padrões de hifenização
do português carregados. No Overleaf, com hifenização ativa, há folga: se quiser
a figura maior, suba para `0.58`–`0.60` e reconfira a contagem de páginas antes
de entregar. Os outros dois controles são a altura da Figura 2 (`figsize` em
`make_fig2.py`, hoje `1.44`) e o corpo da bibliografia (`\fontsize{5.7}{6.2}`).

## Rastreabilidade dos números citados no documento

| Afirmação no texto | Origem |
|---|---|
| Ablação: 0,8855 / 0,4834 / 0,4689 | relatório de ablação da Entrega 01 |
| Recife: 154 pos. / 124 neg.; n = 278 | `PROJETO/local_runs/recife_modelo_v12_extracao_final/dataset_v12_final.csv` |
| Curitiba: 1.045 pos. / 426 neg.; 1.471 unidades | `outputs_public/data/susc_20k_.../registries/v20n_dataset_curitiba_features_v2.csv` |
| Piloto UK: 7.476 pontos (3.738/3.738), 201 eventos independentes | `local_runs/mod-uk-01-firth/resumo_rnl.json`; `ext_balanco_e_lacunas_por_regiao_v1.md` §2.2 |
| Critérios N1–N4 do negativo por exclusão (400 m, cobertura do solo) | `ext_uk_adjudicacao_negativo_v1.md` §4 |
| CEMS: 25.249 pontos, 119 AOIs (22 serra + 97 planície) | `local_runs/mod-serra-01/resultado.json` |
| EMSR720 (RS): 216,55 km², razão 5,94:1 | `ext_balanco_e_lacunas_por_regiao_v1.md` §2.1 |
| Queda de 0,159 na aplicação cruzada; 0,7798 no sentido inverso | `local_runs/mod-neg-01/resumo.json` |
| Contraste de HAND: 2,95 m em planície, 27,86 m em serra | `local_runs/mod-serra-01/resultado.json` |
| Colapso prospectivo de 2026 em Curitiba (7 diagnósticos) | `PLANO_ACAO_produto_v1.md`, SUSC-20O a 20W |
| Firth 0,056 s / EBM 25,3 s; máquina 10 núcleos, 15,5 GB | `docs/ambiente_treino_susc.md` |
| Marcos M1–M7 e datas da disciplina | `docs/cronograma_cientifico_planejamento_2026.md` |

## O que falta preencher antes de entregar

- Turma e número da equipe (marcados em vermelho no cabeçalho).
- E-mail institucional `@pucpr.edu.br`.
- Decisão sobre a pergunta ao professor registrada em `NOTA_v1_para_v2.md` §4.

## Aderência às regras fixas do projeto

O documento declara, em texto, figura e tabela, que a camada orbital
(Sentinel/\emph{embeddings}) é auxiliar e nunca entra no escore, que nenhuma
variável derivada do rótulo é admitida como preditor, e que validação científica
e otimização de desempenho são etapas separadas. A hierarquia do negativo
(observação > exclusão > ausência) é declarada por linha, e o portão de negativo
formal é apresentado como **aberto** para Recife, Curitiba e Petrópolis — que é
o estado real registrado em `ext_balanco_e_lacunas_por_regiao_v1.md`.
