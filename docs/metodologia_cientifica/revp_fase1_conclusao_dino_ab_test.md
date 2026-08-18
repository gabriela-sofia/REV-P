# Fase 1 concluida -- DINO nao agrega valor incremental ao modelo fisico (com o dado atual)

**Status**: FASE1_CONCLUIDA_COM_PROVA -- decisao tomada com numero real, nao suposicao.

## Cadeia de evidencia (v1r5 -> v1r6)

1. **v1r5** (`revp_v1r5_dino_v12_ab_test.py`): join espacial de `dataset_v12_final.csv`
   (n=278, o pool mais forte) contra os 23 patches Recife com embedding DINO real
   produziu n=109 (58 pos / 51 neg). Orcamento de EPV nao comportou as 6 features
   fisicas completas + DINO; reduzido para as 3 mais fortes por evidencia real do
   proprio v12 (`rain_decay_index_api_chirps`, `twi_dinf`, `slope_deg`). LRT ingenuo
   Modelo A (fisico) vs Modelo B (fisico+DINO): **p=0.0048** -- pareceria significante.
2. Mas v1r5 descobriu, ao auditar o proprio join, que `dino_pca1`/`dino_pca2` sao
   calculados **por patch Sentinel, nao por ponto**: os 109 pontos caem em so 23
   patches unicos, ate 10 pontos compartilhando o vetor DINO identico. Isso e
   pseudorreplicacao -- o LRT ingenuo tratou 109 "observacoes DINO" como
   independentes quando na pratica sao 23.
3. **v1r6** (`revp_v1r6_dino_v12_cluster_robust_sensitivity.py`) corrigiu isso com
   erro-padrao cluster-robusto (sandwich, clusterizado por patch_id, com correcao de
   pequena amostra) e um teste de Wald conjunto para os 2 componentes DINO: **p=0.1752**
   -- ja nao e significante. A correlacao descritiva por patch (n=23, DINO PCA fixo por
   patch x fracao de pontos positivos naquele patch) confirma: rho=0.065 (p=0.77) e
   rho=-0.026 (p=0.91) -- essencialmente zero.

## Conclusao

O sinal de v1r5 era, pelo menos em grande parte, um artefato de pseudorreplicacao, nao
evidencia real de conteudo visual preditivo do DINO. Uma vez que a nao-independencia
entre pontos do mesmo patch e contabilizada corretamente, **o teste A/B nao suporta
promover DINO a feature do score de suscetibilidade**.

Isso fecha a pergunta central do `PLANO_ACAO_produto_v1.md` com a primeira das "duas
saidas possiveis" ja previstas no plano (secao 1): **modelo do produto = Firth-so-fisica
(v12, LOO-AUC=0.6781, ja documentado e validado); DINO vira evidencia visual explicavel
na interface, nunca input do score.**

Nota de reprodutibilidade: o pacote `firthlogist` usado nas sessoes anteriores exige
Python <3.11 e nao esta disponivel neste ambiente (Python 3.12). A regressao Firth foi
reimplementada localmente e validada contra os coeficientes e log-verossimilhanca ja
publicados do v12 antes de qualquer uso (ver `revp_v1r5_dino_v12_ab_test.py`,
docstring de `fit_firth`).

## Ressalva sobre generalizacao

Esta conclusao vale para o n e a cobertura de patches disponiveis agora (23 patches
Recife com embedding DINO real). Se mais patches independentes (nao s5o Sentinel scenes
ja usadas) forem processados no futuro, o teste pode ser refeito com mais poder
estatistico -- os scripts v1r5/v1r6 sao reexecutaveis sem alteracao (apontando as
mesmas variaveis de ambiente para um `dino_recife_sedec_all_embeddings_*.csv` maior).

## Arquivos desta cadeia

- `scripts/dino/revp_v1r5_dino_v12_ab_test.py`, `datasets/dino_v12_ab_comparison_summary_v1r5.csv`,
  `datasets/dino_v12_ab_firth_model_coefs_v1r5.csv`, `docs/metodologia_cientifica/revp_v1r5_dino_v12_ab_test.md`
- `scripts/dino/revp_v1r6_dino_v12_cluster_robust_sensitivity.py`,
  `datasets/dino_v12_cluster_robust_sensitivity_v1r6.csv`,
  `docs/metodologia_cientifica/revp_v1r6_dino_v12_cluster_robust_sensitivity.md`
