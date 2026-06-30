# SUSC-17C7 — Plano de Extração de Features para Patches Candidatos

Status: somente para revisão. `review_only=true`; `trainable=false`; `ground_truth=false`; `score_v6_changed=false`; `score_v7_created=false`; `official_patch_created=false`; `official_patch_link_created=false`; `eligible_for_17b_now=false`.

## O que o 17C6 demonstrou

O SUSC-17C6 demonstrou que 5 patches candidatos podem ser representados por uma grade candidata separada e por vínculos candidatos patch-evento, sem alterar a grade oficial SUSC. O 17C6 também mostrou que apenas a camada `documentary_evidence` existe para os candidatos; as demais camadas ficaram ausentes ou dependentes de extração futura.

## Por que o 17C7 é plano, não extração completa

Este marco não extrai raster, não baixa Sentinel, não executa SAR, não executa DINOv2/SatMAE e não preenche valores físicos, espectrais ou chuvosos sintéticos. Ele apenas mapeia quais tarefas seriam necessárias para converter os patches candidatos em unidades multimodais reais.

## Camadas existentes para patches oficiais

O dataset oficial `datasets/suscetibilidade/susc_features_by_patch_v1.csv` contém colunas para camadas físicas, urbanas, espectrais Sentinel-2 e chuva/runoff em patches oficiais. Isso indica pipeline ou produto existente para a grade oficial, mas não significa que essas features existam para os patches candidatos.

## Camadas ausentes para patches candidatos

Para os candidatos, seguem ausentes `physical_static`, `urban_territorial`, `sentinel2_spectral`, `rainfall_trigger`, `sar_observational` e `embedding_representation`. O plano registrou `200` linhas de inventário de feature para `5` patches candidatos.

## Pipelines adaptáveis

As camadas físicas, urbanas e de chuva podem ser planejadas como adaptação de pipeline existente para a grade candidata. Essa adaptação ainda depende de política para candidatos e execução controlada, sem promover patch candidato a patch oficial.

## Dependências de raster, API ou credencial

Sentinel-2 depende de tile/cena real e lineage de aquisição. SAR depende de runtime e credenciais. Embedding DINOv2/SatMAE depende primeiro de tile real e política de normalização, composição de bandas e nuvem.

## DINO/SatMAE real

Não há tile real pronto para os 5 patches candidatos. Portanto `embedding_input_ready=false`, `embedding_can_run_now=false` e nenhum embedding real foi executado. O smoke test sintético herdado do 17C6 continua não científico.

## SAR real

SAR permanece bloqueado por runtime indisponível e por ausência de execução controlada pré/pós evento para a grade candidata. Footprint pós-evento não pode ser usado como feature de suscetibilidade pré-evento.

## Uso científico futuro

Para tornar os patches candidatos cientificamente utilizáveis, ainda faltam política de grade candidata, QA humano, extração real de features, tile real, embedding real, controle anti-leakage e decisão explícita sobre score.

## Score v6, score v7 e 17B

O score v6 não foi alterado. O score v7 continua inexistente. O 17B segue bloqueado porque não há patch oficial ou política aprovada, QA aceito, features reais completas nem evidência observacional suficiente.

## Próximo marco recomendado

`SUSC-17C8 Extração Real Controlada de Features para Patches Candidatos`. Se o maior bloqueio operacional for embedding, a alternativa é `SUSC-17C8 Preparação de Tile Real / Entrada DINO`. Se o maior bloqueio institucional continuar sendo P0, a alternativa é `SUSC-17C8 Pacote de Solicitação Formal`.
