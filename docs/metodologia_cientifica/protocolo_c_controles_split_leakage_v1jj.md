# Protocolo C v1jj - Controles, split, leakage e sandbox

A v1jj formaliza a fronteira entre revisao multimodal, sandbox local e treino supervisionado. Ela parte do batch v1ji, no qual 9 anchors oficiais possuem Sentinel-2 com QA e embeddings DINOv2 frozen.

## O que ja pode seguir como review-only

Os 9 anchors oficiais melhoram a base de revisao porque deixam de ser um caso unico. Cada anchor tem coordenada explicita, fonte oficial, data e material multimodal auditado.

Isso permite comparar pares pre e pos, revisar contexto espectral, revisar DEM e usar DINO frozen como diagnostico estrutural. Ainda assim, esse conjunto nao e um conjunto de labels.

## Controles candidatos

A v1jj separa controles candidatos de negativos formais.

Um controle candidato pode ajudar a revisar contexto, mas nao prova ausencia de evento. Por isso:

- pre-evento do mesmo anchor e controle temporal, nao negativo independente;
- patch de fundo existente e contexto, nao prova ausencia;
- Recife e Curitiba entram apenas como contexto estrutural;
- qualquer uso de ausencia de registro como negativo e bloqueado.

Todos os controles ficam com `can_be_negative_label=false`.

## Regras anti-leakage

O protocolo exige:

- split por unidade documental;
- split por localidade;
- split por evento/data;
- buffer espacial minimo entre amostras independentes;
- pre e pos do mesmo anchor sempre juntos como unidade pareada;
- controles derivados do mesmo anchor fora de qualquer lado oposto do split;
- material cross-region apenas como contexto ou robustez de revisao.

Essas regras estao prontas para revisao, mas nao liberam treino porque faltam labels formais e negativos com protocolo.

## Sandbox permitido

A v1jj permite sandbox local fraco somente como teste de engenharia. Ele pode existir para verificar plumbing, one-class/prototype ou agregacao de features, desde que:

- nao salve pesos;
- nao seja apresentado como resultado cientifico;
- nao crie label operacional;
- nao descongele DINO para claim;
- mantenha status `INVALID_FOR_SCIENTIFIC_CLAIM`.

## Gate final

O gate permanece:

- `can_create_training_label=false`;
- `can_train_model=false`;
- `can_unfreeze_dino_for_scientific_claim=false`;
- `SUPERVISED_TRAINING_BLOCKED`.

Para treino cientifico ainda faltam positivos formalizados como labels, negativos ou controles com evidencia de ausencia, split fechado, protocolo de vazamento e metricas em conjunto separado.
