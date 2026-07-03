# SUSC-18C - Self-Supervised Multimodal Pretraining e Ground Reference Factory

## Objetivo
Iniciar aprendizado multimodal REAL sem labels (self-supervised sobre o feature tensor 18B) e, em paralelo, construir a fabrica de Ground Reference necessaria para destravar treino supervisionado futuro de forma cientificamente valida.

## Resultado: A_self_supervised_complete_supervised_blocked
- self_supervised_pretraining_completed=True
- supervised_training_executed=False | ready_pending_approval=False | allowed_after_18c=False

## Frente A - Self-supervised pretraining (numpy deterministico)
- Training tensor: 316 x 26 features (targets proibidos excluidos).
- Masked feature modeling: 6 experimentos (mask 10/20/30% + family masks spectral/physical/hydrology).
- Denoising autoencoder (SVD linear 5d): 316 embeddings.
- Contrastive similarity: 3160 pares (top10).
- Representation quality audit: 7 metricas.

## Frente B - Ground Reference Factory
- Target pack: 316 | tentativas de aquisicao: 60 (rede=False, fail-closed).
- Ground reference candidates: 11 (ocorrencias oficiais geocodificadas dos canarios, 17C33/17C32) | patch-links: 11 | QA queue: 33.
- Label contract: accepted=0 (req>=50), eventos=0 (req>=5), regioes=0 (req>=3) -> training_allowed=False.
- Supervised gate recheck: allowed_after_18c=False, executed=False.

## Por que supervised continua bloqueado (Resultado A honesto)
Os candidatos de ground reference sao reais (ocorrencias hidrologicas oficiais geocodificadas), mas: (1) pendentes de QA; (2) apenas 1 evento (REC_2022_05_24_30) < 5; (3) 1 cidade (Recife) < 3 regioes; (4) geocode nivel-rua ~800m; (5) politica de negativos nao aprovada. Candidato != ground truth. Nada e aceito -> .fit supervisionado NAO executa.

## Guardrails cientificos
- Self-supervised treina mas e review_only; NAO usa labels/GT; canario!=positivo; controle!=negativo.
- Ocorrencia so referencia/ancora (nunca feature causal); score v6/index so referencia (nunca target); flow_acc nao equivalente.
- Footprint pos-evento nunca como feature pre-evento; ground reference candidate nao e GT ate QA+contrato.
- Score v6 intacto; score v7 inexistente; ground truth nao criado; labels nao criados; 17B fail-closed.

## minimum_success_achieved: True | result_class: A_self_supervised_complete_supervised_blocked

## Proximo marco recomendado
SUSC-18D executar QA humano da fila de ground reference (7 tipos de decisao) e ampliar aquisicao para >=5 eventos e >=3 regioes; ao atingir o contrato, habilitar supervised baseline sob aprovacao explicita (Resultado C). Ate la: self-supervised pretraining review-only e triagem de candidatos. Sem score v7, sem ground truth, 17B fail-closed.
