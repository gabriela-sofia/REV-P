# Relatorio v1jc - Multiplos anchors, controles e labels

## Escopo

A v1jc formaliza a passagem de um anchor oficial forte para uma governanca de multiplos anchors e controles candidatos. A etapa nao cria dado novo de imagem, nao cria label operacional e nao treina modelo.

O foco e responder uma pergunta metodologica: o que ja existe para revisao multimodal e o que ainda falta para uma base supervisionada defensavel.

## Fontes lidas

A etapa leu:

- `datasets/official_documented_event_unit_registry.csv`;
- `datasets/official_event_spatial_anchor_registry.csv`;
- outputs locais v1ir/v1is quando disponiveis;
- registries Sentinel/DINO ja emitidos para o anchor oficial.

Foram aceitas apenas coordenadas textuais explicitas ja registradas. Localidade, bairro, rua ou municipio sem coordenada nao foram convertidos em ponto.

## Resultado de anchors

Foram avaliadas 11 unidades documentais oficiais CPRM.

Resultado:

- anchors confirmados: 1;
- coordenadas explicitas recuperadas: 1;
- candidatos positivos de referencia: 1;
- eventos documentais sem coordenada: 9;
- unidade com precisao/evidencia insuficiente: 1.

O anchor confirmado e:

- `ANCHOR_PET2022_CPRM_ANEXOII_19022022`;
- unidade documental: `PET2022_CPRM_ANEXOII_19022022`;
- localidade: Bairro Moinho Preto;
- data: 19/02/2022;
- fenomeno: movimento de massa;
- coordenada: -22.484251, -43.211257.

As outras unidades CPRM continuam uteis para inventario documental, mas nao entram como anchors espaciais porque falta coordenada explicita.

## Resultado de controles candidatos

Foram criados 6 registros de controle candidato:

- 1 controle temporal do proprio anchor;
- 1 placeholder de controle espacial regional, dependente de regra de buffer;
- 3 patches PET existentes como candidatos de fundo para revisao;
- 1 linha de governanca `INVALID_NEGATIVE_LABEL`.

Nenhum controle candidato foi marcado como negativo formal. A v1jc registra explicitamente que ausencia de registro nao prova ausencia de evento.

## Prontidao de labels e treino

A matriz final registrou:

- positive_reference_candidates_count: 1;
- confirmed_anchors_count: 1;
- control_candidates_count: 6;
- negative_labels_ready_count: 0;
- sentinel_patch_coverage_count: 1;
- dino_embedding_count: 1;
- review_only_status: `REVIEW_ONLY_READY`;
- training_boundary_status: `TRAINING_BLOCKED_INSUFFICIENT_LABELS`.

As flags permanecem bloqueadas:

- `can_create_training_label=false`;
- `can_train_model=false`;
- `can_unfreeze_dino_for_scientific_claim=false`;
- `can_reopen_protocol_b=false`.

## Interpretacao

O projeto ja tem uma referencia multimodal forte para revisao: documento oficial, coordenada, par Sentinel com QA, embedding DINO frozen e probe espectral/estrutural. Isso permite continuar a auditoria review-only.

Ainda nao existe base supervisionada. Para treino real, faltam multiplos anchors, controles formais, labels aprovados, split por evento/localidade, protocolo de vazamento e metricas. O DINO deve permanecer congelado ate que essa governanca exista.
