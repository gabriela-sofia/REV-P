# Integracao: marco label-free MV1 + evidencias externas (pos-navegacao)

> Passada review-only. `pode_virar_label_agora` e sempre `false`. Nenhum evento candidato e promovido automaticamente a label.

## Artefatos do marco label-free encontrados

- `revp_fechamento_marco_validacao_label_free_evidencia_estrutural_mv1.md`
- `revp_manifesto_marco_validacao_label_free_evidencia_estrutural_mv1.csv`
- `revp_fechamento_marco_validacao_label_free_evidencia_estrutural_mv1.json`
- `revp_proximos_passos_pos_marco_label_free_mv1.csv`

## Resultado da navegacao

- Pendentes iniciais: 8
- Resolvidas com download: 3
- Parcialmente resolvidas: 1
- Alternativas oficiais baixadas: 1
- Bloqueadas apos navegacao real: 2
- Requerem solicitacao formal: 1
- Arquivos baixados: 8
- Geometrias candidatas: 6

## Guardrails preservados

- Sem treino supervisionado; sem label binario; sem positivo formal; sem negativo formal.
- Sem ground truth operacional patch-level nesta passada.
- unknown nao vira negativo; ausencia de evidencia nao vira classe 0.
- Curitiba nao vira negativo formal.
- Evidencia externa nao vira label automaticamente.
- DINOv2 nao prova inundacao.
- Fonte textual sem geometria nao fecha patch-level.
- Landslide scar nao prova flood extent.
- Geometria de suscetibilidade nao e geometria de evento observado.
- Download de fonte nao significa validacao operacional.

## Proximo passo recomendado

Solicitar formalmente os dados sob formulario/e-mail (CEMADEN) e os portais sem arquivo estatico (APAC, Copernicus EMS rapid mapping); complementar a serie hidrologica da ANA via HidroWeb por estacao do Capibaribe; e submeter as geometrias candidatas (bacias GeoCuritiba, malhas IBGE, carta de suscetibilidade SGB) a revisao humana. Nenhuma fonte e promovida a label nesta passada.
