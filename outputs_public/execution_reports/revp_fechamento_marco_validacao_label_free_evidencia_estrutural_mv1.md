# Fechamento do marco de validação label-free e evidência estrutural MV1

## 1. Escopo

Este relatório consolida o marco público `revp_fechamento_marco_validacao_label_free_evidencia_estrutural_mv1`. O objetivo é registrar, em português técnico do Brasil, a leitura metodológica da restauração `v2dz-v2ef`, da auditoria temporal MV1 e da validação estrutural label-free executada com embeddings DINOv2 congelados.

Este fechamento não cria análise numérica pesada, não cria labels, não cria negativos formais, não cria ground truth operacional e não libera treino supervisionado. A função do pacote é organizar o estado já produzido, explicitar bloqueios e fixar o próximo passo científico correto.

## 2. Branch usada

Branch confirmada no início da execução: `marco/validacao-label-free-evidencia-estrutural-mv1`.

Nada estava staged no início da execução. Não foi criada nova branch.

## 3. Linha do tempo técnica

1. Restauração `v2dz-v2ef`: a trilha forense recuperou sete CSVs públicos com 53 linhas cada, além de scripts e teste relacionados, mantendo decisões humanas vazias, gates positivo/negativo fechados e `ground_truth_operational_status=ABSENT`.
2. Auditoria temporal MV1: a auditoria de metadados Sentinel/DINO contabilizou 164 patches e 508 assets, mas registrou metadados temporais e de nuvem insuficientes para a Trilha A.
3. Validação label-free estrutural: a Trilha B foi executada como piloto exploratório com 12 embeddings DINOv2 válidos, dimensão 768 e distribuição 4/4/4 entre Curitiba, Petrópolis e Recife.

## 4. Estado consolidado dos dados

O corpus territorial/contextual original aparece em artefatos públicos existentes com 59 patches, sem autorização para criar label. Esse corpus é distinto do universo auditado temporalmente na MV1.

Estado consolidado confirmado pelos artefatos:

- 59 patches como corpus territorial/contextual original.
- 164 patches auditados temporalmente.
- 508 assets contabilizados na auditoria temporal.
- 12 embeddings DINOv2 válidos.
- Dimensão dos embeddings: 768.
- Distribuição dos embeddings válidos: 4 Curitiba, 4 Petrópolis e 4 Recife.
- 0 patches com 3 ou mais datas úteis confirmadas na auditoria temporal.
- 0 patches elegíveis para deslocamento temporal.
- 162 patches com metadados de nuvem ausentes.
- 156 patches com metadados temporais ausentes.

## 5. Decisão sobre Trilha A

A Trilha A permanece bloqueada. A razão técnica é a ausência de volume suficiente de patches com 3 ou mais datas úteis e metadados de nuvem rastreáveis por patch.

O estado consolidado é `METADADOS_TEMPORAIS_INSUFICIENTES_TRILHA_B_RECOMENDADA`. A auditoria temporal não libera deslocamento temporal amplo, não libera tabela multimodal de atributos e não libera treino supervisionado.

## 6. Decisão sobre Trilha B

A Trilha B foi executada como piloto label-free exploratório. Ela é válida para leitura estrutural inicial da topologia dos embeddings, vizinhança entre amostras e uso de evidência contextual como probe externo.

Ela é insuficiente como evidência estatística final porque o conjunto tem `n=12`. O status metodológico consolidado é `PILOTO_LABEL_FREE_COM_LIMITACAO_AMOSTRAL`.

## 7. O que o marco prova

O marco prova que existe uma infraestrutura auditável para manter a cadeia pública de evidências, bloqueios e decisões metodológicas sem promover automaticamente labels ou treino.

O marco também prova que é possível executar uma validação label-free exploratória com embeddings DINOv2 congelados e métricas estruturais rastreáveis, desde que a leitura permaneça limitada a piloto, topologia e priorização de revisão humana.

## 8. O que o marco não prova

O marco não prova detecção de inundação, suscetibilidade, classe binária, negativo formal, positivo confirmado, ground truth operacional ou validação operacional.

O marco não prova suficiência estatística final. Também não prova que Curitiba seja negativo formal, nem que ausência de evento possa ser convertida em classe 0.

## 9. Guardrails preservados

- `unknown` nunca vira negativo.
- Ausência de evento nunca vira classe 0.
- Curitiba nunca vira negativo formal.
- Evidência contextual nunca vira label.
- DINOv2 não prova inundação.
- Restauração `v2dz-v2ef` não vira ground truth operacional.
- Auditoria temporal não libera Trilha A.
- Piloto label-free com `n=12` não vira evidência estatística final.
- Nenhum treino supervisionado é liberado.

## 10. Riscos metodológicos remanescentes

- Limitação amostral do piloto label-free.
- Ausência de múltiplas datas Sentinel úteis por patch.
- Ausência de cobertura de nuvem suficiente por patch.
- Ausência de revisão humana/adjudicação fechada.
- Ausência de ontologia formal de labels patch-level.
- Ausência de política formal de evidência negativa.
- Risco de linguagem pública promover evidência contextual a label se os guardrails forem relaxados.

## 11. Próximo passo recomendado

O próximo passo científico correto é expandir os embeddings DINOv2 rastreáveis para pelo menos 30 patches e, idealmente, para os 59 patches do corpus territorial/contextual original. Em paralelo, devem ser registradas múltiplas datas Sentinel e cobertura de nuvem por patch.

Somente depois dessa expansão, da checagem de sanidade de domínio do DINOv2, da ontologia de labels, da política de evidência negativa e da revisão humana/adjudicação fechada, pode ser avaliada uma feature table multimodal. Mesmo assim, baselines supervisionados leves continuam bloqueados até evidência formal suficiente.

## 12. Linguagem pública permitida

- "infraestrutura auditável"
- "validação label-free"
- "piloto exploratório"
- "topologia dos embeddings"
- "representação visual congelada"
- "evidência contextual como probe externo"
- "priorização de revisão humana"
- "ground truth operacional ausente"
- "treino supervisionado bloqueado"

## 13. Linguagem pública proibida

- "modelo detecta inundação"
- "modelo prediz suscetibilidade"
- "Curitiba é negativo"
- "classe 0"
- "positivo confirmado"
- "ground truth operacional"
- "treino liberado"
- "validação operacional"
- "acurácia de detecção"

## 14. Conclusão executiva

O marco MV1 fecha uma virada metodológica conservadora: a trilha forense `v2dz-v2ef` foi restaurada como base auditável, a auditoria temporal bloqueou a Trilha A por falta de metadados temporais e de nuvem suficientes, e a Trilha B foi executada como piloto label-free com limitação amostral explícita.

A decisão pública correta é manter `PILOTO_LABEL_FREE_COM_LIMITACAO_AMOSTRAL`, preservar os guardrails e bloquear qualquer treino supervisionado até expansão amostral, metadados temporais adequados, política de negativos, ontologia de labels e revisão humana/adjudicação.
