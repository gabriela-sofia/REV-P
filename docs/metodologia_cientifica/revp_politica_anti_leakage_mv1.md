# Política anti-leakage MV1

## Separação de fontes
A fonte de label deve ser independente da fonte de feature. Um embedding DINOv2, um asset Sentinel, uma métrica de similaridade, uma evidência contextual ou uma variável administrativa não pode ser usada simultaneamente como feature e como fonte de decisão supervisionada.

## Fonte de label
Fonte de label futura deve vir de evento observado independente, revisão humana/adjudicação e documentação rastreável. A fonte precisa ser auditable e não derivada do mesmo sinal usado no modelo.

## Fonte de feature
Fonte de feature pode incluir asset Sentinel, embedding DINOv2 congelado e metadados controlados, desde que não carregue o alvo ou proxy direto do alvo.

## Fonte contextual
Evidência contextual pode ser usada como probe externo ou priorização de revisão humana. Ela não deve preencher label, negativo formal ou decisão de treino.

## Checagens obrigatórias
- Verificar se a fonte de label aparece como feature.
- Verificar se a feature foi usada para selecionar o estado de label.
- Verificar se evidência contextual foi promovida a alvo supervisionado.
- Verificar se cidade, ausência ou unknown foram usados como negativo formal.
