# SUSC-17C26 - Pacote multimodal operacional, evidence graph e fila G4/G5

## Objetivo
Consolidar offline todos os dados reais ja obtidos (Sentinel-2 multitemporal 17C24, CHIRPS real 17C25, deltas) em um pacote operacional por patch, preparando o desbloqueio G4/G5 sem coletar dados novos nem liberar 17B.

## Entregas
- Dossies por patch: 5.
- Evidence packets: 5.
- Evidence graph: 98 nos, 150 arestas.
- Prioridades review-only: 5.
- Targets Ground Reference (fila G4/G5): 15.
- Source query packages: 15.
- Checklists de revisao: 5.
- Campos obrigatorios de Ground Reference: 14.
- Hashes verificados: 55 (all_hashes_verified=True).

## G4/G5
- G4_true_count=0, G5_true_count=0. G4/G5 permanecem false por design: exigem artefato de evento observado com geometria (G4) e classificacao de fenomeno (G5).

## Guardrails
- Sensor e chuva sao contexto review-only; nenhum vira evento observado, Ground Reference, label, treino, score v7, patch oficial ou 17B. Delta e mudanca observacional review-only.

## minimum_success_achieved: True

## Proximo marco recomendado
SUSC-17C27 Aquisicao dirigida de Ground Reference (defesa civil/APAC/CEMADEN) usando a fila G4/G5 para desbloqueio futuro de G4/G5
