# Relatório de restrições metodológicas

## Restrições em vigor nesta entrega

O DINOv2 é usado exclusivamente como codificador visual congelado para análise estrutural e revisão. Não há rótulo binário, alvo supervisionado, classificador treinado nem afirmação de detecção, predição ou acurácia operacional.

O Protocolo C permanece bloqueado no gate C4 por ausência de negativos formais (código interno: `C4_BLOCKED_NO_FORMAL_NEGATIVES`). Isso significa que ausência de registro de evento, pseudo-ausência, área de fundo aleatória e distância de âncora não são tratadas como negativos formais nesta versão.

Essas restrições são deliberadas e auditáveis. Elas não serão removidas sem evidência que satisfaça os gates correspondentes.
