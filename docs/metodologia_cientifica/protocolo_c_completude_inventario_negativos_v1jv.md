# Protocolo C - completude de inventario para negativos v1jv

v1jv testa uma rota cientifica diferente da busca textual de negativos: uma camada oficial/local de feicoes mapeadas so pode originar candidatos negativos se a area de cobertura e a completude do inventario forem auditaveis.

Regras:
- ausencia de registro nao e ausencia de evento;
- areas fora de poligonos podem ser, no maximo, candidatas review-only se a completude nao estiver provada;
- `can_create_training_label` continua falso enquanto `complete_inventory_gate` nao for PASS;
- DINO permanece congelado e review-only.
