# Protocolo C - Relatorio Metodologico Consolidado

## 1. Resumo Executivo

O Protocolo C implementou uma camada completa de auditoria para estabelecimento
de ground truth operacional no projeto REV-P (Recife). O resultado final e um
**achado cientifico negativo auditavel**: nenhuma evidencia foi promovida a ground
truth operacional, nenhum rotulo foi criado, nenhum modelo foi treinado.

Status temporal: **TEMPORAL_RECOVERY_FAIL_CLOSED**
Status observacional: **OBSERVED_EVIDENCE_REVIEW_ONLY_FAIL_CLOSED**

## 2. O que o Protocolo C tentou resolver

O objetivo era estabelecer se existem condicoes suficientes para criar ground
truth operacional de inundacao urbana em Recife a partir de:
- Dados Sentinel-2 pre/pos-evento
- Fontes oficiais externas (decretos, boletins de defesa civil)
- Embeddings DINOv2 para representacao estrutural

A resposta e: **nao, nas condicoes atuais**. E esse resultado negativo e
metodologicamente valido e publicavel.

## 3. Recuperacao Temporal Sentinel

- Patches avaliados: 2654
- Datas de produto confirmadas: 0
- Desbloqueios temporais: 0
- C3+ candidatos: 0

A recuperacao temporal tentou estabelecer cadeia de proveniencia
patch -> asset local -> produto Sentinel -> data de aquisicao real.
Em 2654 tentativas, nenhuma cadeia completa foi confirmada.

Isso significa que nenhum patch pode ser associado temporalmente a um evento
com confianca suficiente para adjudicacao operacional.

## 4. Camada Observacional de Evidencias Externas

- Fontes escaneadas: 330
- Candidatos permitidos: 22
- Evidencia contextual: 2
- Evidencia bloqueada: 2
- Vinculos evento-patch: 12
- Vinculos temporais confirmados: 0

A camada observacional identificou candidatos a eventos de inundacao em Recife
(decretos, dossiers, gaps identificados), mas nenhum foi confirmado por fonte
institucional adquirida. As fontes primarias (COMPDEC Recife, Diario Oficial)
nao foram obtidas durante o periodo do projeto.

## 5. Linkage Evento-Patch

Os 12 vinculos evento-patch gerados sao todos contextuais ou
temporalmente bloqueados. Sem data Sentinel confirmada, qualquer vinculo
temporal seria baseado em inferencia — explicitamente proibido pelo protocolo.

## 6. Niveis C1/C2/C3/C4

| Nivel | Contagem | Significado |
|-------|----------|-------------|
| C1    | 2     | Contextual apenas — evidencia territorial documentada |
| C2    | 2     | Review-only — representacao DINO sem label |
| C3    | 0        | Nao alcancado — requer date Sentinel confirmada |
| C3+   | 0        | Nao alcancado — requer temporal + espacial confirmados |
| C4    | 0        | Fechado — requer negativo formal explicito |

## 7. Papel do DINO

DINOv2 with registers e usado exclusivamente para representacao estrutural
visual. A fila DINO review-only contem 2 entradas. Nenhum target,
nenhum label, nenhum ground truth e derivado de embeddings DINO.

DINOv2 permanece valido como:
- Representacao visual/semantica de patches
- Ferramenta de triagem estrutural review-only
- Evidencia de similaridade visual (nao classificacao)

DINOv2 NAO e usado para:
- Classificacao supervisionada
- Validacao de evento
- Criacao de rotulos
- Treinamento de modelos

## 8. Guardrails Anti-Overclaim

Invariantes verificadas em todo o pipeline:
- ground_truth=true: NAO encontrado
- can_train_model=true: NAO encontrado
- can_create_operational_label=true: NAO encontrado
- labels_created: 0
- training_targets_created: 0
- DINO: apenas REVIEW_ONLY_REPRESENTATION

## 9. Achado Cientifico Negativo

O resultado negativo do Protocolo C e um achado metodologico, nao um fracasso.
Demonstra que:

1. O projeto evitou transformar evidencia fraca em label operacional
2. A cadeia de proveniencia temporal nao pode ser estabelecida sem metadados
   Sentinel reais (produto, nao filename ou mtime)
3. Evidencia contextual (dossiers, decretos nao-adquiridos) nao substitui
   confirmacao institucional direta
4. DINOv2 permanece valido como representacao, nao como classificador

## 10. Como Escrever no TCC

### Principios

- Resultado negativo e resultado cientifico
- Evidencia de rigor metodologico e publicavel
- A ausencia de ground truth operacional comprova que o pipeline e conservador
- DINOv2 nao perde validade — muda apenas o escopo de uso documentado

### Armadilhas a evitar

- NAO escrever "falha na obtencao de ground truth" (e decisao metodologica, nao falha)
- NAO escrever "DINOv2 nao funciona" (funciona como representacao)
- NAO escrever "impossivel detectar inundacao" (possivel com mais dados, fora do escopo)
- NAO escrever "Protocolo C nao encontrou nada" (encontrou achado negativo auditavel)

## 11. Texto Pronto para Metodos

"O Protocolo C foi estruturado como camada auditavel de estabelecimento de ground
truth, operando em cascata temporal (v1og-v1ot) e observacional (v1ou-v1pa). A
recuperacao temporal avaliou 2654 patches Sentinel unicos via cadeia de
proveniencia patch-asset-produto-aquisicao. A camada observacional escaneou
330 arquivos do repositorio, identificando 22 candidatos a fontes
externas. Todas as decisoes sao classificadas em niveis C1 a C4, com guardrails
que impedem promocao automatica de evidencia a ground truth operacional."

## 12. Texto Pronto para Resultados

"A recuperacao temporal resultou em TEMPORAL_RECOVERY_FAIL_CLOSED: nenhuma das
2654 tentativas de resolucao de data produziu cadeia de proveniencia
confirmada (product_dates_confirmed_real=0). A camada observacional classificou
2 candidatos como C1 (contextual) e 2 como C2 (review-only), sem alcance
de C3+ (bloqueado por ausencia de data Sentinel confirmada) ou C4 (bloqueado
por ausencia de negativo formal). O status final e OBSERVED_EVIDENCE_REVIEW_ONLY_FAIL_CLOSED,
indicando que o Protocolo C encerrou sem estabelecer ground truth operacional
mas preservando a integridade metodologica do pipeline."

## 13. Texto Pronto para Discussao

"O resultado negativo do Protocolo C demonstra que o pipeline REV-P opera sob
principios conservadores: na ausencia de cadeia temporal Sentinel confirmada e
de negativos formais, nenhuma evidencia e promovida a rotulo operacional. Este
achado e consistente com a literatura sobre ground truth establishment em sensoriamento
remoto, onde a qualidade da referencia determina a validade dos modelos downstream.
Os embeddings DINOv2 permanecem validos como representacao visual nao-supervisionada,
preservando seu potencial para futuros pipelines que disponham de ground truth adequado."

## 14. Limitacoes

1. Fontes oficiais de Recife (COMPDEC, Diario Oficial) nao foram adquiridas
2. Metadados Sentinel locais nao continham sidecars com data de aquisicao
3. O escopo temporal foi restrito a patches existentes no corpus
4. Nao houve acesso a rede durante a execucao do pipeline
5. Nao houve revisao supervisora formal dos candidatos a eventos
6. O pipeline e reproduzivel apenas com os mesmos assets locais

## 15. Proximos Passos

Condicoes para destravar ground truth operacional no futuro:
1. Adquirir boletins COMPDEC Recife via pedido LAI
2. Adquirir decreto completo do Diario Oficial Municipal
3. Re-executar pipeline Sentinel com assets contendo sidecars STAC/MTD
4. Estabelecer negativo formal de ao menos uma fonte oficial
5. Executar revisao supervisora com especialista em desastres
6. Nenhum desses passos deve ser executado automaticamente por este pipeline
