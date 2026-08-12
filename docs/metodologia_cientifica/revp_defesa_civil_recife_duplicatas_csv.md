# Decisão: duplicatas de CSV da Defesa Civil de Recife

**Status**: nota curta de decisão — etapa 3 de
`revp_proxima_linhagem_programacao_pos_api.md`. Não altera dado, não apaga
nada em `PROJETO` (read-only por definição de `REV-P/CLAUDE.md`).

## O que foi comparado

Em `PROJETO/data/raw/recife/seced_defesa_civil/` existem 3 pares de CSV com
nome-base igual e UUID diferente no nome do arquivo (padrão de recurso CKAN),
mais 8 pares de dicionário de dados (`.json`) no mesmo padrão. Comparação por
hash MD5, contagem de linhas e tamanho em bytes:

| Par (nome-base) | Linhas | Bytes | MD5 | Timestamp de download |
|---|---|---|---|---|
| `solicitacoes_156_atendimentos_de_servicos_da_emlurb` (2 arquivos) | 104 = 104 | 18104 = 18104 | idêntico | 2026-05-03 20:25 (ambos) |
| `sedec_solicitacoes_tempo_real` (2 arquivos) | 174 = 174 | 57240 = 57240 | idêntico | 2026-05-03 20:25 (ambos) |
| `sedec_vistorias_tempo_real` (2 arquivos) | 2 = 2 | 299 = 299 | idêntico | 2026-05-03 20:25 (ambos) |
| 8 dicionários de dados (`dicionario_de_dados_*`, `faixa_azul_*`) | — | — | idênticos par a par | 2026-05-03 20:25 (todos) |

## Decisão

Todos os pares são **byte-a-byte idênticos** (mesmo MD5, mesma contagem de
linhas, mesmo timestamp de download). Não há diferença de cobertura
temporal nem de conteúdo — é o mesmo pacote CKAN baixado 2x na mesma sessão
de aquisição (2026-05-03 20:25), e o CKAN reatribuiu um novo UUID de recurso
a cada fetch, o que explica o nome duplicado.

**Canônico**: para cada par, o arquivo com o UUID lexicamente menor é
adotado como canônico (arbitrário entre duas cópias idênticas, só para dar
unicidade a referências futuras):

- `solicitacoes_156_atendimentos_de_servicos_da_emlurb__c394160a-...csv`
- `sedec_solicitacoes_tempo_real__2f8f3a95-...csv`
- `sedec_vistorias_tempo_real__c9654429-...csv`

O par gêmeo de cada um (`f3b3a0ab-...`, `fa135ecc-...`, `d7c98300-...`) é
duplicata confirmada, não cobertura adicional.

## O que não foi feito

Nenhum arquivo foi apagado em `PROJETO` — é diretório privado/histórico,
read-only por `REV-P/CLAUDE.md`. Se a limpeza física for desejada, é ação
manual do usuário em `PROJETO`, fora do escopo do REV-P.

## Efeito prático

Nenhuma pendência de dado real (nenhum evento de enchente novo escondido no
par duplicado). O pipeline de Recife (v7→v12) já usa esses dados; esta nota
apenas remove a ambiguidade de qual arquivo é a fonte de verdade para
qualquer replicação futura do padrão de aquisição em Curitiba/Petrópolis.
