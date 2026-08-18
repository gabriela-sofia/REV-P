# Contrato de redução a pontos — fontes do Nível 1, v1

**Data**: 2026-08-07
**Status**: CONTRATO OPERACIONAL — vale para toda fonte do Nível 1
**Escopo**: define o que entra no pipeline e o que fica arquivado

---

## 1. O problema que este contrato previne

As fontes do Nível 1 são **imagem**: chips de 512×512 a 10 m (Sen1Floods11),
de 1024×1024 a 3 m (UFO), GeoTIFFs de 913 eventos a 250 m (GFD), vetores por
ativação (Copernicus EMS). Juntas somam dezenas de gigabytes.

Existe uma tentação óbvia e errada: tratá-las como material de treino de
visão computacional. Isso não funcionaria por dois motivos independentes, e é
importante que os dois estejam escritos.

**Motivo técnico**: treinar segmentação exigiria GPU, que não existe na
máquina de trabalho (10 núcleos, 15,5 GB RAM, sem placa dedicada).

**Motivo metodológico, que é o que realmente decide**: a linha de imagem já
foi testada e fechada. Em 01–02/08/2026 (SUSC-22/v1r9), três tentativas
independentes convergiram para a mesma conclusão — patch estático de
composição multi-mês não carrega assinatura de evento pontual. A rota primária
declarada é Firth penalizado sobre features físico-hidrológicas, com o EBM
como candidato interpretável. Nenhuma delas consome imagem.

Portanto a imagem do Nível 1 não é insumo de modelo. É **fonte de rótulo**.

---

## 2. O contrato

Toda fonte do Nível 1 passa por uma etapa de redução antes de qualquer uso, e
o que segue para o pipeline é somente a tabela resultante.

### Saída obrigatória

Uma tabela de pontos, uma linha por observação, com no mínimo:

| Coluna | Conteúdo | Obrigatória |
|---|---|---|
| `ponto_id` | identificador estável e reproduzível | sim |
| `lat`, `lon` | WGS84 | sim |
| `classe_obs` | `INUNDADO` \| `NAO_INUNDADO_OBSERVADO` \| `SEM_DADO` | sim |
| `data_evento` | data ou intervalo do evento observado | sim |
| `fonte` | identificador da base de origem | sim |
| `evento_id` | agrupador do evento na fonte | sim |
| `resolucao_m` | resolução nativa da observação | sim |
| `confianca` | manual, automática, ou derivada | sim |
| `licenca` | licença da fonte | sim |

### Regras

**R1. `SEM_DADO` é uma classe, não um vazio.** Pixel sem observação precisa
ser distinguível de pixel observado e seco. É essa distinção que dá ao Nível 1
o valor que ele tem; perdê-la na redução destrói o motivo de usar essas bases.

**R2. Resolução nativa fica registrada por linha.** Um ponto do GFD carrega
250 m; um do UFO, 3 m. Misturar sem marcar produziria um dataset onde a
mesma coluna significa coisas diferentes.

**R3. Água permanente é removida antes de reduzir.** Rio e lago não são
inundação. Sen1Floods11 e GFD trazem camada de água permanente (JRC) para
exatamente isso; usá-la não é opcional.

**R4. A imagem é arquivada, não carregada.** Depois da redução, os chips e
GeoTIFFs permanecem no disco como evidência de proveniência e não são
reabertos pelo pipeline. Nenhuma etapa posterior deve depender de abrir chip.

**R5. A redução é reproduzível e versionada.** Cada tabela vem acompanhada do
script que a produziu e do `PROVENIENCIA.md` da fonte. Tabela sem esses dois
não entra.

**R6. Redução não é promoção.** Uma linha com `classe_obs =
NAO_INUNDADO_OBSERVADO` é candidata a negativo, não negativo. A promoção
passa pelo critério de adjudicação, em tarefa separada.

---

## 3. Por que isso também resolve o problema de máquina

Com o contrato, o volume que o pipeline enxerga deixa de ser função do tamanho
das imagens e passa a ser função do número de pontos amostrados — que é uma
escolha nossa. Dezenas de gigabytes de chip viram alguns megabytes de tabela.

Os travamentos observados até aqui foram todos de carregamento de arquivo,
nunca de treino: 296 MB de GeoJSON estourando um ambiente de 3,9 GB, e SQLite
falhando ao escrever GPKG em pasta montada. Em contraste, o treino medido no
`docs/ambiente_treino_susc.md` foi de 0,056 s para o Firth e 25,3 s para o EBM,
no dataset real de Curitiba. O gargalo nunca foi o modelo.

---

## 4. Aplicação por fonte

| Fonte | O que reduzir | Cuidado específico |
|---|---|---|
| **Sen1Floods11** | rótulo manual de 3 estados (`LabelHand`) | usar só `HandLabeled`; os chips de rótulo fraco são de outra procedência e não podem se misturar sem marcação |
| **UFO** | máscara `inundated` / `non-inundated` | 3 m; ao reamostrar para outra resolução, documentar o método |
| **Copernicus EMS** | AOI menos mancha de inundação | a AOI é o que dá o `NAO_INUNDADO_OBSERVADO`; sem baixar a AOI a fonte perde o valor |
| **GFD** | `flooded` + `clear_views` + `jrc_perm_water` | `clear_views = 0` é `SEM_DADO`, nunca negativo; licença CC BY-NC proíbe uso no produto comercial |

---

## 5. Declaração

Este contrato não promove nenhum ponto a rótulo, não extrai feature e não
altera gate algum. Ele define formato e regra de entrada.
