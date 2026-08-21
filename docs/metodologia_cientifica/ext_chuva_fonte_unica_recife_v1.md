# Fonte única de chuva em Recife

**Data**: 2026-08-16
**Status**: correção aplicada; substitui a mistura CHIRPS×ERA5-Land descrita em
`ext_tabela_unica_e_pool_harmonizado_v1.md` §4 e reafirmada como pendência
aberta em `ext_resolucao_unica_30m_v2.md` §7

---

## 1. O que mudou, em uma frase

Os 278 pontos de Recife, que tinham `rain_max_24h_chirps`/`rain_decay_index_api_chirps`
vindos de duas fontes diferentes na mesma coluna (181 em CHIRPS v2.0, 97 em
Open-Meteo/ERA5-Land), passam a ter os 278 vindos de uma única fonte:
Open-Meteo/ERA5-Land.

## 2. De onde veio a pendência

`aud_chuva01_fontes_incompativeis.py` (e627af3) mediu, não supôs: o indicador
de qual fonte produziu o valor tinha AUC de rank (0,826) maior que a própria
chuva (0,738) contra o rótulo. Como a chuva antecedente é o preditor
declaradamente dominante da trilha pluvial, parte do que aquele número media
não era chuva — era proveniência. O próprio script dizia o que faltava:
"reamostrar CHIRPS para os 97 pontos de ERA5-Land é aquisição de dado, não
auditoria, e a auditoria precisa existir antes para dizer se vale a pena."

## 3. Por que a mistura existia

Investigando a origem em `PROJETO/local_runs/recife_modelo_v9_final` e
`recife_modelo_v12_extracao_final`: a base v9 (181 pontos, `dataset_role_source`
majoritariamente anterior a 2022) foi extraída com CHIRPS v2.0. Duas ondas
posteriores de pontos novos — 88 negativos "bairro-matched" da própria v9 e 9
pontos dos leads A/C do v12 (decreto de 28/mai/2022 e Global Flood Database) —
usaram Open-Meteo/ERA5-Land, API mais simples de chamar, sem que isso fosse
uma decisão declarada sobre qual produto é o correto. As 97 datas cobrem 2015
a 2022, 85 datas distintas — não são dois eventos concentrados.

## 4. A decisão: qual fonte vira única, e por quê

Duas direções possíveis, com custo de natureza diferente:

- **CHIRPS para os 97 restantes**: cientificamente preferível (produto com
  estação real, gauge-satellite blend; é o que a maioria já usa). Mas o
  próprio `aq_chirps3_v3.py` deste projeto documenta que o servidor da CHC
  tem proteção contra scraping — uma tentativa anterior rodou 163 minutos e
  gravou quase nada. Os 97 pontos cobrem 85 datas distintas, o que exigiria
  uma campanha dia-a-dia sem garantia de terminar.
- **Open-Meteo/ERA5-Land para os 181 restantes**: mesma API já usada com
  sucesso 3 vezes neste projeto (v8, v9 bairro-matched, v12 lead A/C), sem
  histórico de bloqueio, uma requisição por ponto. É reanálise (modelo), não
  produto com estação.

Decidido com a Gabriela em 2026-08-16: **Open-Meteo/ERA5-Land para os 278**.
O motivo foi risco operacional, não preferência de produto — está declarado
aqui para não virar decisão silenciosa se algum dia valer a pena reabrir.

## 5. Execução

`scripts/suscetibilidade/chuva02_padronizar_fonte_unica_recife.py`:

1. Lê `point_id`/`lat`/`lon`/`event_date` de `dataset_v12_final.csv`
   (PROJETO, somente leitura) para os 181 pontos hoje em CHIRPS.
2. Busca Open-Meteo/ERA5-Land com a **mesma fórmula** já usada nas fontes
   Open-Meteo existentes deste projeto — janela de 14 dias antes do evento,
   índice de decaimento com fator 0,85/dia — para não introduzir uma segunda
   inconsistência dentro da fonte única.
3. Concorrência de 4 requisições simultâneas (mesmo padrão do
   `fetch_rain_v9_bairro_matched.py`), com repetição e espera crescente em
   erro 429 (limite de taxa, não ausência de dado).
4. Sobrescreve `rain_max_24h_chirps`, `rain_decay_index_api_chirps` e
   `rain_data_source` em `local_runs/ter-03-brasil-harmonizado/recife_harmonizado.csv`
   (artefato do REV-P, não do PROJETO) para os 181 pontos; mantém os 97 que já
   estavam corretos.

**Resultado**: 181/181 reamostrados, 269/278 com valor de chuva depois da
correção. Os 9 sem valor **já eram assim antes** — são pontos `v4_canonical_base`
sem `event_date` registrado na origem, uma lacuna pré-existente e não uma
regressão desta correção (nenhum dos 9 tinha `rain_max_24h_chirps` preenchido
no v12 original).

## 6. Verificação: a auditoria muda de veredito

```
python scripts/suscetibilidade/ds04_reduzir_por_fonte.py
python scripts/suscetibilidade/aud_chuva01_fontes_incompativeis.py
```

| | antes (mistura) | depois (fonte única) |
|---|---|---|
| veredito Recife | MISTURA_DE_FONTES | **FONTE_UNICA** |
| fontes | chirps 181 / open_meteo 97 | open_meteo 278 |
| AUC rain_max_24h vs rótulo | 0,738 | 0,628 |
| AUC indicador de fonte vs rótulo | 0,826 | — (não há mais indicador; fonte única) |
| veredito global do script | (não citado; Recife era o único MISTURA) | "nenhuma fonte com mistura de produtos de precipitação" |

A `aud_chuva01` tinha um bug preexistente e não relacionado a esta correção:
`periodo_por_fonte` quebrava (`TypeError`) quando a coluna `data_evento` mistura
string e NaN no mesmo grupo — os 9 pontos sem `event_date` expuseram isso pela
primeira vez porque é a primeira vez que o script roda até o fim para Recife
com fonte única. Corrigido com `pd.to_datetime(..., errors="coerce")` antes do
`min()`/`max()`; não muda nenhum resultado numérico, só evita o traceback.

## 7. Efeito no modelo pluvial de Recife (a pergunta que importa)

`scripts/suscetibilidade/mod_recife03_pluvial_fonte_unica.py` repete
**exatamente** a metodologia do v12 publicado (Firth penalizado, 6 features
causais, LOO, mesma semente) trocando só a fonte de chuva. Terreno
permanece NATIVO (mesma resolução do v12 original) — a mudança para 30 m
(`ext_resolucao_unica_30m_v2.md`) é uma decisão diferente e fica de fora
deste antes/depois para não misturar dois efeitos na mesma comparação.

| | v12 publicado (mistura) | fonte única (2026-08-16) |
|---|---|---|
| n | 278 | 269 (9 sem chuva, pré-existente, excluídos do ajuste) |
| LOO-AUC | 0,6781 | **0,6276** |
| `rain_decay_index_api_chirps` | coef +0,9896, p < 0,0001 | coef **+0,4910**, p = 0,0005 |
| `hand_m_dinf` | coef −0,0001, p = 0,978 | coef −0,1124, p = 0,696 |
| `twi_dinf` | coef +0,2786, p = 0,046 | coef +0,2126, p = 0,123 |
| `elevation_m` | coef +0,2662, p = 0,374 | coef +0,3588, p = 0,204 |
| `slope_deg` | coef −0,1698, p = 0,224 | coef −0,1388, p = 0,300 |
| `rain_peak_residual_orthogonalized` | coef −0,1402, p = 0,347 | coef −0,1264, p = 0,330 |

**Leitura**: isto não é o modelo de Recife colapsando — é o modelo ficando
mais honesto. A chuva continua sendo o preditor dominante, com sinal correto
e estatisticamente significativo (p = 0,0005): o que muda é que metade da
magnitude do coeficiente antigo (+0,99) media proveniência do dado, não
precipitação, exatamente como `aud_chuva01` havia previsto ao medir o AUC do
indicador de fonte. HAND continua sem separar as classes (mecanismo não é
terreno em Recife — leitura que se mantém e fica mais limpa, sem o
confundimento por trás). A queda de LOO-AUC (0,678 → 0,628) é o custo
esperado de remover um viés, não uma perda de poder explicativo real.

O único achado que pede atenção: `twi_dinf` era marginalmente significativo
(p = 0,046) e deixa de ser (p = 0,123). N caiu de 278 para 269 (3%), o que
sozinho não explica a mudança — é possível que parte daquela significância
también estivesse carona no confundimento de fonte. Registrado aqui, não
escondido; não é um resultado que motive ação imediata, mas deve ser citado
se `twi_dinf` for usado em qualquer afirmação sobre Recife especificamente.

> **Nota de 20/08/2026 — a correção deixou de ser só de Recife.** No mesmo dia
> desta padronização, o `chuva04_adquirir_era5_global.py` reextraiu a chuva de
> CEMS, Sen1Floods11, UFO e do piloto inglês com a mesma fórmula, e a base
> inteira passou a ter produto único. O invariante que aqui era um teste de
> Recife virou dois testes de projeto. E a auditoria seguinte
> (`ext_chuva_estado_do_projeto_v1.md`) mostrou o que esta correção não
> alcançava: em Recife, positivos e negativos dividem só 5 das 205 datas, então
> o coeficiente de chuva que sobrou descreve o dia, não o lugar.

## 8. O que isto não faz

Não promove esta rota a canônica em `PROJETO` — `dataset_v12_final.csv`
continua intocado (a correção vive inteiramente em `REV-P`, no arquivo
harmonizado). Não decide se o v12 publicado deve ser formalmente substituído
nos textos do TCC — isso é decisão de redação, feita em conjunto com a
Gabriela, não automática. Não muda a exclusão de Recife do pool fluvial
(continua PLUVIAL_URBANO, fora por mecanismo). Não recalcula
`rain_peak_residual_orthogonalized` de forma diferente da fórmula original —
só deixou de precisar de agrupamento por fonte, porque agora há uma só.

## 9. Reprodução

```
python scripts/suscetibilidade/chuva02_padronizar_fonte_unica_recife.py
python scripts/suscetibilidade/ds04_reduzir_por_fonte.py
python scripts/suscetibilidade/aud_chuva01_fontes_incompativeis.py
python scripts/suscetibilidade/ds05_admissao_consolidacao.py
python scripts/suscetibilidade/mod_mec02_fluvial_pool_expandido.py
python scripts/suscetibilidade/mod_recife03_pluvial_fonte_unica.py
python -m pytest tests/test_ds03_ds05_tabela_unica.py -q
```

`chuva02` sobrescreve `recife_harmonizado.csv` — se `ter03_reextrair_brasil.py`
for rodado de novo, ele regenera esse arquivo a partir do v12 bruto (que
continua com a mistura, porque PROJETO não foi tocado) e desfaz esta correção
silenciosamente. **Rodar `chuva02` sempre depois de `ter03`, nunca só uma
vez.** É por isso que `test_recife_tem_fonte_de_chuva_unica` existe: se a
mistura voltar, o teste avisa.
