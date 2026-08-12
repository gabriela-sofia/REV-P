# O que não é uma enchente — efeito da definição de negativo, v2

**Data**: 2026-08-09
**Substitui**: `ext_o_que_nao_e_enchente_v1.md`, que continha um achado incorreto
**Artefatos**: `local_runs/mod-neg-01/`, `scripts/suscetibilidade/mod_neg01_o_que_nao_e_enchente.py`

---

## 0. O que mudou da v1 para a v2, e por quê

A v1 afirmava que o contraste de HAND era **invariante** entre as duas
definições de negativo (+4,89 contra +4,65) e concluía que "HAND carrega a
física, elevação e declividade carregam a região".

**Essa conclusão estava errada.** Ela era artefato da amostragem: até
2026-08-09 o `cems02` amostrava 300 pontos por classe **por ativação**, o que
concentrava a amostra nas AOIs maiores e sub-representava as pequenas. Com a
amostragem corrigida para 120 pontos por classe **por AOI**, o contraste
observado caiu para **+2,15** — menos da metade do valor da exclusão.

A v1 também comparava AUCs sem trava de EPV, com 5 grupos e EPV 1,67. A trava
agora existe e a comparação só roda acima de 10.

Esta versão traz o achado real, que é mais forte que o anterior.

---

## 1. A pergunta

O REV-P nasceu sem negativo formal. Todo negativo do Protocolo C é construído
por **ausência de registro**. O gate `C4_BLOCKED_NO_FORMAL_NEGATIVES` existe
por isso.

A frente externa trouxe negativo por **observação**: área que um analista
declarou ter examinado e onde não detectou inundação.

As duas definições ensinam a mesma coisa ao modelo?

---

## 2. Base de comparação

| | negativo por exclusão | negativo por observação |
|---|---|---|
| Origem | Inglaterra, critério N1–N4 | 5 ativações Copernicus EMS |
| n | 7.476 (3.738 / 3.738) | 9.304 (4.384 / 4.920) |
| Grupos de validação | 401 (evento) | 41 (AOI) |
| EPV | 133,7 | 13,7 |

Ambos acima do mínimo de 10. A comparação é legítima.

---

## 3. Achado 1 — o negativo por exclusão é sistematicamente mais extremo

Contraste entre a mediana do negativo e a do positivo, **dentro de cada fonte**:

| | `elevation_m` | `slope_deg` | `hand_m` |
|---|---|---|---|
| exclusão | **+34,06** | +0,51 | **+4,89** |
| observação | +7,50 | +1,01 | +2,15 |

Em elevação a diferença é de **4,5×**; em HAND, de **2,3×**.

Isso tem explicação direta no critério. A condição N3 exige **400 m de
afastamento** de qualquer área inundável. Esse afastamento seleciona terreno
mais alto e mais distante da drenagem do que a realidade de "foi olhado e
estava seco" — no caso observado, o negativo está frequentemente a poucos
metros da mancha de inundação.

O critério de exclusão não amostra o negativo: ele **fabrica** um negativo mais
fácil.

---

## 4. Achado 2 — a assimetria da transferência

Cada modelo avaliado sobre o dado do outro:

| Treinado em | Avaliado em | AUC | AUC próprio | variação |
|---|---|---|---|---|
| exclusão | observação | 0,6222 | 0,7812 | **−0,159** |
| observação | exclusão | **0,7798** | 0,6634 | **+0,116** |

**O modelo treinado com negativo observado atinge 0,7798 no dado de exclusão —
praticamente igual ao 0,7812 do modelo nativo daquele conjunto.** Ele
generaliza quase perfeitamente para fora.

O caminho inverso não funciona: o modelo de exclusão perde 0,159 ao encontrar
negativo observado.

Isto é contraintuitivo e é o ponto central: **o modelo com AUC próprio menor é
o que transfere melhor.** O AUC alto do modelo de exclusão não mede competência
sobre o fenômeno; mede ajuste ao próprio critério de construção do negativo.

---

## 5. Achado 3 — o mecanismo, visível no coeficiente

| feature | exclusão | observação | concorda? |
|---|---|---|---|
| `elevation_m` | −0,3883 | +0,2168 | não |
| `slope_deg` | +0,5983 | −0,0107 | não |
| `hand_m` | **−2,6144** | **−1,2752** | sim |

Concordância de sinal: **1 de 3**. Apenas `hand_m` concorda — e mesmo assim o
coeficiente da exclusão é **duas vezes maior**.

Essa inflação é o mecanismo da assimetria da seção 4. O buffer de 400 m produz
uma separação em HAND mais nítida do que a que existe no mundo. O modelo
aprende a esperar essa nitidez; quando encontra o negativo real, erra.

`elevation_m` e `slope_deg` invertem de sinal entre as fontes, o que indica que
carregam informação específica de região e não de processo.

---

## 6. Consequências

**Para a hierarquia de negativo.** O `ds01` carimba cada linha com
`tipo_negativo`, e a regra é: modelo que misturar os tipos **deve reportar a
proporção de cada um**. Um AUC obtido majoritariamente sobre negativo por
exclusão não pode ser apresentado como validado por observação. A seção 4
mostra que essa não é uma formalidade — a diferença é de 0,159 em AUC.

**Para a transferibilidade.** Um modelo cujo desempenho dependa de `elevation`
e `slope` não deve transferir entre regiões: esses coeficientes trocam de sinal.
Isso é hipótese relevante para o colapso temporal de Curitiba (AUC 0,6459 →
0,5246 em holdout 2026), que sete diagnósticos internos não explicaram.

**Para o Protocolo C.** Se o negativo por exclusão qualificada — que exige
afastamento, ausência de registro histórico e ausência de zona modelada — já
distorce dessa forma, o negativo por ausência pura do Protocolo C distorce
mais. Isso é argumento a favor de manter o `C4` fechado até haver observação.

---

## 7. Limitações declaradas

**Três features, não quatro.** `twi_dinf` ficou de fora porque as tabelas CEMS
saíram com TWI nulo (ausência do `pysheds` no ambiente, corrigida em
2026-08-09). Refazer com 4 features é pendência registrada.

**A AOI não é unidade perfeita.** Ela resolve autocorrelação espacial, mas AOIs
da mesma ativação compartilham o evento meteorológico — duas AOIs a 200 km no
Vietnã foram inundadas pela mesma tempestade. O agrupamento trata o espaço, não
o tempo.

**Regiões diferentes.** Exclusão é Inglaterra; observação são cinco ativações
tropicais. A comparação de coeficientes é entre modelos que veem geografias
distintas. A seção 4, por ser transferência cruzada, é o teste menos vulnerável
a essa confusão — mas não é imune.

---

## 8. Declaração

Nenhum modelo aqui é o modelo do artigo. Nenhum gate foi alterado. O objetivo
foi medir o efeito da definição de negativo, não maximizar desempenho.
