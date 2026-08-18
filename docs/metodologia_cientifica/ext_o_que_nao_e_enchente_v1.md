> **ESTE DOCUMENTO ESTA SUPERADO E CONTEM UM ACHADO INCORRETO.**
>
> A afirmacao de "invariancia do HAND" (+4,89 contra +4,65) era artefato da
> amostragem por ATIVACAO, corrigida em 2026-08-09 para amostragem por AOI.
> O valor correto do contraste observado e **+2,15**, e a conclusao se inverte.
>
> Use `ext_o_que_nao_e_enchente_v2.md`. Este arquivo permanece apenas como
> registro de que a conclusao existiu e foi retirada.

# O que não é uma enchente — efeito da definição de negativo, v1

**Data**: 2026-08-09
**Status**: achado com ressalva metodológica declarada; não é resultado final
**Artefatos**: `local_runs/mod-neg-01/`, `scripts/suscetibilidade/mod_neg01_o_que_nao_e_enchente.py`

---

## 1. Por que esta pergunta existe

Um classificador binário só sabe o que é um evento na medida em que sabe o que
**não** é. O REV-P nasceu sem essa segunda metade: todo negativo do Protocolo C
era construído por **ausência de registro** — "não há notícia de enchente aqui,
logo aqui não inundou". O gate `C4_BLOCKED_NO_FORMAL_NEGATIVES` existe por isso.

A frente externa trouxe, pela primeira vez, negativo por **observação**: área
que um analista declarou ter examinado e onde não detectou inundação
(Copernicus EMS, `areaOfInterest ∩ imageFootprint` menos `observedEvent`).

Pergunta: **as duas definições ensinam a mesma coisa ao modelo?**

---

## 2. O achado principal

Contraste entre a mediana do negativo e a mediana do positivo, **dentro de cada
fonte** (o que elimina a diferença de região do numerador):

| | `elevation_m` | `slope_deg` | `hand_m` |
|---|---|---|---|
| negativo por exclusão (Inglaterra) | +34,06 | +0,51 | **+4,89** |
| negativo por observação (CEMS) | +23,98 | +2,68 | **+4,65** |

**O `hand_m` é praticamente invariante: +4,89 contra +4,65.** A altura acima da
drenagem separa enchente de não-enchente do mesmo modo, independentemente de
como o "não" foi construído.

`slope_deg` difere por um fator de cinco. `elevation_m` difere em 10 m.

Os coeficientes do modelo confirmam:

```
CONCORDÂNCIA DE SINAL = 1/3
  hand_m       concorda
  elevation_m  inverte
  slope_deg    inverte
```

### Interpretação

**HAND carrega a física; elevação e declividade carregam a região.**

Isso é um argumento direto a favor do enquadramento causal do projeto: HAND é a
variável que se comporta como grandeza física transferível, enquanto elevação e
declividade se comportam como descritores de contexto local. Numa região o
negativo é mais alto que o positivo por 34 m; noutra por 24 m — mas em ambas
está ~4,7 m mais acima da drenagem.

Consequência prática para transferibilidade: um modelo que dependa fortemente de
elevação e declividade não deve transferir entre regiões. Um que dependa de HAND
tem chance.

---

## 3. A ressalva que invalida a comparação de desempenho

Os números de AUC obtidos **não podem** ser comparados entre as duas fontes:

| | AUC CV agrupada | grupos | EPV |
|---|---|---|---|
| exclusão (Inglaterra) | 0,7812 | 401 | 67,0 |
| observação (CEMS) | 0,5690 | **5** | **1,67** |

O modelo de observação tem apenas **cinco grupos** — cinco AOIs, em cinco partes
do mundo. `GroupKFold` sobre isso é *leave-one-continent-out*: treina em
Madagascar, Réunion, Sri Lanka e Vietnã, testa no Equador. É tarefa
incomparavelmente mais difícil que validação dentro do noroeste inglês.

Portanto **a diferença 0,781 vs 0,569 mede dificuldade de validação, não
definição de negativo.**

Pior: pela regra de EPV do próprio projeto (mínimo de 10 eventos por variável,
contados em eventos), o modelo de observação **não deveria ser interpretado** —
EPV = 1,67. A trava implementada em `mod_uk01_firth_agrupado.py` teria barrado a
execução. Neste script a trava não foi aplicada, e isso é uma falha de desenho,
registrada aqui em vez de omitida.

A caracterização da seção 2 **não** sofre dessa limitação: ela compara
distribuições, não desempenho, e não depende de validação cruzada.

---

## 4. O que seria necessário para responder de verdade

1. **Mais AOIs com negativo observado.** Cinco é pouco. Cada ativação CEMS
   adicional é um grupo a mais; com ~30 grupos o EPV chegaria a 10.
2. **Separar as AOIs internamente.** Cada ativação tem várias AOIs; hoje todas
   recebem o mesmo `grupo_cv` (o código da ativação). Usar a AOI individual
   como grupo multiplicaria os grupos por 5 a 8 sem adquirir nada novo.
3. **Negativo observado e por exclusão na MESMA região.** É a única forma de
   comparar sem confundir com geografia. Exigiria uma ativação CEMS que caia
   dentro da AOI inglesa — que a consulta ao portal mostrou não existir.

O item 2 é barato e deveria ser feito antes de qualquer conclusão de desempenho.

---

## 5. Consequência para o dataset e para o artigo

Independentemente da comparação de desempenho, a seção 2 já sustenta duas
afirmações defensáveis:

- **A hierarquia de negativo importa e deve ser declarada.** O `ds01` carimba
  cada linha com `tipo_negativo`, e todo modelo que misturar os tipos precisa
  reportar a proporção de cada um. Um AUC obtido majoritariamente sobre negativo
  por exclusão não pode ser apresentado como validado por observação.

- **HAND é a feature com maior pretensão de transferibilidade** entre as três
  testadas. Elevação e declividade mudam de sinal entre fontes.

---

## 6. Declaração

Nenhum modelo aqui é o modelo do artigo. Nenhum gate foi alterado. `twi_dinf`
ficou de fora porque as tabelas CEMS saíram com TWI inteiramente nulo (ausência
do `pysheds` no ambiente do projeto, corrigida em 09/08/2026 mas ainda não
reprocessada) — e as duas fontes precisam ter as mesmas colunas, senão a
comparação mede feature faltando em vez de definição de negativo.
