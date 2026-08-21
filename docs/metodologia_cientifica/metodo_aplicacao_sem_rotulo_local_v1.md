# Aplicação sem validação por rótulo local — método e resultado para Curitiba (2026-08-19)

**Princípio, pra citar em qualquer lugar do projeto daqui pra frente**: o
propósito central não é o modelo acertar o rótulo administrativo de uma
região — é prever suscetibilidade em regiões **sem** inventário local
confiável, usando a lógica físico-geográfica (HAND, TWI, declividade, chuva)
treinada onde existe evidência real, e transferida por semelhança de
terreno, não por comparação com o próprio dado da região-alvo. Isso já
estava na pergunta de pesquisa ("sem inventário local de eventos"); o que
faltava era um método concreto pra aplicar isso sem depender do rótulo da
região-alvo. Este documento registra esse método pela primeira vez, testado
em Curitiba, pra ser reaplicado (Petrópolis, e como conferência cruzada em
Recife) conforme o projeto evolui.

## O método, em três passos, nenhum deles usa o rótulo da região-alvo como critério

1. **Treinar só onde há evidência real** — as fontes com negativo observado
   ou exclusão qualificada bem estabelecida (CEMS, UK, Sen1Floods11).
   A região-alvo (Curitiba) **não entra no treino em nenhuma linha**.
2. **Caracterizar a distância de domínio** — comparar, variável a variável,
   a distribuição da região-alvo contra a distribuição de treino (diferença
   padronizada de média, % dentro do intervalo 5–95% do treino). Isso não
   depende do rótulo da região-alvo, só das variáveis físicas.
3. **Checar coerência física do escore dentro da região-alvo** — correlação
   entre o escore do modelo e cada variável física, dentro só dos pontos da
   região-alvo. Se o escore sobe com TWI e desce com HAND *mesmo sem nunca
   ter visto Curitiba*, isso é evidência de que a lógica causal transferiu —
   independente de bater com o rótulo administrativo local.

O AUC contra o rótulo local continua sendo calculado, mas como **contexto
secundário**, não como critério de aprovação — é exatamente essa inversão
de prioridade que estava faltando nas rodadas anteriores desta sessão.

## Resultado para Curitiba

**Achado principal: a lógica física transfere para Curitiba mesmo sem
nenhum dado de Curitiba no treino.**

| variável | correlação do escore com a variável, dentro de Curitiba | esperado |
|---|---:|---|
| HAND | ρ = −0,90 (p≈0) | negativo — bate |
| TWI | ρ = +0,79 (p≈0) | positivo — bate |

Essas duas são as variáveis causais centrais do projeto (HAND e TWI têm
sinal exigido em todo o resto do pipeline). A correlação é forte e no
sentido certo, apesar de o modelo nunca ter visto um único ponto de
Curitiba. **Isso é a evidência mais direta até agora de que a "essência do
projeto" — prever por semelhança de terreno, não por comparação com o dado
local — funciona.**

## O achado que faltava explicar, e que este método revelou

Elevação absoluta (`elevation_m`) tinha **0% dos pontos de Curitiba dentro
do intervalo 5–95% da distribuição de treino** (diferença padronizada = 2,76
desvios — enorme). Curitiba fica a ~900 m de altitude; as fontes de
evidência real (Reino Unido, eventos do Copernicus EMS, Sen1Floods11) estão,
em geral, perto do nível do mar. O modelo estava sendo forçado a
extrapolar numa variável completamente fora do que já viu.

**Por que isso faz sentido remover, decidido antes de olhar o resultado**:
elevação absoluta não é uma grandeza causal comparável entre cidades de
altitude de base diferente — é HAND (altura acima do dreno mais próximo,
uma medida *relativa*) que carrega o significado físico transferível. Testei
sem `elevation_m` (5 variáveis): o domínio de Curitiba passou a caber
inteiro (90–100% em todas as variáveis restantes), a coerência física
continuou igual de forte (HAND ρ=−0,90, TWI ρ=+0,79), e o AUC contra o
rótulo de Curitiba não mudou (0,4636 → 0,4639, dentro do ruído).

Isso separa limpamente os dois problemas que a sessão inteira vinha
misturando: **o domínio de elevação estava mesmo errado, e vale corrigir por
mérito próprio** (deixa o modelo honestamente transferível); **e o rótulo de
Curitiba continua não servindo pra validação own-region, e nenhuma correção
de modelo muda isso** — não é o tipo de problema que se resolve ajustando
variável.

## O que muda no pipeline, se você aprovar

- **Remover `elevation_m` do conjunto padrão do modelo multirregião**
  (`mod_mec03` e adiante) — mérito próprio (domínio), não por causa de
  Curitiba especificamente. Ainda não fiz essa mudança no script oficial;
  fiz só no teste isolado. Avisar antes de tocar em `mod_mec03.py` de
  verdade, porque isso reabre a comparação TERRENO-vs-COMPLETO que ele já
  publicou.
- **Curitiba (e Petrópolis, quando tiver variáveis físicas) passam a ser
  avaliadas por este método** — domínio + coerência física —, não por AUC
  contra o próprio rótulo administrativo. Isso é literalmente o que a
  maturidade declarada do serviço (`insufficient_data`, `region_not_supported`)
  já deveria refletir.
- **Recife** pode rodar o mesmo diagnóstico de domínio como conferência
  cruzada (Recife tem rótulo bom, então não precisa deste método pra
  validar — mas checar se o domínio de Recife cabe dentro do treino externo
  é um teste barato de coerência que ainda não foi feito).

## O que não fica resolvido, e não fica escondido

- O rótulo próprio de Curitiba continua não servindo pra validação
  own-region — isso é conclusão, não pendência.
- `rain_max_24h` e `rain_decay_index` têm correlação quase nula com o
  escore dentro de Curitiba (ρ=0,00 e ρ=0,08) — a chuva não está fazendo
  diferença local em Curitiba neste modelo. Coerente com o achado anterior
  de que HAND/TWI dominam o escore; não investigado a fundo aqui.
- Isso não é validação operacional nem promove Curitiba a `region_maturity`
  mais alta — é caracterização de transferência, o texto do `resultado.json`
  já diz isso explicitamente.

## Arquivos

`local_runs/app01-transferencia-curitiba-sem-rotulo-local/resultado.json`
(6 variáveis) e `resultado_sem_elevacao.json` (5 variáveis, sem elevação).
