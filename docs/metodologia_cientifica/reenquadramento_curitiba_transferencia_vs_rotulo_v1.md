# Curitiba é exceção isolada, não falha da tese — 2026-08-19

**Gatilho**: preocupação de que o colapso de Curitiba derrubasse o propósito
central do projeto (prever suscetibilidade onde não há inventário local, via
lógica físico-geográfica transferível entre regiões). Os números abaixo já
existiam desde `c3284b6` (16/08, `local_runs/mod-mec-03/resultado.json`) —
ninguém tinha olhado a tabela completa de transferência lado a lado até
agora.

## A tese central está sustentada por três fontes externas e dois relevos

`leave_one_source_out` do `mod_mec03` (modelo físico de 6 variáveis,
treinado sem a fonte, testado nela — é literalmente "prever onde não há
dado local de treino"):

| fonte deixada de fora | AUC | IC95 |
|---|---:|---|
| CEMS | 0,7200 | [0,699; 0,740] |
| Sen1Floods11 | 0,7063 | [0,671; 0,735] |
| UK (Environment Agency) | 0,7346 | [0,706; 0,762] |
| **Curitiba** | **0,4636** | **[0,404; 0,525]** |

`transferencia_relevo` (treinado fora da classe de relevo, testado nela):

| relevo deixado de fora | AUC | IC95 |
|---|---:|---|
| Íngreme (serra) | 0,8005 | [0,754; 0,846] |
| Plano/ondulado | 0,6877 | [0,669; 0,705] |

Três fontes independentes e duas classes de relevo transferem com AUC
0,69–0,80, IC95 sempre acima do acaso. **Curitiba é a única exceção, e não
por pouco** — é o único caso com IC95 que cruza 0,5 e fica com o extremo
inferior abaixo dele.

## Por que Curitiba é diferente das outras três

As outras três fontes (CEMS, Sen1Floods11, UK) têm negativo por
**observação direta** ou **exclusão qualificada bem estabelecida** — alguém
olhou a área, ou um critério de quatro pontos validado há mais tempo. O
rótulo de Curitiba vem de um canal de queixa administrativo (SIAC 156), com
76% do negativo originalmente sem evidência real (`ausência`, corrigido
nesta sessão) e um positivo cuja composição já muda ano a ano por razões não
testadas exaustivamente (`SUSC-20Q`/`21A`) — e o teste de hoje (modelo
hierárquico) mostrou que nem emprestar 62 mil pontos de fora resolve, porque
o problema não é volume de treino, é a confiabilidade do **rótulo de teste**
de Curitiba especificamente.

## O que isso muda pra tese

Nada do propósito central precisa mudar. Muda o que se espera de Curitiba:

- **Não é** um lugar onde o projeto deveria conseguir validar o modelo
  contra o próprio inventário administrativo, porque esse inventário já foi
  auditado e mostrou não ser confiável o bastante pra esse papel — o mesmo
  destino que Petrópolis já tem, por um motivo diferente (Petrópolis não tem
  inventário nenhum; Curitiba tem inventário, mas ele é instável).
- **É** exatamente o tipo de região que a pergunta de pesquisa já prevê:
  "aplicado de forma auditável a territórios urbanos vulneráveis **sem
  inventário local de eventos [confiável]**". O serviço já tem a
  maquinaria certa pra isso — `maturidade da região`, `insufficient_data`,
  `model_card` com limites de uso. Curitiba deveria ser servida como
  **aplicação** do modelo treinado nas fontes com evidência real, com
  maturidade baixa declarada — não como uma região que precisa passar no
  próprio teste prospectivo isolado.

Isso é literalmente o que E5 ("Aplicação às regiões brasileiras... nenhuma
afirmação de acerto onde não há inventário local") já promete fazer. Não é
uma mudança de rota — é a confirmação de que a rota já certa se aplica a
Curitiba com mais força do que se pensava.

## O que não fica resolvido

- O colapso do modelo *próprio* de Curitiba (SIAC 156 sozinho, treinado e
  testado nele mesmo) continua real e não explicado — isso não desaparece,
  só passa a ser lido como "evidência de que o rótulo local de Curitiba não
  serve pra validação own-region", não como "o projeto não funciona".
- Sen1Floods11 no LOSO tem só 11 grupos de bootstrap — IC mais largo que
  os outros, ler com um pouco mais de cautela que CEMS/UK.
- Nenhum número novo entra no `main.tex` nesta rodada — mesma razão de
  sempre (achado ainda não é resultado desta entrega). Este documento é
  insumo pra quando a Entrega 02 escrever isso, e pra você decidir se quer
  adiantar a leitura em conversa com o orientador antes disso.
