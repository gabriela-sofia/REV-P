# O serviço e seu contrato — E6/M5 executável

**Data**: 2026-08-20
**Artefatos**: `local_runs/svc-01-modelos/`, `local_runs/svc-02-contrato/`
**Scripts**: `scripts/servico/svc01_construir_modelos_servidos.py`, `svc02_contrato_inferencia.py`
**Testes**: `tests/test_svc_contrato_inferencia.py` (29)
**Resolve**: as pendências declaradas em `revp_contrato_inferencia_v0_revalidacao_cientifica.md` §5

---

## 1. O que existia, e por que não bastava

O contrato de inferência existia em três lugares e em nenhum deles executava
dentro do REV-P: como texto na §II do plano, como esboço de três estados em
`esboco_telas_minimas_produto_v1.md`, e como MVP no repositório privado. E o
documento de revalidação de 23/07 terminou com uma frase que travava tudo:

> a semântica do `confidence_interval` "precisa ser tomada antes de qualquer
> código de API, não depois".

Essa decisão nunca foi tomada. Este trabalho a toma, e com ela o contrato vira
função pura, auditável e testada.

**Não é um servidor HTTP, e isso é escolha.** O que precisa ser auditado é o
contrato — quais portões existem, em que ordem, o que cada recusa significa e
como o escore é construído. Embrulhar isso em FastAPI é trabalho de transporte,
não de método, e traria dependência de rede para dentro de um repositório que é
*fail-closed* por regra. `inferir()` é a função que um servidor chamaria.

---

## 2. A decisão que estava travando: o IC do escore

**Adotado: bootstrap do preditor linear, reamostrando grupos.** Reamostram-se os
grupos do conjunto de ajuste com reposição, N=1000 vezes; cada reamostra
reajusta o Firth; as réplicas de coeficiente ficam gravadas no artefato servido.
O IC de um escore é o percentil das N projeções.

**Por que não o delta method**: ele depende da aproximação assintótica do
erro-padrão, discutivelmente frágil exatamente no regime de n pequeno que
motivou usar Firth. Usar Firth por causa do n pequeno e depois propagar
incerteza por aproximação assintótica seria incoerente com a própria escolha do
estimador.

**Por que reamostrar grupo e não linha**: regra U2 do projeto. Reamostrar linha
infla a precisão por pseudo-replicação, e o IC sairia estreito demais — o modo
de falha mais perigoso num número que vai dentro de uma resposta de API.

**Custo**: o reajuste custa 0,2 s no maior conjunto, então N=1000 custa 239 s.
Isso acontece uma vez, na construção. A requisição só projeta: servir é
multiplicação de matriz.

A segunda pendência do documento de 23/07 — o orçamento de EPV para o teste A/B
com DINO — não é mais pré-requisito de nada: o DINO foi descartado como
*feature* na Fase 1 e vive como evidência visual.

---

## 3. Os três modelos servidos

| modelo | ajustado em | variáveis | AUC agrupada | veredito |
|---|---|---|---:|---|
| `recife_pluvial` | Recife, 269 pontos | 6 | 0,6409 | **`FORA_DOS_CRITERIOS`** |
| `fluvial_planicie` | 56.654 pontos, 509 grupos, **sem Curitiba e sem Recife** | 4 | 0,7336 | `COERENTE_COM_CRITERIOS` |
| `fluvial_serra` | 5.162 pontos, 24 grupos, estrangeiros | 1 | 0,7916 | `COERENTE_COM_CRITERIOS` |

**A região-alvo não entra no treino do modelo que a serve.** É a regra de
`metodo_aplicacao_sem_rotulo_local_v1.md` virada construção: o propósito é
prever onde não há inventário local, e validar contra o rótulo da própria região
seria a pergunta errada. Um teste guarda esse invariante.

**O modelo de Recife não atinge os critérios de leitura do projeto.** Com a
fonte de chuva corrigida, o LOO-AUC é 0,6409 — abaixo da faixa 0,70–0,88 fixada
em 09/08 — e o IC de `hand_m` cruza zero. Isso não é novidade desta rodada
(está em `ext_chuva_fonte_unica_recife_v1.md`), mas é a primeira vez que o
serviço tem de decidir o que fazer com isso.

---

## 4. Os cinco portões, na ordem em que são avaliados

A ordem importa: cada portão só é avaliado se o anterior fechou, e a resposta
nomeia o **primeiro** que falhou. Recusar por "faltou HAND" quando a geometria
nem era válida esconderia o erro real de quem chamou.

| # | portão | o que verifica | falha devolve |
|---|---|---|---|
| G1 | `geometria_valida` | ao menos um ponto, CRS suportado, coordenadas finitas e dentro do globo | `insufficient_data` |
| G2 | `regiao_resolvida` | região declarada, ou deduzida pela caixa envolvente das regiões que existem na base | `region_not_supported` |
| G3 | `modelo_para_a_regiao` | existe modelo servível mapeado para a região | `region_not_supported` |
| G4 | `variaveis_presentes` | toda variável do modelo presente e finita em todo ponto | `insufficient_data` |
| G5 | `dominio_coberto` | quantas variáveis caem na faixa 5–95% que o modelo viu | `insufficient_data` acima do limite |

**G5 é a novidade metodológica.** Ele nasceu de uma medida do
`metodo_aplicacao_sem_rotulo_local_v1.md`: 0% dos pontos de Curitiba caem na
faixa de `elevation_m` do treino, diferença padronizada de 2,76 desvios. Aquilo
era uma recomendação num documento; aqui é um portão avaliado por requisição.
Abaixo do limite de extrapolação, a variável entra como limitação declarada;
acima dele, o serviço recusa em vez de extrapolar.

---

## 5. Maturidade da região — o eixo de predição seletiva

| maturidade | significa |
|---|---|
| `validado` | modelo próprio, sinais corretos, IC das causais sem cruzar zero, AUC na faixa |
| `mvp_local` | modelo próprio da região, mas algum critério de leitura não atingido |
| `transferencia_caracterizada` | a região não entrou no treino; declara-se distância de domínio, não acerto contra rótulo local |
| `nao_suportada` | sem modelo mapeado — `region_not_supported` |

**Nenhuma região do projeto está hoje em `validado`.** Recife é `mvp_local`
porque tem modelo próprio que não atinge o critério; Curitiba é
`transferencia_caracterizada` por construção; Petrópolis é `nao_suportada`.

### A decisão que fica exposta, e não escondida

Quando o modelo da região existe mas não atinge o critério, há duas posturas
defensáveis, e a constante `POLITICA_CRITERIO_NAO_ATINGIDO` escolhe entre elas:

- **`declara`** (padrão): devolve o escore com a falha escrita em `limitacoes` e
  maturidade rebaixada. Coerente com "resultado negativo é publicado" — esconder
  o escore esconderia também o quanto ele é fraco.
- **`recusa`**: devolve `insufficient_data`. Coerente com *fail-closed* estrito.

A escolha muda o que o produto responde para Recife. Está numa constante
visível, não enterrada numa condição.

---

## 6. As três respostas reais

```
recife           status=ok   maturidade=mvp_local
                 escore=0,5346  IC95 [0,4561; 0,6079]  12 pontos, unidade=área
                 limitação: negativo por ausência de registro
                 limitação: positivos e negativos dividem 5 das 205 datas
                 limitação: HAND não separa as classes em Recife
                 limitação: critério de leitura não atingido — IC de hand_m
                            cruza zero; AUC 0,6409 fora da faixa

curitiba         status=ok   maturidade=transferencia_caracterizada
                 escore=0,1822  IC95 [0,0615; 0,2543]  12 pontos
                 domínio: elevation_m 0,0% | slope_deg 91,7% | hand_m 83,3%
                          | twi_dinf 100%
                 limitação: extrapolação de domínio em elevation_m

petropolis       status=region_not_supported
                 gate=modelo_para_a_regiao
                 "sem modelo ajustado e validado; nenhum escore por analogia"
```

### Um erro que a construção pegou, e vale registrar

A primeira versão mapeou o modelo de serra para servir Petrópolis — o modelo
existe, é coerente com os critérios, e Petrópolis é serra. **A demonstração
devolveu escore 0,5049 para Petrópolis, com valores de HAND que eu mesmo tinha
inventado para preencher a requisição.**

Está corrigido, e a correção é conceitual, não de código: **ter modelo de serra
ajustado não é o mesmo que ter região mapeada**. `regioes_servidas` do modelo de
serra ficou vazio, e a demonstração de Petrópolis passou a mandar camadas
vazias — que é o estado real da região na base.

Vale registrar também que dois documentos do projeto divergem sobre isso:
`esboco_telas_minimas_produto_v1.md` declara Petrópolis como
`region_not_supported`, e `ext_criterios_de_acerto_v1.md` §6 diz que "para
PREDIZER em Petrópolis não falta nada; o que falta é a validação". Enquanto a
divergência não for resolvida, o serviço segue a leitura conservadora — a mesma
que o manuscrito entregue declara.

---

## 7. A camada de explicação

Gerada **por regras**, lendo o *payload* já decidido. O plano declarava essa
rota como alternativa caso a explicação divergisse do *payload*; adotá-la desde
o início torna a divergência impossível por construção, e não por verificação
posterior.

A contribuição de cada variável é `coeficiente_padronizado × z(valor)`, medida
no mesmo espaço em que o modelo decide; a frase só descreve o sinal dessa
contribuição, ordenada por peso. Não há texto livre nem modelo de linguagem
envolvido.

```
curitiba: altura acima da drenagem (HAND) = 13.528 m pesa contra;
          índice topográfico de umidade (TWI) = 7.573 pesa contra;
          elevação = 913.867 m pesa contra;
          declividade = 4.351 graus pesa a favor.
```

Três testes guardam a propriedade: o sinal de cada frase bate com o sinal da
contribuição, as variáveis explicadas são exatamente as usadas, e a ordem é a de
peso decrescente.

---

## 8. O que fica para depois, e é honesto dizer

- **Transporte HTTP.** `inferir()` é a função; falta o servidor que a expõe. É
  trabalho de infraestrutura, não de método.
- **Nenhuma região em `validado`.** Chegar lá depende de negativo melhor em
  Recife e de inventário em Petrópolis — não de código.
- **O escore é de predisposição do terreno, não previsão de evento.** Está no
  `nao_e` de todo *model card*.
- **A unidade de resposta é a área.** O escore da AOI é a média dos pontos, e o
  IC é o percentil das médias por réplica. Responder por pixel exigiria validar
  por pixel, que o projeto não faz.
