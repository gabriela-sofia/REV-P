# Revisão de literatura — alinhamento dos métodos reais do REV-P (v1)

**Contexto**: após a rodada SUSC-20G/20G2 (script D-infinity HAND/TWI validado bit a bit contra
Recife; troca do MDT de Petrópolis de MDS pra FABDEM; achado de que o candidato de
Petrópolis/Valparaíso parece assinatura de encosta, não de acumulação de água), esta revisão
busca o que a literatura real diz sobre cada método que já usamos e que deu resultado — não é
proposta de mudança de rumo, é checagem de alinhamento e identificação de refinamentos reais.

## 1. DEM de terreno nu em relevo íngreme (FABDEM) — método validado, ressalva confirmada pela literatura

FABDEM (Fathom/Bristol, base Copernicus DEM + ML treinado contra LiDAR de 12 países) reduz erro
vertical absoluto médio de 5,15 m para 2,88 m em área florestada e de 1,61 m para 1,12 m em área
urbanizada, com redução de 24% no RMSE e 135% no viés frente ao Copernicus-30 puro — exatamente
o padrão que encontramos hoje (rebaixo de −7,64 m em formação florestal, −1,96 m em área
urbanizada, quase nulo em água). A validação publicada contra LiDAR de drone em bacia florestada
de montanha confirma que FABDEM é adequado pra HAND/inundação especificamente por remover esse
viés — reforça que a escolha de hoje (FABDEM sobre GLO-30) tem base real, não só teste interno.

**Ressalva que a literatura confirma, não resolve**: a resolução de 30 m do FABDEM é a mesma do
Copernicus-DEM que o origina — não existe hoje um FABDEM de 10 m. A limitação de paridade com
Recife/Curitiba (10 m) é estrutural do produto, não um erro nosso de processamento.

## 2. HAND/TWI em terreno de serra — a literatura confirma o alerta encontrado hoje

Achado de hoje: Petrópolis/Valparaíso tem HAND=50,9 m, declividade 23,3° (próxima da mediana
regional de 23,57°), TWI abaixo da mediana — sinal de encosta, não de vale/acúmulo. A literatura
sustenta esse alerta diretamente: enchente se concentra em relevo suave de vale/piemonte, não em
encosta íngreme; terreno propenso a deslizamento é caracterizado por encostas quase planas com
canais retos e pouco espaçados; áreas convergentes com declividade ≥35° têm risco elevado de
deslizamento, e HAND+declividade juntos respondem por 27% do poder preditivo em detecção de
extensão de enchente em estudos publicados (HAND sozinho, 15%). Um estudo comparativo real em
terreno complexo (Himalaia, bacia do rio Beas) tratou exatamente da pergunta que temos agora —
se métodos baseados só em topografia (HAND/TWI/posição de encosta) discriminam bem em relevo
complexo e escasso de dados — sem resposta definitiva publicada, o que é coerente com tratarmos
o achado de hoje como reexame do candidato, não como veredito automático.

**Implicação real pro REV-P**: o critério de adjudicação atual (SUSC-20A) não usa HAND/TWI como
filtro — foi adjudicado por reflectância + corroboração externa (rio mapeado + notícia). A
leitura HAND/TWI de hoje é a primeira vez que um critério físico independente é aplicado
retroativamente a esse candidato, e ele não bate. Isso não invalida a adjudicação original (que
usou critério diferente e válido), mas é evidência real a favor de reabrir a discussão do
candidato — decisão que continua sua, documentada como achado, não decidida aqui.

## 3. Detecção de enchente por satélite — nosso critério de reflectância absoluta tem paralelo real na literatura

MODIS MCDWD: a própria documentação da NASA confirma que nuvem é a limitação primária, mitigada
por composição multi-observação (1/2/3 dias) — exatamente por isso testamos 3+ dias ao redor da
data suspeita. Limitações adicionais documentadas: resolução grosseira (250 m), confusão de
sombra (nuvem ou relevo) com água, dificuldade sob dossel contínuo — coerente com a máscara
estrutural que encontramos em Petrópolis (provável sombra de relevo em terreno de serra).

Sentinel-2 NDWI/MNDWI: a literatura confirma que MNDWI (SWIR) separa melhor água de área urbana
que NDWI (NIR), mas ambos ainda sofrem confusão real com sombra de nuvem — nenhum dos dois
resolve sozinho. O padrão publicado mais robusto não é um único índice com limiar relativo (o
que gerou nosso falso-positivo de nuvem na primeira tentativa), e sim voto de consenso entre 2
de 3 índices (NDWI, MNDWI, AWEI) com limiar de mudança pré/pós-evento. Nosso critério de
reflectância absoluta (B08<0,15 E B11<0,15) é uma variante mais simples da mesma ideia central
— física direta, não índice relativo sozinho — mas o refinamento real disponível na literatura,
se quisermos mais robustez na próxima aquisição, é calcular também AWEI e exigir concordância de
2 dos 3 índices, não só o par NDWI/MNDWI que já usamos.

## 4. Amostragem de ponto negativo — o método de Recife está alinhado com a literatura, com uma ressalva importante

A literatura recente é explícita: amostra negativa aleatória ou "das áreas mais seguras" produz
viés sistemático — se os negativos não compartilham características de ambiente com os
positivos, o modelo superestima suscetibilidade em qualquer área não parecida com os negativos
"seguros" escolhidos. Isso é exatamente o que o v8 de Recife mostrou na prática (AUC 0,7032 com
só 3/39 bairros sobrepostos — confundimento geográfico) e o v9 corrigiu (pareamento por bairro,
AUC honesto 0,6578). Achamos isso por tentativa e erro real; a literatura confirma que é um
problema estrutural conhecido do método, não peculiaridade de Recife.

Segundo ponto de alinhamento real: a regra fundadora de Recife ("ausência de registro não é
evidência negativa; negativo é presença registrada de *outro* fenômeno") corresponde ao que a
literatura chama de aprendizado positivo-não-rotulado (positive-unlabeled) — área sem registro
de enchente é dado não-rotulado, não negativo confirmado. Um viés real permanece documentado na
literatura (persistente falta de verdade de campo faz de qualquer amostra negativa um
pseudo-negativo, nunca um negativo certo) — o que reforça, e não contradiz, a cautela do REV-P
em não amostrar negativo pra Curitiba/Petrópolis com N=1 positivo.

## 5. Orçamento EPV e Firth — a regra de ~10 é mais fraca na literatura do que supúnhamos; nosso piso de ~20-30 tem base melhor

Achado real da revisão: a regra de "10 eventos por variável" que usamos como referência
histórica tem base empírica fraca — de três estudos de simulação que a examinaram, só um
sustenta EPV mínimo de 10; revisões mais recentes (incluindo reamostragem sobre 2 milhões de
registros) apontam que EPV ≥ 20 é o que de fato elimina viés de coeficiente, e que a regra
"1 em 10" não tem justificativa robusta isolada de outros fatores do modelo.

**Implicação real, não cosmética**: o "piso técnico" de ~20-30 já documentado na linhagem de
Curitiba/Petrópolis não é conservador demais — é o número que a literatura mais recente
recomenda como mínimo real pra eliminar viés, mais alinhado à evidência atual do que a regra de
10 popularizada. O "patamar confortável" (~50-80) e "paridade Recife" (~150-270, N real do v12)
seguem com folga confortável acima desse novo piso. Sobre o uso de Firth em si: a literatura
confirma que é o método padrão pra viés de amostra pequena em eventos raros, incluindo aplicação
documentada especificamente em estudos ambientais de eventos infrequentes — sem alternativa mais
recomendada pra esse cenário. Nenhuma mudança de método sugerida aqui; a validação é que Recife
v12 já usa a abordagem certa, e o piso que a linhagem propõe pras outras duas regiões é, se
algo, apropriadamente rigoroso.

## Resumo — o que muda e o que não muda

| Domínio | Método atual do REV-P | Literatura | Ação sugerida |
|---|---|---|---|
| DEM terreno nu | FABDEM sobre Copernicus/MDS | Confirma FABDEM correto pra HAND; 30m é limite estrutural do produto | Nenhuma — documentar o limite como definitivo, não temporário |
| HAND/TWI em serra | Sem filtro topográfico na adjudicação original | Confirma que assinatura de Valparaíso é atípica pra enchente, típica de encosta | Reabrir avaliação do candidato Petrópolis/Valparaíso (decisão sua) |
| Detecção satélite | Reflectância absoluta (B08+B11) | Consenso 2-de-3 índices (NDWI+MNDWI+AWEI) é o padrão mais robusto publicado | Considerar AWEI na próxima aquisição de Sentinel-2, não retroagir nas já feitas |
| Ponto negativo | Pareamento por bairro (Recife), regra fundadora positivo-não-rotulado | Ambos confirmados como prática correta na literatura | Nenhuma — manter, replicar quando N permitir |
| EPV/Firth | Piso ~20-30, Firth em Recife v12 | EPV≥20 é o número real defensável; Firth é o método padrão pra amostra pequena | Nenhuma — o piso já documentado está correto, não frouxo |

## Fontes

- FABDEM: [Hawker et al., IOPscience](https://iopscience.iop.org/article/10.1088/1748-9326/ac4d4f); [validação LiDAR floresta de montanha](https://iopscience.iop.org/article/10.1088/2515-7620/acc56d); [Fathom](https://www.fathom.global/academic-papers/a-30-m-global-map-of-elevation-with-forests-and-buildings-removed/)
- HAND em terreno complexo: [Himalaia, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0022169423002512); [extensão multi-fonte fluvial, AGU](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022WR032039)
- Enchente vs deslizamento por topografia: [assinatura geométrica, Springer](https://link.springer.com/article/10.1007/BF00890333); [declividade e deslizamento, NHESS](https://nhess.copernicus.org/preprints/nhess-2020-87/nhess-2020-87.pdf)
- MODIS MCDWD: [NASA Earthdata, guia do usuário](https://www.earthdata.nasa.gov/s3fs-public/2022-02/MCDWD_UserGuide_RevA.pdf); [NASA Earthdata, blog de atualização](https://www.earthdata.nasa.gov/news/blog/nasa-enhances-global-flood-products-smarter-detection-flooding-release-23-year-archive)
- Sentinel-2 NDWI/MNDWI/AWEI: [mapeamento automatizado, ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0303243419303952); [MNDWI original, ResearchGate](https://www.researchgate.net/publication/301565305_Water_Bodies'_Mapping_from_Sentinel-2_Imagery_with_Modified_Normalized_Difference_Water_Index_at_10-m_Spatial_Resolution_Produced_by_Sharpening_the_SWIR_Band)
- Amostragem negativa/pseudo-ausência: [aprendizado positivo-não-rotulado, MDPI](https://doi.org/10.3390/land11111971); [amostragem hidrologicamente informada, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0022169425003919)
- EPV/Firth: [crítica à regra de 10, PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5122171/); [tamanho de amostra não é só EPV, ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0895435616300117); [Firth para eventos raros, arXiv](https://arxiv.org/abs/2101.07620)
