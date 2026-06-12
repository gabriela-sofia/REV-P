# v2bk Recife Human Review Dossier and Data Request Pack

Esta etapa existe porque Recife (`REC_2022_05_24_30`) passou a ter um candidate reference
pendente de revisao humana apos o intake real da v2bi e a reconciliacao da v2bj. O gargalo
deixou de ser pesquisa generica: agora e (1) revisao humana e (2) pedido formal dos dados
que faltam.

A v2bk consome os resultados reais da v2bi/v2bj (mapa Charter 758, APAC mensal, ANA cota
Capibaribe, auditoria INMET) e gera, de forma estritamente aditiva: um dossie de revisao
humana, pacotes de solicitacao de vetor/CRS ao CENAD/Charter, pacotes de solicitacao de
chuva local ao Cemaden/APAC, um checklist C5/C6 e uma matriz de decisao.

Status de referencia: `CANDIDATE_REFERENCE_PENDING_HUMAN_REVIEW`. C7 continua BLOCKED porque o ground truth
final exige todos os gates resolvidos, revisao humana, vetor/CRS oficial e serie local de
chuva - nada disso esta completo. Um pacote de solicitacao nao e evidencia; um dossie nao e
ground truth; um raster Charter nao e vetor; cota ANA nao e precipitacao; PDF mensal APAC nao
e serie de estacao; proxy INMET nao e a estacao local de Recife.

Ground truth final, labels, negativos e treino = 0.
