# SUSC-15B - Forensic Precision Rescue

Status: review-only. `allowed_for_training=false`; `can_be_ground_truth=false`.

SUSC-15B e review-only: nao aceita bairro-only como sucesso, nao aceita street segment sem numero/intersecao como calibracao, nao cria ground truth, nao libera treino supervisionado e nao cria score v7 automatico.

## Resultado
SUSC-15B executou uma rodada densa de auditoria forense sobre campos ocultos, identificadores internos, fontes profundas candidatas, joins oficiais e linkage patch-evento. Nenhuma fonte adicional materializou ponto, poligono, lote, intersecao ou faixa numerica controlada suficiente para calibracao.

## Por que cada familia passou ou falhou
- IDs/codigos internos: retidos para join manual, mas sem tabela oficial auxiliar materializada.
- Colunas escondidas de coordenada/UTM: auditadas; nenhuma nova geometria oficial elegivel foi promovida.
- Tabelas oficiais auxiliares: bloqueadas por ausencia de arquivo/fonte oficial precisa nesta execucao.
- PDFs/anexos: bloqueados por ausencia de anexo local oficial parseavel.
- ArcGIS/WFS/FeatureServer: registrados como alvo, sem download automatico nesta execucao.
- Pontos criticos/156/Defesa Civil: bloqueados sem camada oficial precisa vinculavel.
- Intersecoes oficiais: bloqueadas sem gazetteer oficial materializado.
- Numeracao predial: bloqueada sem faixa/lote oficial controlado.
- Footprints Copernicus/EMS/SGB/INEA/APAC: bloqueados sem footprint oficial vinculavel aos eventos.

## Metricas
- Eventos avaliados: 4412
- Eventos elegiveis para calibracao: 0
- Links patch-evento precisos: 0
- Score v7 criado: False

## Limitacoes
Nao houve push, ground truth, treino supervisionado, modelo persistido, coordenada inventada, data inventada, bairro como coordenada ou score v7 automatico.
