# SUSC-17G Extracao direta de atributos fisicos/topograficos dos canarios observacionais

## Estado herdado do 17F

- Branch: `marco/reavaliacao-pos-mapbiomas-sensibilidade-territorial`
- HEAD: `b408e6c`
- Status final 17F herdado: 17F_EXPLORATORIA_APRIMORADA_COM_COMPONENTE_FISICO
- Referencias comparativas no 17F: 5
- Status 17B herdado: 17B_PARCIAL_COM_CALIBRACAO_EXPLORATORIA_SEM_PRONTIDAO_BENCHMARK
- `score_v6` alterado: False
- `score_v7` criado: False

## Objetivo da extracao direta

Extrair diretamente elevacao, declividade, HAND, proximidade hidrica, TWI e acumulacao de fluxo
dos 5 canarios `S17C6_CANARY_REC_00001..00005`, superando a referencia comparativa distante do 17F.

## Auditoria dos insumos locais

- S17G_INSUMO_0001 (modelo_digital_elevacao_dominio_bacia): cobre_aoi=true; status=cobre_aoi_e_utilizavel
- S17G_INSUMO_0002 (hidrografia_oficial_faixas_marginais): cobre_aoi=true; status=cobre_aoi_e_utilizavel
- S17G_INSUMO_0003 (modelo_digital_elevacao_janela_sul): cobre_aoi=false; status=nao_cobre_aoi
- S17G_INSUMO_0004 (manifesto_dem_bacia): cobre_aoi=true; status=cobre_aoi_e_utilizavel

O modelo de elevacao do dominio de bacia (17C38, Copernicus GLO-30) cobre a area dos canarios, ao
contrario da janela sul do 17C36. A hidrografia oficial (faixas-marginais, 17C35) tambem cobre a
area, permitindo a proximidade hidrica direta.

## Resultado por canario

- S17C6_CANARY_REC_00002: modo=direta_por_dem_e_hidrografia_local; elevacao=43.4218 m; declividade=5.4818 graus; HAND=12.2470 m; dist_agua_min=19.4700 m; completas=true
- S17C6_CANARY_REC_00001: modo=direta_por_dem_e_hidrografia_local; elevacao=64.5593 m; declividade=8.2415 graus; HAND=25.1534 m; dist_agua_min=155.7400 m; completas=true
- S17C6_CANARY_REC_00003: modo=direta_por_dem_e_hidrografia_local; elevacao=43.9214 m; declividade=7.5026 graus; HAND=19.2429 m; dist_agua_min=34.2100 m; completas=true
- S17C6_CANARY_REC_00004: modo=direta_por_dem_e_hidrografia_local; elevacao=49.3445 m; declividade=7.4668 graus; HAND=19.9464 m; dist_agua_min=19.4700 m; completas=true
- S17C6_CANARY_REC_00005: modo=direta_por_dem_e_hidrografia_local; elevacao=41.3008 m; declividade=5.7226 graus; HAND=16.6967 m; dist_agua_min=34.2100 m; completas=true

- Features diretas completas: 5
- Features diretas parciais: 0
- Podem calibracao forte review-only: 5

## Features diretas completas/parciais/ausentes

Todas as seis features fisicas-alvo foram extraidas diretamente para os canarios a partir do modelo
de elevacao local e da hidrografia oficial. A qualidade e review-only, metodo reconstruido (17C36),
resolucao aproximada de 92 m; nao provada equivalente ao pipeline oficial.

## Insumos externos minimos necessarios

A fila `fila_insumos_externos_features_fisicas.csv` registra, como refinamento opcional de baixa
prioridade, um modelo de elevacao de 30 m nativo por canario para aprimorar HAND, TWI e fluxo. A
extracao ja esta concluida a ~92 m, entao esse refinamento nao e bloqueante.

## Funcionamento do extrator

O extrator reutiliza o pipeline hidrologico numpy do 17C36 (preenchimento de depressoes por
inundacao prioritaria, direcao D8, acumulacao de fluxo, HAND por celula de drenagem a jusante, TWI)
aplicado ao modelo de elevacao do dominio de bacia. A proximidade hidrica usa distancia
ponto-segmento a hidrografia oficial. O modo de execucao futura le a fila de insumos, valida os
arquivos esperados e falha com erro claro se faltar insumo, sem baixar nada automaticamente.

## Simulacao exploratoria com features diretas

- S17C6_CANARY_REC_00002: descritor_topo_direto=0.4032; sem_fisico=0.3469 (low); com_fisico_direto=0.3706 (low); comparativo_17f=0.4516; delta=-0.0810
- S17C6_CANARY_REC_00001: descritor_topo_direto=0.1723; sem_fisico=0.4499 (medium); com_fisico_direto=0.3330 (low); comparativo_17f=0.5315; delta=-0.1985
- S17C6_CANARY_REC_00003: descritor_topo_direto=0.2915; sem_fisico=0.3589 (low); com_fisico_direto=0.3305 (low); comparativo_17f=0.4449; delta=-0.1144
- S17C6_CANARY_REC_00004: descritor_topo_direto=0.2469; sem_fisico=0.3550 (low); com_fisico_direto=0.3095 (low); comparativo_17f=0.4356; delta=-0.1261
- S17C6_CANARY_REC_00005: descritor_topo_direto=0.3506; sem_fisico=0.3459 (low); com_fisico_direto=0.3479 (low); comparativo_17f=0.4190; delta=-0.0711

O descritor topografico direto (review-only) contrasta com a referencia comparativa do 17F: a
extracao direta descreve o terreno do proprio canario, corrigindo a leitura feita a partir de um
patch oficial distante. Em todas as linhas: score_oficial=false, substituir_score_v6=false,
usar_em_treino=false, ground_truth=false, score_v7_allowed=false.

## Impacto na calibracao forte

Com features fisicas diretas (elevacao, declividade, HAND, TWI, fluxo) mais espectral e chuva reais,
a calibracao forte review-only fica possivel para os canarios. Ela permanece review-only, com metodo
reconstruido, sem virar oficial, sem ground truth, sem treino e sem score_v7.

## Gate final 17G

- canarios_processados: passou=true (5 / 5)
- features_diretas_completas: passou=true (5 / >=0)
- features_diretas_parciais: passou=true (0 / >=0)
- aguardando_insumo_externo: passou=true (0 / >=0)
- pode_calibracao_forte_review_only: passou=true (5 / >=0)
- exploratoria_com_features_diretas: passou=true (5 / >=0)
- ground_truth_zero: passou=true (0 / 0)
- trainable_zero: passou=true (0 / 0)
- score_v7_allowed_zero: passou=true (0 / 0)
- score_v6_intacto: passou=true (true / true)
- caminho_funcional_entregue: passou=true (true / true)
- status_final_17g: passou=true (17G_CALIBRACAO_FORTE_REVIEW_ONLY_POSSIVEL / enum)

- Status final: **17G_CALIBRACAO_FORTE_REVIEW_ONLY_POSSIVEL**
- Caminho funcional: **calibracao_forte_review_only_possivel_com_features_diretas**

## Impacto no 17E e no 17B

- 17E: 17E_CALIBRACAO_FORTE_REVIEW_ONLY_POSSIVEL_COM_FEATURES_FISICAS_DIRETAS.
- 17B: 17B_PARCIAL_COM_CALIBRACAO_EXPLORATORIA_SEM_PRONTIDAO_BENCHMARK (segue sem prontidao de benchmark: sem ground truth, amostra
  concentrada em um evento, sem score_v7).

## Proximo marco recomendado

SUSC-17H: montar a calibracao forte review-only completa dos canarios combinando as features fisicas
diretas, espectrais e de chuva, com auditoria de incerteza e refinamento opcional para modelo de
elevacao de 30 m, mantendo tudo review-only.
