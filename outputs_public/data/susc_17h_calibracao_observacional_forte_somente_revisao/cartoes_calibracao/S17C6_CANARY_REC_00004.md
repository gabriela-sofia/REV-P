# Cartao de calibracao S17C6_CANARY_REC_00004

## Identificacao

- Canario: `S17C6_CANARY_REC_00004`
- Evento: `S17C_REF_0063`
- Vinculo/geometria: `S17C5_LINK_00066` / `S17C5_GEOM_0063`
- Cidade/regiao: Recife / REC
- Fenomeno/data: flood_inundation_alagamento / 2022-05-24..2022-05-30

## Features fisicas diretas (17G)

- elevacao=49.3445 m; declividade=7.4668 graus; HAND=19.9464 m
- distancia hidrica media=316.3700 m; TWI=7.8824; fluxo(log)=1.1544

## Features espectrais (17E, pre-evento)

- NDVI=0.0438; NDWI=-0.0213; MNDWI=0.1467; NDBI=-0.1675

## Chuva (17E, CHIRPS pre-evento)

- CHIRPS_3d=20.5343; CHIRPS_7d=20.5343; CHIRPS_30d=139.6736

## Componentes calculados (0-1)

- fisico_topografico=0.3069; urbano_espectral=0.4472; umidade_espectral=0.5314; chuva_gatilho=0.4087; qualidade_evidencia=0.7400

## Indice observacional somente revisao

- indice=0.4045 (classe low)
- score_v6 referencia (patch oficial mais proximo)=0.6048 (classe medium)
- diferenca indice vs score_v6=-0.2003

## Simulacoes principais

- cenario_base_v6_compativel: indice=0.4045 (low), mudanca_vs_base=0.0000
- cenario_sem_documental_penalizante: indice=0.3869 (low), mudanca_vs_base=-0.0176
- cenario_gatilho_chuva_reforcado: indice=0.4004 (low), mudanca_vs_base=-0.0041
- cenario_umidade_espectral_reforcada: indice=0.4373 (medium), mudanca_vs_base=0.0328
- cenario_fisico_dominante: indice=0.3708 (low), mudanca_vs_base=-0.0337
- cenario_urbano_espectral_reforcado: indice=0.4224 (low), mudanca_vs_base=0.0179

## Divergencias detectadas

- componente_fisico_baixo_terreno_elevado;componente_espectral_baixo;componente_chuva_moderado
- Aderencia: **divergencia_multicomponente**

## Decisao de calibracao

Calibracao observacional forte somente revisao registrada com componentes, pesos e sensibilidade.
A divergencia fisica, quando presente, e preservada e nao ajustada para forcar aderencia.

## Por que nao e ground truth

O indice descreve suscetibilidade observacional review-only; nao confirma ocorrencia no patch e nao e verdade de referencia.

## Por que nao e treinavel

Sem rotulo validado, o canario nao alimenta treino supervisionado.

## Por que nao altera o score_v6

O indice usa componentes review-only e metodo reconstruido; nunca substitui nem recalibra o score_v6 oficial.

## Impacto no 17B

Amostra de um unico evento e regiao; 17B permanece sem prontidao de benchmark.
