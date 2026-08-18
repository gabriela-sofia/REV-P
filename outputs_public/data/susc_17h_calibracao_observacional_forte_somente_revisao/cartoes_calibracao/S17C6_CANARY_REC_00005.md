# Cartao de calibracao S17C6_CANARY_REC_00005

## Identificacao

- Canario: `S17C6_CANARY_REC_00005`
- Evento: `S17C_REF_0063`
- Vinculo/geometria: `S17C5_LINK_00067` / `S17C5_GEOM_0063`
- Cidade/regiao: Recife / REC
- Fenomeno/data: flood_inundation_alagamento / 2022-05-24..2022-05-30

## Features fisicas diretas (17G)

- elevacao=41.3008 m; declividade=5.7226 graus; HAND=16.6967 m
- distancia hidrica media=190.4000 m; TWI=8.1855; fluxo(log)=1.1759

## Features espectrais (17E, pre-evento)

- NDVI=0.2494; NDWI=-0.2230; MNDWI=-0.1110; NDBI=-0.1148

## Chuva (17E, CHIRPS pre-evento)

- CHIRPS_3d=20.5343; CHIRPS_7d=20.5343; CHIRPS_30d=139.6736

## Componentes calculados (0-1)

- fisico_topografico=0.4208; urbano_espectral=0.4090; umidade_espectral=0.4165; chuva_gatilho=0.4087; qualidade_evidencia=0.7400

## Indice observacional somente revisao

- indice=0.4309 (classe medium)
- score_v6 referencia (patch oficial mais proximo)=0.6048 (classe medium)
- diferenca indice vs score_v6=-0.1739

## Simulacoes principais

- cenario_base_v6_compativel: indice=0.4309 (medium), mudanca_vs_base=0.0000
- cenario_sem_documental_penalizante: indice=0.4147 (low), mudanca_vs_base=-0.0162
- cenario_gatilho_chuva_reforcado: indice=0.4196 (low), mudanca_vs_base=-0.0113
- cenario_umidade_espectral_reforcada: indice=0.4313 (medium), mudanca_vs_base=0.0004
- cenario_fisico_dominante: indice=0.4265 (medium), mudanca_vs_base=-0.0044
- cenario_urbano_espectral_reforcado: indice=0.4298 (medium), mudanca_vs_base=-0.0011

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
