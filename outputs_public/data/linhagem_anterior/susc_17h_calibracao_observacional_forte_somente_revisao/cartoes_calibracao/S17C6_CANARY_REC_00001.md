# Cartao de calibracao S17C6_CANARY_REC_00001

## Identificacao

- Canario: `S17C6_CANARY_REC_00001`
- Evento: `S17C_REF_0063`
- Vinculo/geometria: `S17C5_LINK_00064` / `S17C5_GEOM_0063`
- Cidade/regiao: Recife / REC
- Fenomeno/data: flood_inundation_alagamento / 2022-05-24..2022-05-30

## Features fisicas diretas (17G)

- elevacao=64.5593 m; declividade=8.2415 graus; HAND=25.1534 m
- distancia hidrica media=289.9900 m; TWI=7.7522; fluxo(log)=1.1106

## Features espectrais (17E, pre-evento)

- NDVI=-0.0115; NDWI=0.0364; MNDWI=0.2976; NDBI=-0.2641

## Chuva (17E, CHIRPS pre-evento)

- CHIRPS_3d=18.2800; CHIRPS_7d=39.4522; CHIRPS_30d=121.8943

## Componentes calculados (0-1)

- fisico_topografico=0.2682; urbano_espectral=0.4368; umidade_espectral=0.5835; chuva_gatilho=0.4488; qualidade_evidencia=0.7400

## Indice observacional somente revisao

- indice=0.4022 (classe low)
- score_v6 referencia (patch oficial mais proximo)=0.6048 (classe medium)
- diferenca indice vs score_v6=-0.2026

## Simulacoes principais

- cenario_base_v6_compativel: indice=0.4022 (low), mudanca_vs_base=0.0000
- cenario_sem_documental_penalizante: indice=0.3844 (low), mudanca_vs_base=-0.0178
- cenario_gatilho_chuva_reforcado: indice=0.4094 (low), mudanca_vs_base=0.0072
- cenario_umidade_espectral_reforcada: indice=0.4478 (medium), mudanca_vs_base=0.0456
- cenario_fisico_dominante: indice=0.3572 (low), mudanca_vs_base=-0.0450
- cenario_urbano_espectral_reforcado: indice=0.4178 (low), mudanca_vs_base=0.0156

## Divergencias detectadas

- componente_fisico_baixo_terreno_elevado;componente_chuva_moderado
- Aderencia: **divergencia_fisica**

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
