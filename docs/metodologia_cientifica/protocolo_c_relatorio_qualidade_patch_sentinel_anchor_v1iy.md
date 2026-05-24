# Relatorio v1iy - Auditoria local de qualidade dos patches Sentinel

## Escopo

A v1iy avaliou somente os patches locais gerados pela v1ix. Nao houve download novo, criacao de label, target, treino ou reabertura de Protocolo B.

Anchor oficial:

- documento: ANEXO-II CPRM;
- localidade: Moinho Preto, Petropolis;
- data da vistoria: 19-02-2022;
- fenomeno: movimento de massa;
- latitude: -22.484251;
- longitude: -43.211257.

## Achado principal

A v1ix gerou um par real de patches Sentinel-2, pre e pos-evento, centrados no anchor oficial. A v1iy confirmou que os dois patches possuem qualidade local minima para revisao:

- bandas esperadas presentes;
- shape 96 x 96;
- CRS EPSG:32723;
- pixels validos completos;
- nodata igual a zero;
- sem NaN/inf;
- variacao por banda suficiente;
- NDWI aproximado e NDBI aproximado computaveis.

## Nuvem e risco local

O metadado global de nuvem da cena pre-evento e alto (`90.359181`). Esse valor e importante como alerta, mas nao representa automaticamente a fracao de nuvem dentro do recorte do anchor.

O patch local nao contem SCL/QA60, entao a v1iy nao mediu diretamente a fracao local de nuvem. Por isso, a decisao correta nao e reprovar automaticamente o patch pre-evento. A decisao correta e marcar o par como `PRE_PATCH_CLOUD_RISK_HIGH`.

## Decisao do par

Status do par: `PRE_PATCH_CLOUD_RISK_HIGH`.

Interpretacao:

- o par permanece candidato de patch de referencia;
- o par permanece candidato para revisao multimodal;
- a cena pre-evento exige cautela por risco de nuvem sem mascara local;
- o par nao vira ground truth operacional;
- o par nao cria label;
- o par nao libera treino.

## Uso recomendado

Na metodologia, o resultado deve ser descrito como um par Sentinel real centrado em anchor oficial e aprovado em QA local basico, mas com ressalva de nuvem para o pre-evento. Em resultados, o par pode ser citado como evidencia de avanço operacional do Protocolo C, nao como validacao final de evento.

Para uma etapa posterior mais forte, e necessario obter SCL/QA60 local ou uma cena pre-evento alternativa com menor risco de nuvem no recorte.
