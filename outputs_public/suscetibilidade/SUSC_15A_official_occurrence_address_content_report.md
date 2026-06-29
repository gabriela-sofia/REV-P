# SUSC-15A - auditoria de conteudo de endereco

Status: review-only. A auditoria nao cria coordenadas, ground truth, treino ou score v7.

- Ocorrencias oficiais avaliadas: **4412**
- Ranking de precisao: **{'has_street_and_landmark': 560, 'has_street_and_number': 716, 'has_street_and_neighborhood': 1535, 'has_street_and_cross_street': 3, 'neighborhood_only': 1598}**
- Fontes/tabelas auditadas: **18**

Bairro-only e endereco sem numero/intersecao permanecem insuficientes para
calibracao fina. Numeros e referencias textuais sao tratados como candidatos
para cruzamento com bases oficiais, nunca como autorizacao para geocoding generico.
