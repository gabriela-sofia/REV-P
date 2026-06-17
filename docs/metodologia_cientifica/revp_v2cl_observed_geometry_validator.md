# REV-P v2cl - contrato de validacao de geometria observada

Este marco procura geometrias vetoriais locais candidatas e bloqueia quando faltam
vetor, CRS, proveniencia, hash, validade geometrica ou vinculo documental.

Status `VALIDATED_OBSERVED_GEOMETRY_CANDIDATE`, quando ocorrer, significa apenas
candidata observada validada para replay metodologico. Nao significa ground truth
operacional, label, negativo formal ou autorizacao de treino.

Se bibliotecas geoespaciais nao estiverem disponiveis, o validador bloqueia com
dependencia indisponivel em vez de inventar validade geometrica.

