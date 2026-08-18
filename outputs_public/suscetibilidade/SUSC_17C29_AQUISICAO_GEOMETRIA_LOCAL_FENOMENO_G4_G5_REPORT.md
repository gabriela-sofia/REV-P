# SUSC-17C29 - Aquisicao de geometria local oficial e separacao de fenomeno para G4/G5

## Objetivo
Aquisicao DIRIGIDA a nivel local (bairro/logradouro/patch) para tentar obter evidencia oficial/institucional suficiente para G4 (vinculo espacial patch-level) e G5 (separacao de fenomeno hidrologico x movimento de massa) do evento REC_2022_05_24_30 (Grande Recife, maio 2022). Este marco nao reprova o evento (ja provado em 17C27/17C28): foca patch-level e fenomeno local.

## Aquisicao local
- Plano de busca local: 80 linhas (bairro/logradouro, abrigo/ocorrencia, hidrologico local, geometria).
- Tentativas de busca/aquisicao local: 60.
- Links locais seguidos: 8.
- Artefatos oficiais/institucionais locais adquiridos: 7 (NOVOS, distintos dos 4 do 17C28); parseados: 7.

## Candidatos, geometria e fenomeno
- Candidatos locais: 7.
- Geocodaveis (bairro/logradouro, sem coordenada): 7; patch-level (coordenada/poligono): 0.
- Hidrologicos especificos (alagamento/inundacao documentado): 1; mistos (inundacao + deslizamento): 1.
- Avaliacoes G4/G5: 14; Ground Reference Candidates avaliados: 7; aceitos: 0.
- G4_true_count=0, G5_true_count=0.

## Resultado cientifico (honesto - Resultado B: bloqueio honesto)
- Fontes oficiais institucionais (Agencia Brasil/EBC) citam bairros do Grande Recife (Ibura, Jardim Monte Verde, Barro, Muribeca, Curado, Jaboatao, Olinda, Guararapes) e ao menos uma documenta alagamento/inundacao, mas:
- G4 permanece false: a localizacao e geocodavel apenas a nivel de bairro/municipio, sem coordenada ou poligono patch-level; incerteza >=3km incompativel com o patch/buffer. Nenhuma coordenada foi inventada.
- G5 permanece false: onde o fenomeno hidrologico aparece, vem MISTO com deslizamento; fenomeno misto nunca vira G5. Nenhuma fonte separa o hidrologico do movimento de massa por local.
- Nenhum Ground Reference Candidate foi aceito; 17B permanece bloqueado.

## Guardrails
- Cidade/municipio nao virou patch-level sem incerteza; fenomeno misto nao virou G5; noticia comercial nao virou Ground Reference; sensor/CHIRPS nao viraram evento observado; nenhum ground truth, label, treino, score v7 ou patch oficial; score v6 intacto.

## minimum_success_achieved: True

## Proximo marco recomendado
SUSC-17C30 Aquisicao de geometria vetorial oficial (poligono/mancha/coordenada de ocorrencia ou setor de risco SGB/CPRM) e classificacao de fenomeno por ponto para tentar G4/G5 patch-level com coordenada real
