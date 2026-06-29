# SUSC-16A - plano de janelas Sentinel/SAR por evento

Status: review-only. `score_v7_created=false`.

O SUSC-16A substitui a tentativa de geocodificacao textual por uma estrategia de footprints observacionais, combinando geometrias locais, fontes oficiais/tecnicas e planejamento Sentinel/SAR. A etapa mantem todos os vinculos review-only, nao cria ground truth, nao libera treino supervisionado e nao cria score v7 automatico.

- Janelas geradas: 161
- Eventos fonte agregados: 6366
- Estrategia principal: usar datas oficiais como gatilho temporal e AOI por extensao dos patches/regiao.

AOI municipal/regional nao e coordenada de ocorrencia. O unico objeto que podera ser cruzado com patches e o footprint orbital candidato produzido por metodo documentado.
