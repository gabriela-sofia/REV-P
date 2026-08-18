# SUSC-16A - targets Sentinel-1/SAR

Status: review-only. Nenhum raster pesado foi baixado.

O SUSC-16A substitui a tentativa de geocodificacao textual por uma estrategia de footprints observacionais, combinando geometrias locais, fontes oficiais/tecnicas e planejamento Sentinel/SAR. A etapa mantem todos os vinculos review-only, nao cria ground truth, nao libera treino supervisionado e nao cria score v7 automatico.

- Targets Sentinel-1 gerados: 161
- Execucao externa requerida: true
- Metodo esperado: Sentinel-1 GRD before/after, mudanca VV/VH, filtro speckle, remocao de agua permanente, mascaras HAND/slope/urbana, unidade minima de mapeamento e poligonizacao.
