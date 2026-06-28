# SUSC-13B-AUTO - Relatorio de descoberta automatica

Status: **review-only** | `can_be_ground_truth=false` | `allowed_for_training=false`

## 1. Politica de rede
- Variavel de ativacao: `SUSC_13B_NETWORK=1`.
- Rede habilitada nesta execucao: **Nao**.
- Bloqueio de rede registrado (offline determinístico): **Sim**.
- robots.txt respeitado antes de qualquer crawling HTML; sem chave de API; sem
  Google Maps; sem geocoding generico; sem raster; limite de 250MB por arquivo.

## 2. Plano de consultas
Total de consultas planejadas: **69** (Recife/Petropolis/Curitiba x
templates de palavra-chave + intents site-scoped). Nenhuma consulta executa
scraping aberto; intents site-scoped sao documentados, nao raspados.

## 3. Endpoints e fontes candidatas
Total de fontes candidatas registradas: **25**.

| regiao | fontes |
|---|---|
| recife | 9 |
| curitiba | 8 |
| petropolis | 8 |

| metodo de descoberta | fontes |
|---|---|
| data_portal_html | 11 |
| ckan_api | 8 |
| arcgis_rest | 5 |
| wfs_getcapabilities | 1 |

## 4. Probes executados
Total de probes: **25**.

| status do probe | n |
|---|---|
| network_disabled | 25 |

## 5. Candidatos a download
Fontes com `download_candidate=true`: **0**.
Offline: nenhum recurso direto foi confirmado; aquisicao registra network_disabled.

## 6. Limites e governanca
- Endpoints sao raizes institucionais conhecidas; nenhum URL de download direto
  e inventado.
- Risco, alerta, previsao e suscetibilidade sao rebaixados, nunca evento observado.
- Tudo permanece review-only; nada cria ground truth, treino ou score v7.
