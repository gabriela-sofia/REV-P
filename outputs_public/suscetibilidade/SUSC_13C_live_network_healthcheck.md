# SUSC-13C-LIVE - Healthcheck de rede

Status: **review-only**

- Verificado em: `2026-06-28T20:30:24.196190+00:00`
- Dominios testados: **13**
- Dominios alcancados: **12**
- Minimo exigido: **3**
- Rede utilizavel: **SIM**

| url | status | content-type | reachable | ms | erro |
|---|---|---|---|---|---|
| https://dados.gov.br/ | 200 | text/html | True | 150 |  |
| https://www.gov.br/mdr/pt-br | 200 | text/html;charset=utf-8 | True | 137 |  |
| https://www.sgb.gov.br/ | 200 | text/html;charset=UTF-8 | True | 817 |  |
| https://www.cemaden.gov.br/ | 200 | text/html;charset=utf-8 | True | 423 |  |
| https://www.ana.gov.br/ | 200 | text/html;charset=utf-8 | True | 365 |  |
| https://dados.recife.pe.gov.br/ | 200 | text/html; charset=utf-8 | True | 677 |  |
| https://www.apac.pe.gov.br/ | 200 | text/html; charset=utf-8 | True | 964 |  |
| https://www.recife.pe.gov.br/ | 200 | text/html; charset=UTF-8 | True | 339 |  |
| https://www.petropolis.rj.gov.br/ |  |  | False | 15284 | TimeoutError: The read operation timed o |
| https://www.inea.rj.gov.br/ | 200 | text/html; charset=UTF-8 | True | 297 |  |
| https://www.curitiba.pr.gov.br/ | 403 |  | True | 38 | HTTPError 403 |
| https://geocuritiba.ippuc.org.br/ | 200 | text/html | True | 326 |  |
| https://www.defesacivil.pr.gov.br/ | 200 | text/html; charset=UTF-8 | True | 742 |  |

Rede real disponivel: a sprint de aquisicao live pode prosseguir.
