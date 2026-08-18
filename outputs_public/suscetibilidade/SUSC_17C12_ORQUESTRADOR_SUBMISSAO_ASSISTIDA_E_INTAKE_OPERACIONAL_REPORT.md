# SUSC-17C12 - Orquestrador de submissao assistida e intake operacional

## O que o 17C11 resolveu
O SUSC-17C11 associou as 9 solicitacoes formais do 17C10 a canais oficiais, candidatos ou ausentes: 5 canais confirmados, 2 candidatos e 2 sem canal externo oficial.

## O que o 17C12 acrescenta
O SUSC-17C12 transforma esse inventario em fila operacional, pacotes copiaveis, mensagens finais, modos CLI seguros, registro local de acao humana e intake futuro de respostas reais. O marco nao e apenas dossie estatico: ele cria comandos auditaveis para o agente preparar, orientar e registrar, sem executar acao externa automaticamente.

## Como a fila funciona
A fila possui 9 linhas. Pedidos com canal confirmado ficam `ready_to_prepare`; canais candidatos ficam `needs_manual_channel_verification`; canais ausentes ficam `blocked_no_official_channel`.

## Modos do orquestrador
`plan`, `prepare`, `open-channel`, `submit-assisted`, `record-submission`, `intake-response`, `status` e `audit` foram implementados. `prepare`, `status` e `audit` nao tem efeito externo. `submit-assisted` e bloqueado por padrao e exige opt-in ambiental e revisao humana.

## Preparacao e canal
Use `prepare --request-id <ID>` para gerar pacote local em `local_runs`. Use `open-channel --request-id <ID>` para imprimir URL e checklist humano. O comando nao preenche formulario, nao autentica e nao registra envio.

## Registro manual e intake
`record-submission` registra envio manual ja feito fora do agente, com data, canal e evidencia literal. `intake-response` exige arquivo local real, calcula hash e grava manifesto local; nada bruto pesado e copiado para `outputs_public`.

## Bloqueios
Envio automatico, protocolo inventado, resposta inventada, hash inventado, bruto pesado em `outputs_public`, score v7, 17B, treino, modelo, label e ground truth permanecem bloqueados.

## Prontidao
- Prontos para preparacao: 5.
- Prontos para submissao manual no build inicial: 0.
- Dependem de verificacao manual de canal: 2.
- Sem canal externo oficial: 2.

## Proximo marco recomendado
SUSC-17C13 Execucao Assistida de Submissoes Manuais para canais confirmados; verificar canais candidatos em paralelo
