# v1r7 (SUSC-21) -- Refinamento de evidencia DINO em Recife, adaptado do H2O-Net

## Papel do DINO nesta etapa

O teste A/B de v1r5/v1r6 permanece fechado: DINO nao e feature do modelo causal (Wald conjunto cluster-robusto p=0.1752 sobre 23 patches). Esta etapa nao reabre aquela decisao; ela muda o papel do DINO para evidencia a ser refinada, cujo unico produto permitido e ordenacao de fila de revisao humana.

## Adaptacao do H2O-Net (o que foi mantido e o que foi descartado)

Mantido: a estrutura de sementes de alta confianca com limiar alto (phi_H) e baixo (phi_L) sobre um indicador continuo, o conjunto explicitamente nao-resolvido, e o score assinado (proximidade ao positivo menos proximidade ao negativo), que e o que o mapa de distancia adaptativo normalizado codifica. Adaptado: no H2O-Net a distancia e euclidiana em COORDENADA DE PIXEL dentro de uma imagem, valida por contiguidade espacial da agua; aqui a unidade e o patch inteiro e contiguidade geografica seria confundidor, entao a distancia passou para o espaco de embedding (cosseno 768D) e a adjacencia geografica virou teste obrigatorio de QA. A semente tambem e mais forte: pontos SEDEC do v12 em vez de MNDWI limiarizado. Descartado: o refiner CNN e a pseudo-mascara que supervisiona a rede de segmentacao -- esse passo GERA label, proibido no projeto. O pipeline para no ranqueamento.

## Pre-registro

phi_H=0.75, phi_L=0.25, suporte minimo=2 pontos por patch, unidade = patch, criterio de utilidade fixado antes de rodar: permutacao p<0.05 e nenhum confundidor (adjacencia geografica, proveniencia do patch, n de pontos) significativo. Sensibilidade com suporte minimo 1 reportada junto.

## Resultado

4 patches semente positiva, 4 semente negativa, 15 alvos de refinamento sobre 23 patches. Separacao semente-a-semente observada=-0.1183 (permutacao p=0.9983, 20000 sorteios; enumeracao exata das 70 particoes: p=1.0000, posicao 70, menor p alcancavel por este desenho=0.0143). Score x frac_positive em todos os patches: rho=-0.1671 p=0.4459. Confundidores: adjacencia geografica rho=-0.0283 p=0.6543; proveniencia rho=0.0000 p=1.0000; n de pontos rho=0.2996 p=0.2780. Veredito: REFINEMENT_SIGNAL_NOT_ESTABLISHED_QUEUE_IS_UNORDERED_NULL_RESULT.

## Limitacoes

n efetivo = 23 patches, nao os 112 pontos v12 que caem dentro deles -- a mesma pseudorreplicacao que fechou v1r5/v1r6 continua sendo o teto desta analise. Nenhum candidato aqui e positivo, label ou feature; a promocao de qualquer linha depende de revisao humana e decisao explicita de orientacao. Recife apenas; nenhum embedding novo, nenhuma imagem nova.
