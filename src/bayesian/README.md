# Bayesian Module (WIP)

Objetivo: ejecutar ajuste bayesiano de una curva clasificada como positiva por el clasificador RF.

## Flujo esperado

1. Recibir curva detrended en formato CSV (tiempo + flujo).
2. Ejecutar ajuste bayesiano del transito.
3. Guardar parametros estimados e incertidumbres.
4. Comparar contra parametros de literatura.
5. Exportar reporte en `src/bayesian/results/`.

## Pendiente

- Implementar `run_bayesian_fit` en `pipeline.py`.
- Definir backend de muestreo (p. ej. emcee, pymc, dynesty).
- Definir contrato de comparacion con literatura.
