# Pronóstico de demanda multi-series

Proyecto final — Curso II, Especialización en Machine Learning Engineering.

Pronóstico de demanda (series de tiempo múltiples, 500 series) con MLflow como
herramienta de administración de experimentos y modelos, y un agente genAI de
insights basado en Groq.

> Documento de contexto general del proyecto: ver [`PROYECTO.md`](PROYECTO.md).

## Estado del proyecto

- Fase 0 (setup del repositorio + MLflow demo): completada.
- Fase 1 (datos y EDA): completada — `notebooks/01_preprocesamiento_eda.ipynb`, datos en `data/`.
- Fase 2 (feature engineering): completada — `src/features/build_features.py`, `notebooks/02_feature_engineering.ipynb`.

## Estructura

```
├── data/            # datos crudos (raw) y procesados
├── docs/            # documentación (estrategia git, etc.)
├── notebooks/       # EDA/preprocesamiento y ML
├── scripts/         # preprocesamiento, entrenamiento y predicción
├── src/             # módulo de código reusable
│   ├── data/        # carga y preprocesamiento
│   ├── features/    # construcción de features (lags, calendario)
│   ├── models/      # entrenamiento, predicción y evaluación
│   └── agent/       # agente genAI de insights (Groq)
└── PROYECTO.md      # contexto completo del proyecto
```
