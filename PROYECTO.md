# PROYECTO — Contexto general

> Archivo de contexto del proyecto. Su propósito es permitir continuar el trabajo
> desde otra computadora/sesión sin perder el contexto de lo conversado.

Última actualización: **2026-08-04**

---

## 1. Resumen del proyecto

**Curso**: Especialización en Machine Learning Engineering — Curso II (MLE2).

**Título**: Pronóstico de demanda multi-series (series de tiempo) con MLflow y
agente genAI de insights.

**Fecha máxima de entrega**: Domingo 30 de agosto de 2026.
Entregable: repositorio de GitHub **v1.0.0** (sin commits adicionales después de esa fecha).

**Estudiante**: Jaime Ramos.

---

## 2. Requisitos del entregable (del PDF `Proyecto Final - Curso MLE2`)

### Repositorio de GitHub v1.0.0 con:
1. **README.md** con:
   - a. Problema de ML
   - b. Diagrama de flujo del proyecto
   - c. Descripción del dataset con su diccionario de datos
   - d. Model Card (referencia: https://www.kaggle.com/code/var0101/model-cards)
   - e. Resultados con métricas de evaluación **offline y online**
   - f. Conclusiones
2. **Estructura del código**:
   - a. Carpeta `notebooks/` (preprocesamiento + ML)
   - b. Carpeta `data/` (.csv, .txt, .parquet)
   - c. Módulo de código reusable
   - d. Scripts de ejecución (preprocesamiento, entrenamiento y predicción)
3. **Link con evidencia de experimentos en MLflow** con artefactos y un modelo
   productivo (ej. https://dagshub.com/abdala9512/fake-news-poc/experiments).
   - Cada experimento debe tener métricas, parámetros y artefactos.
4. **Release v1.0.0** con sus notas de release.
5. **Ramas Main y Development** (al menos una PR cerrada exitosamente).
6. **Documentación de la estrategia git**.
7. [OPCIONAL] `.gitignore`, `requirements.txt`, instrucciones de ejecución.

### Evaluación
| Componente | Tipo | Porcentaje |
|---|---|---|
| Repositorio de GitHub | Obligatorio | 25% |
| Modelo de ML (incluye técnica genAI: RAG/agentes/MCPs, evaluada con métricas coherentes y usando MLflow) | Obligatorio | 50% |
| Buenas prácticas de desarrollo (commits, PRs, releases) | Obligatorio | 15% |
| Documentación (markdown/notebooks, docstrings, naming) | Obligatorio | 10% |
| Reto ML 1 (contenedores/Docker) | Opcional | +10% |
| Reto ML 2 (despliegue en Azure) | Opcional | +10% |

### Restricciones del curso
- Dataset de libre elección, **máximo ~100 MB**, sin ingeniería de datos compleja.
- Fuentes sugeridas: Kaggle, Google Dataset Search, UCI.
- Se recomienda estructura cookiecutter, GitHub Flow, `.gitkeep` en carpetas provisionales.

---

## 3. Decisiones tomadas

| Tema | Decisión |
|---|---|
| Problema de ML | Supervisado / Regresión. Pronóstico de demanda multi-series (500 series). |
| Dataset | **Store Item Demand Forecasting Challenge** (Kaggle `demand-forecasting-kernels-only`): 913k filas, 2013-2017, 10 tiendas × 50 items, ~1 MB. |
| Horizonte | 90 días (3 meses). Split temporal: train hasta 2017-09-30, holdout oct–dic 2017. |
| Métricas | RMSE, MAE, MAPE, sMAPE, **MASE**, **WAPE**, MedAE, Max Error, R², Bias. Decisión por ranking combinado MASE + WAPE. |
| Modelos | Baselines (naive, seasonal-naive, media, año anterior), SARIMA, Prophet, Holt-Winters, LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees, MLPRegressor. Estrategia global vs por-item. |
| MLflow | Local (`mlruns/`) + **DagsHub** como hosting remoto gratuito (link público de experimentos). |
| Modelo productivo | Envoltura `mlflow.pyfunc` registrada en el Model Registry con alias `Production`. |
| genAI | **Agente de insights (RAG-lite)** con **Groq** (API compatible con OpenAI, tier gratuito). Variable de entorno `GROQ_API_KEY`. |
| Componente genAI | 3 pasos: (1) contexto = stats del item/tienda + pronóstico del modelo; (2) retrieval TF-IDF + similitud coseno sobre "fichas de serie"; (3) generación con LLM de Groq (p.ej. `llama-3.3-70b-versatile`). |
| Presupuesto | **$0**. Todo gratuito: MLflow (open-source), DagsHub (free tier ~500 MB), Groq (free tier con rate limits), Kaggle (descarga gratis), cómputo local. |

---

## 4. Plan de fases

| Fase | Tareas | Estado |
|---|---|---|
| 0. Setup | Repo GitHub, estructura, .gitignore, requirements.txt, primer commit, ramas main/development | **COMPLETADA** (PR #1 fusionada en main; gh instalado) |
| 1. Datos y EDA | Descargar dataset a `data/raw/`, notebook `01_preprocesamiento_eda.ipynb`, diccionario en README | **COMPLETADA** (dataset descargado vía Kaggle CLI; notebook ejecutado con 15/15 celdas OK) |
| 2. Features | `src/features/build_features.py`: lags (1,7,30), rolling (7,30), calendario (año, mes, día, día de semana, semana), store/item categóricas; notebook `02` | Pendiente |
| 3. Modelos | Backtesting walk-forward, comparar familias de la tabla de decisión; métricas offline y online | Pendiente |
| 4. MLflow | Experimentos (params + metrics + artifacts), DagsHub remoto, registro de modelo productivo `pyfunc` con alias Production | Pendiente |
| 5. Agente genAI | `src/agent/insights_agent.py` con Groq (RAG-lite) | Pendiente |
| 6. Scripts | `scripts/preprocess.py`, `scripts/train.py`, `scripts/predict.py` ejecutables por CLI | Pendiente |
| 7. Documentación | README completo (diagrama Mermaid, Model Card, métricas offline/online, conclusiones), `docs/git_strategy.md` | Pendiente |
| 8. Release | PR final development→main, tag v1.0.0, notas de release | Pendiente |
| Opcional | Reto ML1 (Docker), Reto ML2 (Azure) | Sin decidir |

### Cronograma sugerido
- Semana 1 (4–10 ago): Setup + fase 1.
- Semana 2 (11–17 ago): features + baselines + LightGBM + MLflow local.
- Semana 3 (18–24 ago): DagsHub/MLflow remoto + modelo productivo + agente genAI.
- Semana 4 (25–30 ago): scripts, README, PR final, release v1.0.0 (último commit 30/8).

---

## 5. Datos técnicos del entorno

- **SO**: Windows (PowerShell 5.1).
- **Python**: 3.14.5. OJO: versiones muy nuevas pueden no tener wheels de algunas
  librerías; si algo falla al instalar, usar un venv con Python 3.11/3.12.
- **Instalados al inicio**: pandas 3.0.3, numpy 2.4.6, matplotlib 3.11.1, seaborn 0.13.2,
  scikit-learn 1.9.0, statsmodels 0.14.6, prophet 1.3.0, openai 2.44.0, joblib 1.5.3.
- **Instalados después (Fase 0/MLflow)**: mlflow 3.15.1, dagshub 0.7.1, pandas 2.3.3
  (mlflow lo degradó desde 3.0.3), websockets 13.1 (necesario para la UI de MLflow),
  kaggle 2.2.4, lightgbm 4.7.0, xgboost 3.4.0, pmdarima 2.1.1.
- **GitHub CLI**: `gh` 2.97.0 instalado y autenticado como JaimeRamosMiranda (los PRs los maneja la IA).
- **MLflow local**: se usa backend **SQLite** (`sqlite:///mlruns/mlflow.db`), porque
  MLflow 3.x puso el filesystem store en modo mantenimiento.
- **Git**: 2.54.0 disponible. `gh` (GitHub CLI) **no instalado** → crear el repo remoto desde la web.
- **Clave API**: Groq → `GROQ_API_KEY` en `.env` (gitignoreado). Nunca subir claves al repo.

---

## 6. Diccionario de datos del dataset (para el README)

| Columna | Tipo | Descripción |
|---|---|---|
| `date` | fecha | Día de la venta (2013-01-01 a 2017-12-31). No hay efectos de feriado ni cierres |
| `store` | int | ID de la tienda (1–10) |
| `item` | int | ID del artículo (1–50) |
| `sales` | int | Unidades vendidas (target) |

- Filas: 913,000. Series: 500 (10 tiendas × 50 items). Frecuencia: diaria.

---

## 7. Decisiones pendientes / riesgos

- [x] Crear repo en GitHub (web) y vincular el remoto local. Repo: https://github.com/JaimeRamosMiranda/ML2_Series_de_tiempo
- [x] Instalar MLflow y DagsHub y validar en Python 3.14 (funciona con SQLite + websockets 13).
- [x] Instalar `gh` y autenticar (PR #1 `development→main` fusionada exitosamente).
- [x] Autenticar Kaggle CLI (`kaggle auth login`, usuario jframosm) y aceptar las reglas de la competición `demand-forecasting-kernels-only`.
- [x] Descargar el dataset a `data/raw/` (train.csv, test.csv, sample_submission.csv). Kaggle 2.2.4 usa `kaggle competitions download -f <archivo>`; el ejecutable está en `$env:APPDATA\Python\Python314\Scripts\kaggle.exe` (no está en PATH).
- [ ] Obtener `GROQ_API_KEY` (tier gratuito) y guardarla en `.env`.
- [ ] Definir si se hacen los retos opcionales (Docker/Azure).

---

## 8. Notas de MLflow (lo esencial)

- MLflow es open-source y **gratis**. Componentes: Tracking (experimentos/runs con
  params + metrics + artifacts), Model Registry (versionado + alias como `Production`),
  pyfunc (envoltura estándar para guardar/cargar/predicir), UI (`python -m mlflow ui`).
- Flujo: `mlflow.set_experiment(...)` → `with mlflow.start_run():` → `log_params`,
  `log_metrics`, `log_artifact`, `log_model` → `mlflow.register_model(runs:/<id>/model, name)`.
- **Demo funcional**: `scripts/demo_mlflow.py` (modelo con AirPassengers). Ejecutar:
  `python scripts/demo_mlflow.py`. Deja un experimento `demo_airpassengers`, el modelo
  `airpassengers_demo` v1 con alias `champion`, y artefactos (gráfica + CSV).
- Detalles de la API en MLflow 3.x: `log_model(..., name="model")` (ya no `artifact_path`),
  alias con `MlflowClient().set_registered_model_alias(...)`, y tracking local con SQLite.
- DagsHub gratuito: `import dagshub; dagshub.init(repo_owner, repo_name, mlflow=True)`
  configura el tracking remoto. Link de experimentos: `https://dagshub.com/<user>/<repo>/experiments`.
- Cargar modelo productivo: `mlflow.pyfunc.load_model("models:/<nombre>@Production")`.

---

## 9. Pasos siguientes (próxima sesión)

1. Fase 2: `src/features/build_features.py` (lags, rolling, calendario) + notebook `02_feature_engineering.ipynb`.
2. Fase 3: modelos y backtesting (baselines, SARIMA, Prophet, LightGBM/XGBoost/CatBoost, etc.).
3. Fase 4: experimentos MLflow (local + DagsHub) y modelo productivo.
4. Fase 5: agente genAI con Groq.
5. Fase 6-8: scripts, documentación y release v1.0.0.
