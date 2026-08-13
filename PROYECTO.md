# PROYECTO — Contexto general

> Archivo de contexto del proyecto. Su propósito es permitir continuar el trabajo
> desde otra computadora/sesión sin perder el contexto de lo conversado.

Última actualización: **2026-08-13** (sesión 6: Fase 6-7 completadas — `scripts/predict.py`, README completo con secciones b/d/e/f, evidencia de DagsHub re-subida con `scripts/upload_dagshub_evidence.py`)

### Registro de PRs (todas cerradas exitosamente)
| PR | Contenido |
|---|---|
| #1 | Fase 0: setup del repositorio + demo de MLflow |
| #2 | Fase 1: dataset y notebook de EDA |
| #3 | Fase 2: feature engineering |
| #4 | Separar material original de clase del repo |
| #5 | README secciones a (problema de ML) y c (dataset/diccionario) |

### Sesión 6 (sin PR todavía, cambios en rama `development`)
- **Fase 6 — `scripts/predict.py`** (CLI): carga el modelo productivo desde
  `models/demand_forecast` (o `models:/demand_forecast@Production`), arma el marco
  historial+test del subconjunto, pronostica ene-mar 2018 **día a día** (recursivo,
  mismo enfoque del notebook 06) y guarda la submission. Probado: smoke test 3 series
  y corrida completa → 13,500 filas, media 52.5 u/día (idéntico al notebook 06).
  Flags: `--model-uri`, `--output`, `--max-series`.
- **Fase 7 — README completo**: secciones **b** (diagrama Mermaid + link DagsHub),
  **d** (Model Card: LightGBM, métricas offline/online, limitaciones), **e**
  (resultados offline 9 modelos + online 5 modelos + componente genAI) y **f**
  (conclusiones). Corregido el criterio de decisión en 1.4 (ranking MASE + WAPE,
  filtro |rel_bias|<=5%, desempate por sesgo). `docs/git_strategy.md` ya estaba listo.
- **Evidencia DagsHub re-subida (resuelto)**: el remoto aparecía vacío (0 experimentos);
  el run `fase4_remoto_production` de la sesión 4 no persistió en DagsHub. En su lugar se
  re-subió la evidencia local con `scripts/upload_dagshub_evidence.py` (sin re-ejecutar el
  notebook 06): experimento `demand_forecast_fase3` (run `lightgbm_production` con
  métricas MASE/WAPE/bias, artefactos `predictions.csv` + `forecast_plot.png` y el modelo),
  experimento `demand_forecast_insights` (los 3 insights LLM con contexto, series
  similares e insight como artefactos) y el modelo `demand_forecast` **v2 → alias
  Production** (READY). Link: <https://dagshub.com/jaimeramos124/ML2_Series_de_tiempo/experiments>
  ⚠️ Detalle técnico: DagsHub no persiste el directorio `model` que crea
  `mlflow.sklearn.log_model` (su sync es asíncrono), así que el script sube los archivos
  del modelo exportado bajo `artifact_path="model"` y espera a que aparezcan antes de
  registrar el modelo.

### Sesión 5 (sin PR todavía, cambios en rama `development`)
- **Fase 5 — agente genAI en `src/agent/insights_agent.py` + notebook `07_agente_insights.ipynb`**:
  - RAG-lite en 3 pasos: (1) **contexto** = estadísticas del histórico + pronóstico
    ene-mar 2018 de la serie; (2) **retrieval** = corpus de 150 fichas de serie con
    TF-IDF + similitud de coseno (excluye la propia serie; se valida la
    auto-coincidencia con similitud 1.0); (3) **generación** = LLM de Groq
    `llama-3.3-70b-versatile` (API compatible con OpenAI) o generador heurístico
    (baseline determinista) si no hay `GROQ_API_KEY`.
  - El notebook analiza 3 series (tienda 3/art. 7, tienda 1/art. 1, tienda 10/art. 49)
    y registra cada insight en el experimento `demand_forecast_insights` con métricas
    coherentes del componente genAI: latencia total/retrieval/generación, tokens y
    `references_series` (si el texto menciona tienda y artículo). Artefactos:
    `contexto.json`, `series_similares.txt` e `insight.txt`.
  - **Ejecutado 15 celdas OK con `GROQ_API_KEY` configurada** → los insights los genera
    `llama-3.3-70b-versatile` (3 series, ~820 tokens cada uno, `references_series=1`).
    El generador heurístico queda como baseline de comparación si no hay clave.
  - **Modelo en el repo**: se exportó el modelo de producción (LightGBM, `model.skops`
    ~1.7 MB) a `models/demand_forecast/` (carpeta NUEVA, NO gitignoreada) con
    `mlflow.sklearn.save_model` y `skops_trusted_types`; carga con
    `mlflow.pyfunc.load_model("models/demand_forecast")`. Decisión del usuario: el
    entregable debe incluir el modelo en GitHub, no solo en DagsHub.
- **OJO 2 carpetas `mlruns/`** (aclarado en sesión 5): `mlruns/` (raíz) es el store
  oficial (SQLite `mlruns/mlflow.db` + artefactos de runs hechos desde la raíz);
  `notebooks/mlruns/` guarda los artefactos de los runs de la Fase 3 (03-05) creados
  con cwd=notebooks/ (la metadata está en el SQLite de la raíz). El `mlflow.db` de la
  raíz es de la demo de Fase 0 (experimento "MLflow Demo"), sin uso. `mlruns1.rar` y
  `notebooks/mlruns2.rar` son backups comprimidos sin trackear. Todo gitignoreado.

### Sesión 4 (sin PR todavía, cambios en rama `development`)
- **Fase 4 — parte local en `notebooks/06_mlflow_dagshub_pyfunc.ipynb`** (14 celdas OK):
  - Se consume el modelo de producción con `mlflow.pyfunc.load_model("models:/demand_forecast@Production")`
    (PyFuncModel) y se inspecciona el `MLmodel` del artefacto (flavors `python_function` + `sklearn`
    con `skops_trusted_types` para LightGBM).
  - **Re-entrenamiento automático**: al clonar en una PC nueva el `mlruns/` local está vacío; el
    notebook detecta que no hay `Production` y re-entrena **LightGBM** con el config de fase 3
    (MASE=0.581, WAPE=10.41%, rel_bias=-0.25%), lo registra como v1 y lo promueve a `Production`.
  - **Predicción recursiva ene-mar 2018**: como las features son día-relativas (lag_1, rolling),
    predecir los 90 días de una sola vez deja NaN desde el día 2 (88200 NaN). Se implementa
    **pronóstico día a día**: se rellenan las ventas predichas para que los lags/rolling del día
    siguiente sean válidos (historial real + predicciones previas).
  - **Submission** `reports/submissions/submission_production.csv` (gitignoreado): 13,500 filas
    (150 series × 90 días), media diaria 52.5 unidades vs 56.2 del historial y 52 del baseline
    del sample_submission → escala coherente. Gráfica en `reports/figures/forecast_ene_mar_2018.png`.
  - **Parte remota (DagsHub)**: la última celda se omite sin credenciales. Falta crear cuenta/repo/token
    y poner `DAGSHUB_TOKEN`/`DAGSHUB_REPO` en `.env`; al re-ejecutar se suben los runs al link público.
- **OJO artefactos**: el kernel de Jupyter corre con `cwd=notebooks/`; si MLflow crea runs desde el
  notebook, sus artefactos caen en `notebooks/mlruns/` (gitignoreado). El notebook 06 fija
  `MLFLOW_DEFAULT_ARTIFACT_ROOT=<raíz>/mlruns` para que los runs nuevos queden en `mlruns/` de la raíz.
- **Entorno en esta PC**: no hay `.venv` (la otra PC usaba Python 3.12); esta máquina usa el Python
  global **3.14.5** con mlflow 3.15.1, lightgbm 4.7.0, dagshub, nbconvert, etc. **No está instalado
  `catboost`** (el notebook 06 define el config de LightGBM sin importar `configs.py`). Ejecutar:
  `python -m nbconvert --to notebook --execute --inplace notebooks\06_*.ipynb`.

### Sesión 3 (sin PR todavía, cambios en rama `development`)
- **Notebooks 01 y 02 corregidos** para trabajar con el subconjunto de 150 series:
  - `01_preprocesamiento_eda.ipynb`: carga `data/raw/train.csv` completo, filtra a las
    150 series (10 tiendas × 15 items) y todo el EDA/split/guardado usa el subconjunto.
    Re-ejecutado: 15/15 celdas OK → `train.csv` 260,100 filas y `holdout.csv` 13,800
    (92 días × 150 series).
  - `02_feature_engineering.ipynb`: corrige el flujo a **features sobre la serie completa
    ANTES del split** (decisión sesión 2): concatena train+holdout, aplica `build_features`
    y vuelve a dividir. Re-ejecutado: 9/9 celdas OK → `train_features.csv` 255,600 filas
    y `holdout_features.csv` **13,800 filas con 0 NaN** (antes el holdout perdía filas por
    lags NaN). Verificado con pandas.
- **nbconvert instalado en el venv** (faltaba): `pip install nbconvert ipykernel`.
  Ejecutar notebooks desde CLI:
  `& .venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks\XX_*.ipynb`
  (con `workdir` en `notebooks/`, porque usan `Path.cwd().parent`).
- **Fase 3 corrida en `notebooks/03_entrenamiento_mlflow.ipynb`** (11/11 celdas OK):
  experimento `demand_forecast_fase3`, modelo registrado `demand_forecast`.

  | Modelo | MASE | WAPE | rel_bias% | ¿Production? |
  |---|---|---|---|---|
  | naive | 1.038 | 19.23% | +0.31% | no |
  | seasonal_naive | 0.884 | 16.01% | +2.44% | no |
  | mean | 0.918 | 17.33% | **+5.32%** | descartado (sesgo > 5%) |
  | **lightgbm** | **0.581** | **10.41%** | **-0.25%** | **sí (v9)** |

  - Producción: alias `Production` → **lightgbm v9** (la media quedó descartada por
    `rel_bias > 5%`, como esperaba la regla). Consumible vía
    `mlflow.pyfunc.load_model("models:/demand_forecast@Production")`.
  - OJO: al ejecutar el notebook 03 se regeneraron los CSVs de `data/processed/` (mismos
    contenidos que los de la sesión 2). `mlruns/` y `.venv/` están gitignoreados.
  - OJO: MLflow 3.x avisa "has no artifacts at artifact path 'model'..." al registrar;
    es inofensivo (usa `name="model"` en lugar de `artifact_path`).
- **Fase 3 ampliada en `notebooks/04_modelos_adicionales.ipynb`** (6/6 celdas OK):
  nuevas familias con `src/models/configs.py` (catálogo central, reusado por `scripts/train.py`):

  | Modelo | MASE | WAPE | rel_bias% | versión |
  |---|---|---|---|---|
  | xgboost | 0.581 | 10.44% | -0.18% | v10 |
  | catboost | 0.586 | 10.53% | -0.39% | v11 |
  | random_forest | 0.611 | 11.03% | -0.07% | v12 |
  | extra_trees | 0.601 | 10.83% | -0.21% | v13 |
  | mlp | 0.604 | 10.94% | +0.66% | v14 |

  - **Producción se mantiene en lightgbm v9** (MASE=0.581, WAPE=10.41%): XGBoost empata en MASE
    pero pierde por WAPE (ranking combinado MASE + WAPE).
  - `scripts/train.py` refactorizado: `BASE_MODELS`/`all_configs()` ahora viven en
    `src/models/configs.py` (un solo lugar para parámetros de comparación).
- **Fase 3b en `notebooks/05_backtesting_walkforward.ipynb`** (7/7 celdas OK):
  backtesting walk-forward (métricas **online**) con `src/models/backtesting.py`:
  4 folds de ~90 días (dic-2016 .. sep-2017), re-entrenando por fold.

  | Modelo | MASE online | WAPE online | bias online | rel_bias% |
  |---|---|---|---|---|
  | **lightgbm** | **0.604** | **10.18%** | -0.10 | -0.11% |
  | xgboost | 0.605 | 10.20% | -0.11 | -0.13% |
  | seasonal_naive | 0.866 | 14.71% | -0.01 | -0.03% |
  | mean | 0.885 | 15.64% | +0.04 | +0.05% |
  | naive | 1.091 | 18.94% | +0.08 | +0.12% |

  - El orden online coincide con el offline (LightGBM primero, XGBoost casi empatado) →
    la selección no depende de un solo split. Resultados en el experimento
    `demand_forecast_backtest`.

### Sesión 2 (sin PR todavía, cambios en rama `development`)
- **Decisión clave**: reducir la base a **150 series (10 tiendas × 15 artículos)** para
  cumplir el límite de ~100 MB del curso. `data/processed/train_features.csv` pasó de
  98.3 MB a ~31 MB (total de data/processed ≈ 38 MB, raw ≈ 18 MB).
- **Nuevo criterio de decisión**: el **sesgo (bias)** se usa para elegir el mejor modelo,
  no solo como reporte. Regla: ranking combinado **MASE + WAPE**, descartando modelos con
  `|rel_bias_pct| > 5%` y desempatando por el menor sesgo absoluto.
- **MLflow desde la Fase 3**: cada modelo probado se registra como run + Model Registry
  (alias `Production` al mejor según la regla). Ver `scripts/train.py`.
- Nuevos archivos: `scripts/preprocess.py`, `scripts/train.py`, `src/models/metrics.py`,
  `src/models/train_model.py`.

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
| Problema de ML | Supervisado / Regresión. Pronóstico de demanda multi-series. |
| Dataset | **Store Item Demand Forecasting Challenge** (Kaggle `demand-forecasting-kernels-only`): 913k filas, 2013-2017, 10 tiendas × 50 items. **Sesión 2: subconjunto de 150 series (10 tiendas × 15 items)** para cumplir <100 MB. Items: `1, 2, 5, 6, 7, 8, 13, 14, 15, 16, 23, 24, 25, 28, 49` (5 de demanda alta, 5 media, 5 baja). Generado con `scripts/preprocess.py`. |
| Horizonte | 90 días (3 meses). Split temporal: train hasta 2017-09-30, holdout oct–dic 2017. **Mejora sesión 2**: las features se construyen sobre la serie completa ANTES del split, así el holdout conserva los 90 días completos con lags válidos. |
| Métricas | RMSE, MAE, MAPE, sMAPE, MASE, WAPE, MedAE, Max Error, R², **Bias** (`mean(pred-real)`, negativo = subestima) y **rel_bias_pct**. Decisión: ranking **MASE + WAPE**, filtrando `|rel_bias_pct| <= 5%`, desempate por menor sesgo. |
| Modelos | Baselines (naive, seasonal-naive, media), LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees, MLPRegressor (estrategia global). SARIMA/Prophet/Holt-Winters pendientes de decidir con el subconjunto. |
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
| 2. Features | `src/features/build_features.py`: lags (1,7,30), rolling (7,30), calendario (año, mes, día, día de semana, semana), store/item categóricas; notebook `02` | **COMPLETADA** (módulo + notebook ejecutado; `data/processed/*_features.csv`) |
| 2b. Subconjunto | `scripts/preprocess.py`: filtrar a 150 series y reconstruir features sobre la serie completa (holdout 90 días completos) | **COMPLETADA** (sesión 2) |
| 3. Modelos | Backtesting walk-forward, comparar familias; métricas offline y online. Runner con MLflow: `scripts/train.py` + `src/models/train_model.py` + `src/models/metrics.py` | **COMPLETADA** (offline en notebooks 03 y 04 → lightgbm v9; online en notebook 05 → lightgbm; pendiente opcional SARIMA/Prophet con el subconjunto) |
| 4. MLflow | Experimentos (params + metrics + artifacts), DagsHub remoto, registro de modelo productivo `pyfunc` con alias Production | **COMPLETADA** (notebook 06: pyfunc + submission ene-mar 2018; remoto en https://dagshub.com/jaimeramos124/ML2_Series_de_tiempo/experiments → `demand_forecast` v2 Production) |
| 5. Agente genAI | `src/agent/insights_agent.py` con Groq (RAG-lite: contexto + retrieval TF-IDF + LLM) y notebook 07; insights registrados en MLflow | **COMPLETADA** (sesión 5; `demand_forecast_insights` con métricas de latencia/tokens; falta solo `GROQ_API_KEY` para usar el LLM en lugar del heurístico) |
| 6. Scripts | `scripts/preprocess.py`, `scripts/train.py`, `scripts/predict.py` ejecutables por CLI | **COMPLETADA** (sesión 6: `predict.py` probado, 13,500 filas media 52.5 u/día) |
| 7. Documentación | README completo (diagrama Mermaid, Model Card, métricas offline/online, conclusiones), `docs/git_strategy.md` | **COMPLETADA** (sesión 6; `docs/git_strategy.md` ya listo desde fase 0) |
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
- **Sesión 2**: se creó `.venv/` con **Python 3.12** (gitignoreado) e instalado
  `pip install -r requirements.txt` (mlflow 3.15.1, lightgbm 4.7.0, xgboost 3.4.0,
  catboost 1.2.10, prophet 1.3.0, pmdarima 2.1.1). Activar con `.venv\Scripts\python.exe`.
- **Instalados al inicio**: pandas 3.0.3, numpy 2.4.6, matplotlib 3.11.1, seaborn 0.13.2,
  scikit-learn 1.9.0, statsmodels 0.14.6, prophet 1.3.0, openai 2.44.0, joblib 1.5.3.
- **Instalados después (Fase 0/MLflow)**: mlflow 3.15.1, dagshub 0.7.1, pandas 2.3.3
  (mlflow lo degradó desde 3.0.3), websockets 13.1 (necesario para la UI de MLflow),
  kaggle 2.2.4, lightgbm 4.7.0, xgboost 3.4.0, pmdarima 2.1.1.
- **GitHub CLI**: `gh` 2.97.0 instalado y autenticado como JaimeRamosMiranda (los PRs los maneja la IA).
- **MLflow local**: se usa backend **SQLite** (`sqlite:///mlruns/mlflow.db`), porque
  MLflow 3.x puso el filesystem store en modo mantenimiento.
- **Git**: 2.54.0 disponible. **`gh` 2.97.0 SÍ está instalado y autenticado** (crear el repo remoto desde la web ya se hizo).
- **Material original de clase** (notebooks previos + PDF + AirPassengers.csv): se movieron fuera
  del repo a `D:\Jaime Ramos 2\00 Entorno Visual code\MLE2_Clase_Originales\` para que el repo
  tenga únicamente los notebooks del proyecto. Pueden usarse como referencia de código.
- **Clave API**: Groq → `GROQ_API_KEY` en `.env` (gitignoreado). Nunca subir claves al repo.
- **Jupyter/nbconvert**: para ejecutar notebooks desde CLI usar `python -m nbconvert --to notebook --execute --inplace <archivo>` (el subcomando `jupyter-nbconvert` no está registrado).
- **Sesión 3**: se instalaron `nbconvert` + `ipykernel` en el venv (faltaban para ejecutar
  notebooks desde CLI).

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
- [x] README: secciones a (problema de ML) y c (dataset + diccionario) redactadas en detalle (PR #5). Pendientes: b, d, e, f.
- [x] Mover material original de clase fuera del repo (a `MLE2_Clase_Originales/`).
- [x] Obtener `GROQ_API_KEY` (tier gratuito) y guardarla en `.env` (sesión 5: la clave ya
      está configurada; el notebook 07 genera los insights con `llama-3.3-70b-versatile`).
- [ ] Definir si se hacen los retos opcionales (Docker/Azure).
- [x] **Sesión 2**: reducir la base a 150 series (<100 MB) con `scripts/preprocess.py`.
- [x] **Sesión 2**: incorporar **bias / rel_bias_pct** como criterio de selección de modelo.
- [x] **Sesión 2**: MLflow integrado a la Fase 3 (`scripts/train.py`, alias `Production`).
- [x] README: actualizar sección c (dataset) al subconjunto de 150 series (sesión 4: nueva
      subsección 3.2 "Subconjunto usado en el proyecto"; aclarado en secciones 1.2/1.3/1.4/3.5/5).
- [x] **Sesión 4**: Fase 4 **COMPLETADA** — notebook 06 consume el modelo
      productivo como `pyfunc`, predice `data/raw/test.csv` (ene-mar 2018) con **pronóstico
      recursivo** y guarda la submission. **Remoto DagsHub resuelto**: cuenta/repo
      `jaimeramos124/ML2_Series_de_tiempo`, `DAGSHUB_TOKEN` en `.env` (gitignoreado); con
      `dagshub.init(repo, owner, mlflow=True)` se subió el run y se registró
      `demand_forecast` **v2 → alias `Production`**. Evidencia:
      https://dagshub.com/jaimeramos124/ML2_Series_de_tiempo/experiments
- [x] **Sesión 5**: Fase 5 **COMPLETADA** — agente de insights RAG-lite con Groq
      (`src/agent/insights_agent.py`, notebook 07, experimento `demand_forecast_insights`).
      Modelo de producción exportado a `models/demand_forecast/` para commitear a GitHub
      (decisión del usuario: el modelo debe subirse al repo, no solo a DagsHub).

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
- DagsHub gratuito: `import dagshub; dagshub.init(repo_name, repo_owner, mlflow=True)`
  configura el tracking remoto (OJO dagshub 0.7.x: el **primer** argumento es el repo).
  Link de experimentos: `https://dagshub.com/<user>/<repo>/experiments`.
- Cargar modelo productivo: `mlflow.pyfunc.load_model("models:/<nombre>@Production")`.
- **OJO MLflow 3.x + skops**: `mlflow.sklearn.log_model` falla para clases no estándar
  (LightGBM, baselines custom) con "references untrusted types". `src/models/train_model.py`
  lo resuelve automáticamente: detecta el error, extrae los tipos y reintenta con
  `skops_trusted_types`. Para el alias `Production`, se espera a que la versión esté
  `READY` antes de asignarlo.
- El alias es **único por modelo**: asignarlo a una versión lo quita de la anterior.

---

## 9. Pasos siguientes (próxima sesión)

> **Estado al cierre de la sesión 6**: Fases 0-7 completadas. Falta la **Fase 8 (release)**.
> La evidencia de DagsHub quedó **re-subida** (`scripts/upload_dagshub_evidence.py`).
> Pendiente: limpiar PROYECTO.md (decisión del usuario: solo contexto interno).

1. **Fase 8 — release**: commitear `scripts/upload_dagshub_evidence.py`, PR final
   development→main, tag v1.0.0 con notas de release (último commit 30/8). OJO: limpiar
   PROYECTO.md antes de la entrega y confirmar que el README quedó bien en GitHub (render
   de Mermaid y tablas) y que el link de DagsHub muestra el run `lightgbm_production`
   con su modelo y Production.

---

## 10. Cómo continuar desde otra computadora

1. **Clonar** el repo:
   `git clone https://github.com/JaimeRamosMiranda/ML2_Series_de_tiempo.git`
   y entrar a `ML2_Series_de_tiempo/`.
2. **Crear/activar entorno** (usar Python 3.12, ya validado):
   `py -3.12 -m venv .venv` y luego `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
   (si algo falla por Python 3.14, usar Python 3.11/3.12).
3. **Configurar credenciales** (una vez por máquina):
   - GitHub: `gh auth login` (los PRs los maneja la IA vía `gh`).
   - Kaggle (solo si hay que re-descargar el dataset): `kaggle auth login`
     — el dataset ya está commiteado en `data/raw/`.
   - Groq: copiar `.env.example` a `.env` y pegar `GROQ_API_KEY`
     (tier gratuito: https://console.groq.com).
   - DagsHub (Fase 4, sesión 4): `DAGSHUB_TOKEN` en `.env` y `dagshub.init(...)`.
4. **Recordar**:
   - Trabajar en la rama `development` (nunca commitear directo a `main`).
   - MLflow local usa SQLite: `sqlite:///mlruns/mlflow.db` (ver `scripts/demo_mlflow.py`).
   - Leer este archivo para retomar el contexto y el `README.md` para el entregable.
5. **Verificar estado**: `git status`, `git log --oneline -5`, `git pull --ff-only origin development`.

> El contexto completo de lo conversado está en este archivo; el entregable formal
> (README por secciones) está en `README.md`.
