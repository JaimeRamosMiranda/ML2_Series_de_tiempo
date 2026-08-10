# PROYECTO — Contexto general

> Archivo de contexto del proyecto. Su propósito es permitir continuar el trabajo
> desde otra computadora/sesión sin perder el contexto de lo conversado.

Última actualización: **2026-08-09** (sesión 3: notebooks 01 y 02 corregidos para el subconjunto de 150 series)

### Registro de PRs (todas cerradas exitosamente)
| PR | Contenido |
|---|---|
| #1 | Fase 0: setup del repositorio + demo de MLflow |
| #2 | Fase 1: dataset y notebook de EDA |
| #3 | Fase 2: feature engineering |
| #4 | Separar material original de clase del repo |
| #5 | README secciones a (problema de ML) y c (dataset/diccionario) |

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
| 4. MLflow | Experimentos (params + metrics + artifacts), DagsHub remoto, registro de modelo productivo `pyfunc` con alias Production | **EN CURSO** (local + SQLite funcional; alias Production automático con filtro de sesgo) |
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
- [ ] Obtener `GROQ_API_KEY` (tier gratuito) y guardarla en `.env`.
- [ ] Definir si se hacen los retos opcionales (Docker/Azure).
- [x] **Sesión 2**: reducir la base a 150 series (<100 MB) con `scripts/preprocess.py`.
- [x] **Sesión 2**: incorporar **bias / rel_bias_pct** como criterio de selección de modelo.
- [x] **Sesión 2**: MLflow integrado a la Fase 3 (`scripts/train.py`, alias `Production`).
- [ ] README: actualizar sección c (dataset) al subconjunto de 150 series.
- [ ] **Sesión 4 (pendiente)**: Fase 4 en dos partes — (a) modelo productivo `pyfunc` local
      (notebook 06) y (b) MLflow remoto en **DagsHub**: crear cuenta/repo, obtener `DAGSHUB_TOKEN`
      (o usar el usuario), `dagshub.init(repo_owner, repo_name, mlflow=True)` y registrar runs
      remotos para tener el link público de evidencia. Ver sección 9.

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
- **OJO MLflow 3.x + skops**: `mlflow.sklearn.log_model` falla para clases no estándar
  (LightGBM, baselines custom) con "references untrusted types". `src/models/train_model.py`
  lo resuelve automáticamente: detecta el error, extrae los tipos y reintenta con
  `skops_trusted_types`. Para el alias `Production`, se espera a que la versión esté
  `READY` antes de asignarlo.
- El alias es **único por modelo**: asignarlo a una versión lo quita de la anterior.

---

## 9. Pasos siguientes (próxima sesión)

> **Estado al cierre de la sesión 3**: Fases 0-3 completadas. Notebooks 01-05 ejecutados
> y commiteados. `Production` local = **lightgbm v9** (offline y online).

1. **Fase 4 — notebook `06_mlflow_dagshub_pyfunc.ipynb`**:
   - Parte local: cargar `models:/demand_forecast@Production`, predecir `data/raw/test.csv`
     (ene-mar 2018), guardar submission y envolver el modelo como `pyfunc` (ya está logueado
     con `mlflow.sklearn.log_model`; la envoltura se consume con `mlflow.pyfunc.load_model`).
   - Parte remota (DagsHub, requiere credenciales del estudiante):
     1. Crear cuenta en https://dagshub.com (usuario: same as GitHub, JaimeRamosMiranda).
     2. Crear repositorio (p.ej. `ML2_Series_de_tiempo`).
     3. Token: Settings → User Settings → Tokens → `DAGSHUB_TOKEN`.
     4. Guardar `DAGSHUB_TOKEN` en `.env` (gitignoreado).
     5. En el notebook: `import dagshub; dagshub.init("JaimeRamosMiranda", "<repo>", mlflow=True)`
        y registrar los runs/experimentos remotos.
     6. Link de evidencia: `https://dagshub.com/<user>/<repo>/experiments`.
   - OJO: si el token va por `dagshub.init(..., mlflow=True)`, MLflow apunta a DagsHub;
     volver a `mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")` para lo local.
2. **Fase 5 — notebook `07_agente_insights.ipynb`**: agente genAI (RAG-lite) con Groq:
   `GROQ_API_KEY` en `.env` (tier gratuito: https://console.groq.com), modelo
   `llama-3.3-70b-versatile`; 3 pasos: contexto (stats + pronóstico), retrieval TF-IDF sobre
   fichas de serie, generación de insights. `src/agent/insights_agent.py` (pendiente de crear).
3. **Fase 6-8**: `scripts/predict.py` (CLI, submission con test.csv), actualizar README
   (subconjunto 150 series, diagrama Mermaid, Model Card, métricas offline/online con los
   resultados de los notebooks 03-05, conclusiones), `docs/git_strategy.md`, PR final
   development→main y release v1.0.0 (último commit 30/8).

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
