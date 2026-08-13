# Pronóstico de demanda multi-series

Proyecto final — Curso II, Especialización en Machine Learning Engineering.

Pronóstico de demanda para **series de tiempo** (ventas diarias por tienda-artículo)
usando **MLflow** como herramienta de administración de experimentos y modelos, y un
**agente genAI de insights** (RAG-lite con Groq) que explica los pronósticos.

> El dataset original tiene **500 series** (10 tiendas × 50 artículos); por el límite de
> ~100 MB del curso, este proyecto trabaja con un **subconjunto de 150 series**
> (10 tiendas × 15 artículos). Ver sección 3.

> Documento de contexto general del proyecto: [`PROYECTO.md`](PROYECTO.md).

---

## 1. Problema de Machine Learning (a)

### 1.1 Contexto de negocio

Una cadena minorista opera **10 tiendas** y comercializa **50 artículos** distintos, lo
que da lugar a **500 combinaciones tienda-artículo**, cada una con su propio patrón de
demanda. El negocio registra diariamente las unidades vendidas de cada artículo en cada
tienda desde enero de 2013 hasta diciembre de 2017.

El área de operaciones/inventarios necesita decidir, con tres meses de anticipación,
cuánto stock mantener de cada artículo en cada tienda. Estas decisiones afectan:

- **Roturas de stock** (stockout): perder ventas por no tener mercancía y deteriorar la
  experiencia del cliente.
- **Exceso de inventario**: capital inmovilizado, costo de almacenamiento y riesgo de
  merma/caducidad.
- **Planificación con proveedores**: órdenes de compra y logística con antelación.

Un pronóstico preciso de la demanda a 90 días permite **equilibrar el nivel de servicio
y el costo de inventario**, y alimenta decisiones de reposición, stock de seguridad,
asignación de espacio y presupuesto por tienda.

### 1.2 Definición formal del problema

- **Tipo**: Machine Learning **supervisado** → **regresión** sobre series de tiempo múltiples.
- **Objetivo**: predecir las **unidades vendidas** (`sales`) de cada serie para los
  **próximos 90 días**.
- **Series**: el dataset original tiene 500 series diarias independientes, identificadas
  por `(store, item)`. **Este proyecto usa un subconjunto de 150 series**
  (10 tiendas × 15 artículos, ver sección 3) elegido para cumplir el límite de tamaño
  del curso.
- **Horizonte**: 90 días (3 meses) por serie.

Formalmente: dado el historial de ventas de cada serie
`Y_s = {y_{s,t-1}, ..., y_{s,t-T}}` para `s = 1..150`, se quiere estimar
`y_{s,t+1}, ..., y_{s,t+90}`.

### 1.3 Enfoque de modelado

Se sigue un enfoque de **backtesting walk-forward** sobre un split temporal (sin fuga de
datos), comparando:

| Familia | Modelos |
|---|---|
| Baselines | naive, seasonal-naive, media por serie |
| Estadísticos | SARIMA (`auto_arima`), Prophet, Holt-Winters (submuestra) |
| Supervisados (GBM) | LightGBM, XGBoost, CatBoost |
| Supervisados (otros) | RandomForest, ExtraTrees, MLPRegressor |

Los modelos supervisados usan un **modelo global** (un solo modelo entrena sobre las
features de las **150 series del subconjunto**) con features de la Fase 2: calendario,
lags (1, 7, 30) y estadísticas móviles (media/desviación de 7 y 30 días), con `store` e
`item` como categóricas.

### 1.4 Métricas de evaluación y criterio de éxito

| Métrica | Interpretación |
|---|---|
| RMSE / MAE | Error absoluto en unidades de venta |
| MAPE / sMAPE | Error porcentual, comparable entre series |
| **MASE** | Error escalado contra el baseline naive; **< 1 significa que supera al naive** |
| **WAPE** | Error absoluto ponderado agregado (visión de negocio/inventario) |

- **Criterio de decisión**: ranking combinado de **MASE + WAPE**, descartando modelos con
  **sesgo relativo** `|rel_bias_pct| > 5%` (sub/sobreestimación sistemática) y desempatando
  por el menor sesgo absoluto.
- Se reportan métricas **offline** (holdout oct–dic 2017) y **online** (backtest por
  ventanas deslizantes que simula producción).

---

## 2. Diagrama de flujo del proyecto (b)

```mermaid
flowchart TD
    A[Kaggle: train.csv 2013-2017<br/>913k filas, 500 series] --> B[Subconjunto 150 series<br/>10 tiendas x 15 items]
    B --> C[Fase 1: EDA + split temporal<br/>train <= 2017-09-30 | holdout oct-dic 2017]
    C --> D[Fase 2: Feature engineering<br/>calendario + lags 1,7,30 + rolling 7,30]
    D --> E[Fase 3: Modelos + MLflow<br/>baselines, GBM, ensembles, MLP + backtest walk-forward]
    E --> F[Seleccion del modelo productivo<br/>regla MASE + WAPE, filtro de sesgo <= 5%]
    F --> G[Fase 4: pyfunc + submission<br/>ene-mar 2018, evidencias en DagsHub]
    F --> H[Fase 5: Agente de insights<br/>RAG-lite con Groq, experimentos en MLflow]
    G --> I[Fase 6: Scripts CLI<br/>preprocess, train, predict]
    H --> I
    I --> J[Release v1.0.0]
```

**Evidencia de experimentos (MLflow en DagsHub)**:
<https://dagshub.com/jaimeramos124/ML2_Series_de_tiempo/experiments>

---

## 3. Descripción del dataset y diccionario de datos (c)

### 3.1 Descripción general

El dataset corresponde a la competición de Kaggle
**"Store Item Demand Forecasting Challenge"** (`demand-forecasting-kernels-only`).

Contiene **913,000 registros** de ventas diarias de **10 tiendas × 50 artículos**
(500 series) durante **5 años** (2013-01-01 a 2017-12-31). Es un dataset tabular limpio:
**sin valores nulos, sin duplicados y sin ventas negativas**, con series diarias completas
(sin fechas faltantes).

### 3.2 Subconjunto usado en el proyecto

Por el límite de **~100 MB del curso**, el proyecto trabaja con un **subconjunto de
150 series = 10 tiendas × 15 artículos** (mezcla de demanda alta, media y baja). Todos
los pasos (EDA, features, entrenamiento, evaluación y predicción) usan solo este
subconjunto.

- **Tiendas**: 1–10 (las 10).
- **Artículos (15)**: `1, 2, 5, 6, 7, 8, 13, 14, 15, 16, 23, 24, 25, 28, 49`.
- **Historial completo del subconjunto**: 273,900 filas (150 series × 1,826 días).
- **Split temporal**: `train` → 260,100 filas (hasta 2017-09-30); `holdout` → 13,800
  filas (oct–dic 2017). Con features: `train_features.csv` 255,600 y
  `holdout_features.csv` 13,800 filas.

> Las métricas de las secciones 5 (offline/online) y la submission de la Fase 4 se
> refieren **exclusivamente a este subconjunto**, no a las 500 series originales.

### 3.3 Diccionario de datos

| Columna | Tipo | Descripción |
|---|---|---|
| `date` | `datetime` | Día de la venta. Frecuencia diaria (2013-01-01 a 2017-12-31). No hay efectos de feriados ni cierres de tienda |
| `store` | `int64` | Identificador de la tienda (1–10) |
| `item` | `int64` | Identificador del artículo (1–50) |
| `sales` | `int64` | **Target**. Unidades vendidas del artículo en la tienda durante esa fecha (entero, >= 0) |

### 3.4 Volumen y estructura

| Atributo | Valor |
|---|---|
| Registros | 913,000 |
| Series (store × item) | 500 |
| Frecuencia | Diaria |
| Rango temporal | 2013-01-01 a 2017-12-31 |
| Valores nulos | 0 |
| Ventas negativas | 0 |
| Tamaño | ~17 MB (train.csv) |

### 3.5 Archivos adicionales

| Archivo | Contenido |
|---|---|
| `test.csv` | 2018-01-01 a 2018-03-31 (próximos 3 meses, 45,000 filas para las 500 series) sin columna `sales`. La Fase 4 predice solo las **150 series del subconjunto** (13,500 filas) y guarda `reports/submissions/submission_production.csv` |
| `sample_submission.csv` | Formato de entrega de la competición (45,000 filas). La submission del proyecto cubre solo el subconjunto: es **demostrativa**, no apta para el leaderboard |

### 3.6 Preprocesamiento y split

- Se separaron las features del target y se construyó el dataset supervisado en
  `src/features/build_features.py` (ver Fase 2).
- **Split temporal** (sin fuga de datos, por fecha y no aleatorio):
  - `train` → hasta 2017-09-30
  - `holdout` → 2017-10-01 a 2017-12-31 (simula el pronóstico de 3 meses)

---

## 4. Model Card (d)

### Resumen

Modelo global de regresión **LightGBM** que pronostica la demanda diaria (unidades
vendidas) de las **150 series del subconjunto** para un horizonte de **90 días**. El
modelo vive en el Model Registry de MLflow con el alias **`Production`** y se consume
como envoltura **pyfunc**.

### Uso previsto

- Pronóstico de demanda a 90 días por tienda-artículo para la planificación de inventario
  (stock de seguridad, reposición y presupuesto por tienda).
- Generación de la submission demostrativa ene-mar 2018 y de insights de negocio por
  serie (agente genAI de la Fase 5).

### Arquitectura y entrenamiento

- **Algoritmo**: LightGBM (Gradient Boosting) — `n_estimators=300`, `learning_rate=0.05`,
  `num_leaves=63`, `random_state=42`.
- **Estrategia**: modelo **global** único entrenado sobre las features de las 150 series.
- **Features (15)**: calendario (`year`, `month`, `day`, `dayofweek`, `weekofyear`,
  `dayofyear`), `store`/`item` y lags (1, 7, 30) + medias/desviaciones móviles (7, 30 días).
- **Datos de entrenamiento**: `train_features.csv` (255,600 filas, hasta 2017-09-30).
- **Datos de evaluación**: `holdout_features.csv` (13,800 filas, oct–dic 2017).

### Métricas de evaluación

| Evaluación | MASE | WAPE | rel_bias |
|---|---|---|---|
| Offline (holdout oct–dic 2017) | **0.581** | **10.41%** | **-0.25%** |
| Online (backtest walk-forward) | **0.604** | **10.18%** | **-0.11%** |

`MASE < 1` → el modelo supera al baseline naive en ~40%. El sesgo relativo bajo indica
que no subestima ni sobreestima de forma sistemática.

### Criterio de selección

Ranking combinado **MASE + WAPE** con filtro `|rel_bias_pct| <= 5%`; desempate por el
menor sesgo absoluto. LightGBM ganó en la comparación offline (notebooks 03-04) y se
confirmó en la online (notebook 05), por lo que se promovió a `Production`.

### Limitaciones

- Cubre solo el **subconjunto de 150 series** (el dataset original tiene 500), así que la
  submission no es apta para el leaderboard de la competición.
- Las features son **autoregresivas** (`lag_1`, rolling): el horizonte de 90 días requiere
  **pronóstico recursivo** día a día, lo que acumula error cuanto más lejos del último dato
  real.
- El dataset no incluye feriados ni eventos especiales; el modelo no los modela.

---

## 5. Resultados con métricas offline y online (e)

Todas las métricas corresponden al **subconjunto de 150 series**.

- **Offline**: entrenamiento con datos hasta 2017-09-30 y evaluación sobre el holdout
  **oct–dic 2017** (3 meses, 13,800 filas). Ver notebooks 03 y 04.
- **Online**: backtesting **walk-forward** sobre el historial con 4 ventanas de ~90 días
  (dic-2016 .. sep-2017), re-entrenando el modelo en cada fold para simular producción.
  Ver notebook 05.

### 5.1 Offline (holdout oct–dic 2017)

| Modelo | MASE | WAPE | rel_bias |
|---|---|---|---|
| naive | 1.038 | 19.23% | +0.31% |
| seasonal_naive | 0.884 | 16.01% | +2.44% |
| media por serie | 0.918 | 17.33% | **+5.32%** *(descartado, sesgo > 5%)* |
| **lightgbm** | **0.581** | **10.41%** | **-0.25%** ✓ *Production* |
| xgboost | 0.581 | 10.44% | -0.18% |
| catboost | 0.586 | 10.53% | -0.39% |
| random_forest | 0.611 | 11.03% | -0.07% |
| extra_trees | 0.601 | 10.83% | -0.21% |
| mlp | 0.604 | 10.94% | +0.66% |

### 5.2 Online (backtest walk-forward)

| Modelo | MASE online | WAPE online | rel_bias online |
|---|---|---|---|
| **lightgbm** | **0.604** | **10.18%** | **-0.11%** |
| xgboost | 0.605 | 10.20% | -0.13% |
| seasonal_naive | 0.866 | 14.71% | -0.03% |
| media por serie | 0.885 | 15.64% | +0.05% |
| naive | 1.091 | 18.94% | +0.12% |

**LightGBM gana en ambas evaluaciones**; XGBoost empata casi exacto en MASE pero pierde
por WAPE (ranking combinado). El orden de los modelos se mantiene entre offline y online,
lo que confirma que la selección no depende de un solo split.

### 5.3 Componente genAI (Fase 5)

El agente RAG-lite (contexto + retrieval TF-IDF + LLM Groq) genera un insight de negocio
por serie. Cada insight se registra en el experimento `demand_forecast_insights` con
métricas coherentes del componente: **latencia** total/retrieval/generación, **tokens**
usados y **cobertura de la serie** (si el texto menciona tienda y artículo). Con
`GROQ_API_KEY` configurada los insights los genera `llama-3.3-70b-versatile`.

---

## 6. Conclusiones (f)

- El enfoque **supervisado global** (LightGBM) supera a todos los baselines: MASE **0.581**
  offline y **0.604** online, es decir, un error ~40% menor que el naive (< 1).
- El orden de los modelos se mantiene entre la evaluación **offline** y la **online**, lo
  que indica que la selección no depende de un solo split temporal.
- Incorporar el **sesgo** al criterio de decisión evitó elegir modelos que sub/sobreestiman
  sistemáticamente: la media por serie quedó descartada por `|rel_bias| > 5%`.
- El modelo productivo se consume como **pyfunc** (`models:/demand_forecast@Production`) y
  genera la submission de 90 días con **pronóstico recursivo** (media 52.5 u/día, escala
  coherente con el histórico de 56.2 u/día).
- **MLflow + DagsHub** dan evidencia pública de experimentos, artefactos y modelo
  productivo; el **agente genAI** (RAG-lite con Groq) explica los pronósticos en lenguaje
  natural, cubriendo la parte genAI del entregable.

---

## Estado del proyecto

- Fase 0 (setup del repositorio + MLflow demo): completada.
- Fase 1 (datos y EDA): completada — `notebooks/01_preprocesamiento_eda.ipynb`, datos en `data/`.
- Fase 2 (feature engineering): completada — `src/features/build_features.py`, `notebooks/02_feature_engineering.ipynb`.
- Fase 3 (modelos + MLflow + backtest): completada — notebooks 03-05; experimentos y Model
  Registry en MLflow; LightGBM en `Production`.
- Fase 4 (pyfunc + submission + DagsHub): completada — `notebooks/06_mlflow_dagshub_pyfunc.ipynb`,
  submission en `reports/submissions/submission_production.csv`, evidencias en DagsHub.
- Fase 5 (agente genAI de insights): completada — `src/agent/insights_agent.py`,
  `notebooks/07_agente_insights.ipynb`.
- Fase 6 (scripts CLI): completada — `scripts/preprocess.py`, `scripts/train.py`,
  `scripts/predict.py`.

## Estructura del repositorio

```
├── data/            # datos crudos (raw) y procesados
├── docs/            # documentación (estrategia git, etc.)
├── models/          # modelo de producción exportado (pyfunc) para el entregable
├── notebooks/       # notebooks del proyecto (EDA, ML y agente genAI)
├── scripts/         # preprocesamiento, entrenamiento y predicción
├── src/             # módulo de código reusable
│   ├── agent/       # agente genAI de insights (RAG-lite con Groq)
│   ├── data/        # carga y preprocesamiento
│   ├── features/    # construcción de features (lags, calendario)
│   └── models/      # entrenamiento, predicción y evaluación
└── PROYECTO.md      # contexto completo del proyecto
```
