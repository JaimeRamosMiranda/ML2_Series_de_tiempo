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

- **Criterio de decisión**: ranking combinado de **MASE + WAPE**; desempate con sMAPE y
  estabilidad entre las series del subconjunto (menor dispersión del error).
- Se reportan métricas **offline** (holdout oct–dic 2017) y **online** (backtest por
  ventanas deslizantes que simula producción).

---

## 2. Diagrama de flujo del proyecto (b)

*Pendiente de completar (se documentará con diagrama Mermaid).*

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

*Pendiente de completar.*

---

## 5. Resultados con métricas offline y online (e)

*Pendiente de completar (se llenará con los resultados de las Fases 3 y 4). Las métricas
corresponden al subconjunto de 150 series: offline sobre el holdout oct–dic 2017 y
online con backtesting walk-forward.*

---

## 6. Conclusiones (f)

*Pendiente de completar.*

---

## Estado del proyecto

- Fase 0 (setup del repositorio + MLflow demo): completada.
- Fase 1 (datos y EDA): completada — `notebooks/01_preprocesamiento_eda.ipynb`, datos en `data/`.
- Fase 2 (feature engineering): completada — `src/features/build_features.py`, `notebooks/02_feature_engineering.ipynb`.

## Estructura del repositorio

```
├── data/            # datos crudos (raw) y procesados
├── docs/            # documentación (estrategia git, etc.)
├── notebooks/       # notebooks del proyecto (EDA y ML)
├── scripts/         # preprocesamiento, entrenamiento y predicción
├── src/             # módulo de código reusable
│   ├── data/        # carga y preprocesamiento
│   ├── features/    # construcción de features (lags, calendario)
│   ├── models/      # entrenamiento, predicción y evaluación
│   └── agent/       # agente genAI de insights (Groq)
└── PROYECTO.md      # contexto completo del proyecto
```
