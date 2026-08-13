"""Agente de insights (RAG-lite) para el pronóstico de demanda multi-series.

Fase 5. El agente genera un texto de negocio para una serie concreta en tres
pasos:

1. **Contexto**: estadísticas de la serie (histórico y pronóstico ene-mar 2018)
   a partir de los datos; se resume en una ficha textual.
2. **Retrieval**: se buscan las series del dataset con perfil histórico más
   parecido, comparando las fichas con similitud de coseno sobre TF-IDF.
3. **Generación**: con el contexto y las series similares como evidencia, un LLM
   de Groq (API compatible con OpenAI) redacta el insight. Si no hay clave de
   API configurada, se usa un generador heurístico determinista para que el
   flujo completo funcione sin red.

Cada insight puede registrarse en MLflow con ``log_insight_run`` (latencia,
tokens y cobertura de la serie), para tener métricas coherentes del componente
genAI en el mismo lugar que el resto de los experimentos.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MONTHS = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]
WEEKDAYS = [
    "lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo",
]

DEFAULT_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

SPANISH_STOP_WORDS = [
    "de", "del", "la", "el", "los", "las", "un", "una", "uno", "unas",
    "con", "en", "para", "por", "y", "que", "se", "al", "su", "sus",
    "serie", "tienda", "articulo", "dia", "unidades", "media", "periodo",
    "historico", "promedio",
]


# ---------------------------------------------------------------------------
# Estadísticas y fichas de serie
# ---------------------------------------------------------------------------
def _month_name(m: int) -> str:
    return MONTHS[int(m) - 1]


def build_series_stats(history: pd.DataFrame) -> dict:
    """Estadísticas de una serie a partir de su historial de ventas.

    Args:
        history: DataFrame con columnas ``date`` (datetime) y ``sales``.

    Returns:
        Dict con media, desvío, rango, perfil semanal, perfil mensual y
        tendencia de los últimos 90 días respecto al periodo previo.
    """
    s = history.sort_values("date").reset_index(drop=True)
    sales = s["sales"].astype(float)

    recent90 = sales.tail(90).mean()
    prev90 = sales.iloc[-180:-90].mean() if len(sales) >= 180 else sales.mean()
    trend_90 = (recent90 - prev90) / prev90 * 100 if prev90 else 0.0

    weekday_mean = s.groupby(s["date"].dt.dayofweek)["sales"].mean()
    month_mean = s.groupby(s["date"].dt.month)["sales"].mean()

    return {
        "mean": float(sales.mean()),
        "std": float(sales.std()),
        "min": float(sales.min()),
        "max": float(sales.max()),
        "last_7_mean": float(sales.tail(7).mean()),
        "trend_90_pct": float(trend_90),
        "weekday_mean": {int(d): float(v) for d, v in weekday_mean.items()},
        "month_mean": {int(m): float(v) for m, v in month_mean.items()},
        "peak_month": int(month_mean.idxmax()),
        "low_month": int(month_mean.idxmin()),
        "peak_weekday": int(weekday_mean.idxmax()),
    }


def build_forecast_stats(forecast: pd.DataFrame) -> dict:
    """Estadísticas del pronóstico de una serie para el periodo futuro.

    Args:
        forecast: DataFrame con columnas ``date`` (datetime) y ``sales``
            (predicciones del modelo de producción).

    Returns:
        Dict con total, media, rango, total por mes, mes pico y tendencia
        dentro del trimestre pronosticado.
    """
    f = forecast.sort_values("date").reset_index(drop=True)
    sales = f["sales"].astype(float)

    head = sales.head(30).mean()
    tail = sales.tail(30).mean()
    trend_in = (tail - head) / head * 100 if head else 0.0

    month_total = f.groupby(f["date"].dt.month)["sales"].sum()

    return {
        "total": float(sales.sum()),
        "mean": float(sales.mean()),
        "min": float(sales.min()),
        "max": float(sales.max()),
        "month_total": {int(m): float(v) for m, v in month_total.items()},
        "peak_month": int(month_total.idxmax()),
        "trend_in_pct": float(trend_in),
    }


def describe_series(store: int, item: int, stats: dict, fc: dict | None = None) -> str:
    """Ficha textual de una serie (histórico y, si se da, pronóstico)."""
    wd = stats["weekday_mean"]
    wd_note = f"El día más fuerte de la semana es el {WEEKDAYS[stats['peak_weekday']]}"

    text = (
        f"Tienda {store}, artículo {item}: media diaria de {stats['mean']:.1f} "
        f"unidades (desvío {stats['std']:.1f}, rango {stats['min']:.0f}-{stats['max']:.0f}). "
        f"Tendencia de los últimos 90 días: {stats['trend_90_pct']:+.1f}% respecto "
        f"al periodo previo. Mes de mayor demanda histórica: {_month_name(stats['peak_month'])}; "
        f"el de menor: {_month_name(stats['low_month'])}. {wd_note}."
    )
    if fc is not None:
        text += (
            f" Pronóstico ene-mar 2018: media de {fc['mean']:.1f} unidades/día "
            f"({fc['mean'] - stats['mean']:+.1f} vs histórico), pico esperado en "
            f"{_month_name(fc['peak_month'])}."
        )
    return text


# ---------------------------------------------------------------------------
# Corpus de series + retrieval por TF-IDF
# ---------------------------------------------------------------------------
@dataclass
class SeriesCorpus:
    """Índice de fichas de series para búsqueda por similitud TF-IDF.

    Se construye una vez con las 150 series del subconjunto; la búsqueda
    devuelve las series con perfil histórico más parecido a la consulta.
    """

    cards: list[dict]
    vectorizer: TfidfVectorizer = field(init=False)
    matrix: np.ndarray = field(init=False)

    def __init__(self, cards: list[dict]):
        """``cards`` es una lista de dicts ``{"store", "item", "text"}``."""
        self.cards = cards
        texts = [c["text"] for c in cards]
        self.vectorizer = TfidfVectorizer(stop_words=SPANISH_STOP_WORDS)
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Top-k series más parecidas a ``query`` por similitud de coseno."""
        qv = self.vectorizer.transform([query])
        sim = cosine_similarity(qv, self.matrix).ravel()
        order = sim.argsort()[::-1][:k]
        return [
            {
                "store": self.cards[i]["store"],
                "item": self.cards[i]["item"],
                "similarity": float(sim[i]),
            }
            for i in order
        ]


# ---------------------------------------------------------------------------
# Generación
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Eres un analista de demanda con experiencia en retail. A partir de la ficha "
    "de una serie y de series históricas similares como referencia, redactas un "
    "insight de negocio breve, concreto y sin relleno."
)


def _build_user_prompt(context: dict, retrieved: list[dict]) -> str:
    lines = [
        "Analiza la serie objetivo (histórico y pronóstico ene-mar 2018):",
        context["card"],
        "",
        "Series del mismo dataset con perfil histórico similar (referencia):",
    ]
    for r in retrieved:
        lines.append(
            f"- tienda {r['store']}, artículo {r['item']} "
            f"(similitud {r['similarity']:.2f}): {r['text']}"
        )
    lines += [
        "",
        "Escribe un insight de negocio en español, máximo 150 palabras, que:",
        "- describa el comportamiento esperado ene-mar 2018 y su magnitud relativa al histórico,",
        "- señale la posible causa de la tendencia,",
        "- cierre con una recomendación accionable (inventario, promociones, abastecimiento).",
        "Menciona explícitamente la tienda y el artículo de la serie objetivo.",
    ]
    return "\n".join(lines)


def rule_based_insight(context: dict) -> str:
    """Insight determinista (baseline heurístico) sin llamar a un LLM.

    Se usa como referencia de comparación y como respaldo cuando no hay clave
    de API. Construye el texto a partir de las estadísticas del contexto.
    """
    store, item = context["store"], context["item"]
    h, f = context["history"], context["forecast"]

    diff = f["mean"] - h["mean"]
    rel = diff / h["mean"] * 100 if h["mean"] else 0.0

    lines = [
        f"La serie tienda {store}, artículo {item} presenta un pronóstico para "
        f"ene-mar 2018 de {f['mean']:.1f} unidades/día, un {rel:+.0f}% respecto a la "
        f"media histórica de {h['mean']:.1f} unidades/día."
    ]
    if h["trend_90_pct"] > 5:
        lines.append(
            "El histórico acumula una tendencia al alza en los últimos 90 días, "
            "consistente con un periodo de demanda creciente."
        )
    elif h["trend_90_pct"] < -5:
        lines.append(
            "El histórico muestra una tendencia a la baja en los últimos 90 días, "
            "por lo que conviene vigilar si el descenso se consolida."
        )
    else:
        lines.append("El nivel de demanda del histórico se mantiene estable.")
    lines.append(
        f"El pico del trimestre se espera en {_month_name(f['peak_month'])} "
        f"({f['month_total'][f['peak_month']]:.0f} unidades en el mes)."
    )

    if rel > 3:
        lines.append(
            "Recomendación: asegurar inventario adicional hacia el pico del "
            "trimestre para evitar roturas de stock."
        )
    elif rel < -3:
        lines.append(
            "Recomendación: ajustar los pedidos a la baja reduce el exceso de "
            "inventario sin afectar la disponibilidad."
        )
    else:
        lines.append(
            "Recomendación: mantener el plan de abastecimiento habitual, con "
            "seguimiento semanal del pronóstico."
        )
    return "\n".join(lines)


class InsightsAgent:
    """Agente RAG-lite: contexto -> retrieval -> generación.

    Args:
        model: modelo de Groq (compatible con OpenAI).
        k: número de series similares a recuperar.
        temperature: temperatura de muestreo del LLM.
        max_tokens: tope de tokens generados.
        api_key: clave de Groq; si es None se usa el generador heurístico.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        k: int = 3,
        temperature: float = 0.2,
        max_tokens: int = 400,
        api_key: str | None = None,
    ):
        self.model = model
        self.k = k
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        if api_key:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    @property
    def llm_available(self) -> bool:
        """True si hay clave de API y se usará el LLM de Groq."""
        return self._client is not None

    def generate(
        self,
        store: int,
        item: int,
        history: pd.DataFrame,
        forecast: pd.DataFrame,
        corpus: SeriesCorpus | None = None,
    ) -> dict:
        """Genera el insight para una serie y registra las latencias.

        Args:
            store: número de tienda.
            item: número de artículo.
            history: historial de la serie (``date``, ``sales``).
            forecast: pronóstico de la serie (``date``, ``sales``).
            corpus: índice de fichas para el retrieval (opcional).

        Returns:
            Dict con contexto, series recuperadas, respuesta, generador usado,
            uso de tokens y latencias en ms.
        """
        t0 = time.perf_counter()
        h_stats = build_series_stats(history)
        f_stats = build_forecast_stats(forecast)
        card = describe_series(store, item, h_stats, f_stats)
        context = {
            "store": int(store),
            "item": int(item),
            "card": card,
            "history": h_stats,
            "forecast": f_stats,
        }

        retrieved = []
        if corpus is not None:
            query_card = describe_series(store, item, h_stats, None)
            for r in corpus.search(query_card, k=self.k + 1):
                if r["store"] == int(store) and r["item"] == int(item):
                    continue
                match = next(
                    (c for c in corpus.cards if c["store"] == r["store"] and c["item"] == r["item"]),
                    None,
                )
                retrieved.append({**r, "text": match["text"] if match else ""})
                if len(retrieved) >= self.k:
                    break
        t1 = time.perf_counter()

        if self.llm_available:
            response, usage = self._generate_llm(context, retrieved)
            generator = "llm_groq"
        else:
            response = rule_based_insight(context)
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            generator = "heuristica"
        t2 = time.perf_counter()

        return {
            "store": int(store),
            "item": int(item),
            "context": context,
            "retrieved": retrieved,
            "response": response,
            "generator": generator,
            "llm_model": self.model if self.llm_available else "none",
            "usage": usage,
            "latency_total_ms": (t2 - t0) * 1000,
            "latency_retrieval_ms": (t1 - t0) * 1000,
            "latency_generation_ms": (t2 - t1) * 1000,
        }

    def _generate_llm(self, context: dict, retrieved: list[dict]) -> tuple[str, dict]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(context, retrieved)},
        ]
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = resp.choices[0].message.content.strip()
        usage = {
            "prompt_tokens": int(resp.usage.prompt_tokens),
            "completion_tokens": int(resp.usage.completion_tokens),
            "total_tokens": int(resp.usage.total_tokens),
        }
        return text, usage


# ---------------------------------------------------------------------------
# Registro en MLflow
# ---------------------------------------------------------------------------
def log_insight_run(result: dict, experiment_name: str) -> str:
    """Registra un insight como run de MLflow (experimento del agente).

    Params: serie, generador y modelo LLM. Metrics: latencias, tokens y
    cobertura de la serie. Artefactos: contexto, series similares e insight.

    Args:
        result: salida de :meth:`InsightsAgent.generate`.
        experiment_name: experimento de MLflow destino.

    Returns:
        run_id del run creado.
    """
    import mlflow

    mlflow.set_experiment(experiment_name)
    response = result["response"]
    references = (
        f"tienda {result['store']}" in response.lower()
        and f"artículo {result['item']}" in response.lower()
    )

    with mlflow.start_run(run_name=f"insight_tienda{result['store']}_articulo{result['item']}"):
        mlflow.log_params({
            "store": result["store"],
            "item": result["item"],
            "generator": result["generator"],
            "llm_model": result["llm_model"],
            "retrieval_k": len(result["retrieved"]),
        })
        mlflow.log_metrics({
            "latency_total_ms": result["latency_total_ms"],
            "latency_retrieval_ms": result["latency_retrieval_ms"],
            "latency_generation_ms": result["latency_generation_ms"],
            "prompt_tokens": result["usage"]["prompt_tokens"],
            "completion_tokens": result["usage"]["completion_tokens"],
            "total_tokens": result["usage"]["total_tokens"],
            "references_series": int(references),
        })
        context_for_log = {
            "store": result["store"],
            "item": result["item"],
            "card": result["context"]["card"],
        }
        mlflow.log_dict(context_for_log, "contexto.json")
        mlflow.log_text(
            "\n".join(
                f"- tienda {r['store']}, artículo {r['item']} "
                f"(similitud {r['similarity']:.2f}): {r['text']}"
                for r in result["retrieved"]
            ),
            "series_similares.txt",
        )
        mlflow.log_text(response, "insight.txt")
        run_id = mlflow.active_run().info.run_id
    return run_id
