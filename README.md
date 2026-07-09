# Proyecta Innova · Sistema de Optimización de Portafolios

Proyecto académico de **Análisis y Diseño de Algoritmos** — comparación de tres estrategias
de optimización de portafolios (Markowitz, NSGA-II y Programación Dinámica) sobre datos
históricos reales descargados de Yahoo Finance.

El proyecto tiene dos entregables equivalentes en lógica pero independientes en ejecución:

- Una **aplicación web interactiva** en Streamlit (`app.py`).
- Cuatro **notebooks de Jupyter/Colab** (`1_Datos_Markowitz.ipynb` … `4_Comparacion.ipynb`),
  con la misma lógica desacoplada de Streamlit, pensados para la sustentación y revisión
  de código paso a paso.

---

## 1. Estructura del repositorio

```
├── app.py                          # Aplicación Streamlit (4 pestañas interactivas)
├── requirements.txt                # Dependencias para Streamlit Community Cloud
├── 1_Datos_Markowitz.ipynb         # Módulo 1: datos + optimización media-varianza
├── 2_NSGA2_Multiobjetivo.ipynb     # Módulo 2: algoritmo genético NSGA-II (DEAP)
├── 3_DP_Rebalanceo.ipynb           # Módulo 3: programación dinámica (Bellman)
├── 4_Comparacion.ipynb             # Módulo 4: consolidación y ranking final
└── README.md
```

Al ejecutar los notebooks 1, 2 y 3 se generan localmente (o en el runtime de Colab)
`resultados_m1.json`, `resultados_m2.json` y `resultados_m3.json`, que el notebook 4
consume con `json.load` para construir la comparativa final.

---

## 2. Módulos y algoritmos implementados

| # | Módulo | Algoritmo | Librería principal |
|---|--------|-----------|---------------------|
| 1 | Datos y Markowitz | Optimización media-varianza (Máx. Sharpe / Mín. Varianza) | `scipy.optimize` (SLSQP) |
| 2 | NSGA-II Multiobjetivo | Algoritmo genético con ordenamiento no dominado | `deap` |
| 3 | Programación Dinámica | Backward induction sobre la ecuación de Bellman | `numpy` |
| 4 | Comparación | Consolidación de métricas y ranking | `pandas` |

### 2.1 Módulo 1 — Markowitz

Descarga precios ajustados con `yfinance`, calcula retornos logarítmicos y los estadísticos
anualizados $\mu$ y $\Sigma$. Resuelve los portafolios de **Máximo Sharpe** y **Mínima
Varianza** con `scipy.optimize.minimize` (método SLSQP, sin ventas en corto), traza la
frontera eficiente de 200 puntos y simula la evolución de riqueza (`CAPITAL` inicial) para
tres estrategias: Buy & Hold, rebalanceo periódico y equiponderado.

### 2.2 Módulo 2 — NSGA-II

Implementa el algoritmo genético completo con `deap`: individuos como vectores de pesos,
fitness biobjetivo (maximizar retorno, minimizar volatilidad), cruce
`cxSimulatedBinaryBounded`, mutación `mutPolynomialBounded` y selección `selNSGA2` con
torneo por distancia de crowding (`selTournamentDCD`). Se calcula un indicador de
**hypervolumen 2D** por generación para evidenciar la convergencia, y se extraen tres
portafolios representativos del frente de Pareto (Conservador, Balanceado, Agresivo).

### 2.3 Módulo 3 — Programación Dinámica

> **Nota metodológica importante:** para que el espacio de estados fuera computacionalmente
> tratable, el problema se reduce a un **estado escalar** $s_t \in [0,1]$: la fracción del
> capital expuesta al portafolio óptimo (tangente) de Markowitz, con el resto en efectivo.
> No es una DP sobre el vector completo de pesos por activo — es una simplificación
> deliberada y debe mencionarse como supuesto en el informe.

Se discretiza $s$ en una grilla `Δw`, se define una recompensa por periodo de tipo utilidad
media-varianza (Merton) y un costo de ajuste lineal entre estados. La resolución es
backward induction clásica: $J^*_t(s) = \max_{s'} [r(s') - c(s,s') + J^*_{t+1}(s')]$,
con $J^*_T = 0$.

### 2.4 Módulo 4 — Comparación

Consolida 7 curvas de riqueza (3 de Markowitz/Equiponderado, 3 de NSGA-II, 1 de DP),
calcula métricas reales (Retorno Total, Retorno Anualizado, Volatilidad, Sharpe, Sortino,
Max Drawdown, Riqueza Final) y genera un ranking ordenado por Sharpe Ratio.

> La curva de DP usa periodicidad **mensual** (12 periodos/año); las demás usan
> periodicidad **diaria** (252 sesiones/año). La tasa libre de riesgo se asume en 0%.

---

## 3. Cómo ejecutar la aplicación Streamlit

### 3.1 Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

### 3.2 Streamlit Community Cloud

1. Sube `app.py` y `requirements.txt` a un repositorio de GitHub.
2. En [share.streamlit.io](https://share.streamlit.io), conecta el repositorio y selecciona
   `app.py` como archivo principal.
3. Configura los parámetros desde el `sidebar` (tickers, fechas, capital, frecuencia de
   rebalanceo) y presiona **🔄 Cargar Datos** seguido de **⚡ Ejecutar Optimización**.

### 3.3 Parámetros configurables en el sidebar

| Parámetro | Descripción | Valor por defecto |
|---|---|---|
| Tickers de Activos | Símbolos de Yahoo Finance separados por coma | `FSM, VOLCABC1.LM, ABX.TO, BVN, BHP` |
| Fecha Inicio / Fin | Rango histórico de descarga | `2015-01-01` – `2024-12-31` |
| Capital Inicial | Capital base para el backtesting (USD) | `100000` |
| Frecuencia de Rebalanceo | Semanal / Mensual / Trimestral | `Mensual` |
| Tamaño de Población / Generaciones | Hiperparámetros de NSGA-II (pestaña 2) | `200` / `100` |
| Costo de Transacción / Horizonte / Δw | Parámetros del modelo DP (pestaña 3) | `0.1%` / `12` / `0.01` |

---

## 4. Cómo ejecutar los notebooks en Google Colab

1. Sube cada `.ipynb` a Colab (`Archivo → Subir cuaderno`) o ábrelos directamente desde
   GitHub si el repositorio es público.
2. Ejecuta **en orden** dentro de la misma sesión de Colab, o descarga los `.json`
   generados por cada uno y súbelos al entorno del siguiente:
   1. `1_Datos_Markowitz.ipynb` → genera `resultados_m1.json`
   2. `2_NSGA2_Multiobjetivo.ipynb` → genera `resultados_m2.json`
   3. `3_DP_Rebalanceo.ipynb` → genera `resultados_m3.json`
   4. `4_Comparacion.ipynb` → lee los tres JSON y produce la comparativa final
3. Cada notebook instala sus propias dependencias en la primera celda (`!pip install ...`),
   por lo que son ejecutables de forma independiente en un runtime limpio de Colab.
4. Los parámetros de cada módulo (tickers, fechas, capital, hiperparámetros de NSGA-II,
   costo de transacción, horizonte, etc.) están expuestos como **variables globales en
   mayúsculas** al inicio de cada notebook, listas para modificar sin tocar el resto del
   código.

---

## 5. Dependencias

```
streamlit
numpy
pandas
plotly
yfinance
scipy
deap
```

`deap` y `yfinance` solo son necesarios para los módulos 2 (NSGA-II) y para la descarga de
datos, respectivamente; `streamlit` no se requiere para ejecutar los notebooks.

---

## 6. Limitaciones y supuestos conocidos

- La tasa libre de riesgo se asume en **0%** para todos los cálculos de Sharpe/Sortino.
- No se modelan costos de transacción en Markowitz ni en NSGA-II (solo en el módulo de DP).
- El modelo de DP opera sobre un **estado escalar** (ver §2.3), no sobre el vector completo
  de pesos por activo — es la simplificación que permite resolver Bellman por backward
  induction en tiempo razonable.
- Los datos dependen de la disponibilidad y calidad de Yahoo Finance para cada ticker; los
  tickers internacionales (`.LM`, `.TO`, etc.) requieren que Yahoo Finance tenga cobertura
  para esa plaza bursátil.
