import random
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as ob
import yfinance as yf
from scipy.optimize import minimize
from deap import base, creator, tools

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA Y TEMA VISUAL (Estilo Corporativo / Premium)
# ==============================================================================
st.set_page_config(
    page_title="Proyecta Innova · Sistema de Optimización de Portafolios",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --ink: #1E2A22;
            --soil: #6B4F3B;
            --sage: #7C9473;
            --paper: #FBF8F1;
            --sand-deep: #E9DEC7;
        }

        .brand-title { font-family: 'Fraunces', serif; font-weight: 600; color: var(--ink); }
        .mono-text { font-family: 'IBM Plex Mono', monospace; font-size: 11px; }

        .financial-card {
            background-color: white;
            border: 1px solid #DCD2B8;
            border-radius: 4px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        }
        .financial-card h3 { font-family: 'Fraunces', serif; color: var(--ink); margin-bottom: 10px; }

        .framework-note {
            background-color: #E9DEC7;
            border-left: 3px solid #6B4F3B;
            padding: 12px;
            font-size: 13px;
            color: #3D4F4A;
            border-radius: 0 4px 4px 0;
        }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# FUNCIONES CORE: DATOS DE MERCADO
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def cargar_datos(tickers_str, start, end):
    """Descarga precios ajustados de Yahoo Finance y calcula retornos log,
    mu anualizado y matriz de covarianza anualizada."""
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if raw is None or raw.empty:
        raise ValueError("No se encontraron datos para los tickers/fechas indicados.")

    if isinstance(raw.columns, pd.MultiIndex):
        precios = raw["Close"]
    else:
        precios = raw[["Close"]]
        precios.columns = tickers

    precios = precios.dropna(how="all").ffill().dropna()
    tickers = list(precios.columns)

    log_returns = np.log(precios / precios.shift(1)).dropna()
    mu = log_returns.mean() * 252
    cov = log_returns.cov() * 252

    return precios, log_returns, mu, cov, tickers


# ==============================================================================
# FUNCIONES CORE: MARKOWITZ (MEDIA-VARIANZA)
# ==============================================================================
def rendimiento_portafolio(pesos, mu, cov):
    ret = float(np.dot(pesos, mu))
    vol = float(np.sqrt(pesos @ cov @ pesos))
    return ret, vol


def _neg_sharpe(pesos, mu, cov, rf=0.0):
    ret, vol = rendimiento_portafolio(pesos, mu, cov)
    if vol == 0:
        return 0.0
    return -(ret - rf) / vol


def _volatilidad(pesos, mu, cov):
    return rendimiento_portafolio(pesos, mu, cov)[1]


def optimizar_max_sharpe(mu, cov, rf=0.0):
    n = len(mu)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    init = np.repeat(1.0 / n, n)
    res = minimize(_neg_sharpe, init, args=(mu.values, cov.values, rf),
                    method="SLSQP", bounds=bounds, constraints=cons)
    return res.x if res.success else init


def optimizar_min_varianza(mu, cov):
    n = len(mu)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    init = np.repeat(1.0 / n, n)
    res = minimize(_volatilidad, init, args=(mu.values, cov.values),
                    method="SLSQP", bounds=bounds, constraints=cons)
    return res.x if res.success else init


def frontera_eficiente(mu, cov, n_puntos=200):
    n = len(mu)
    objetivos = np.linspace(mu.min(), mu.max(), n_puntos)
    vols, pesos_list = [], []
    bounds = tuple((0.0, 1.0) for _ in range(n))
    init = np.repeat(1.0 / n, n)
    for target in objetivos:
        cons = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: float(np.dot(w, mu.values)) - t},
        )
        res = minimize(_volatilidad, init, args=(mu.values, cov.values),
                        method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            vols.append(res.fun)
            pesos_list.append(res.x)
        else:
            vols.append(np.nan)
            pesos_list.append(np.full(n, np.nan))
    return objetivos, np.array(vols), pesos_list


# ==============================================================================
# FUNCIONES CORE: SIMULACIÓN DE RIQUEZA (BACKTESTING)
# ==============================================================================
def _fechas_rebalanceo(precios_index, freq_label):
    freq_map = {"Semanal": "W", "Mensual": "M", "Trimestral": "Q"}
    freq = freq_map.get(freq_label, "M")
    serie = pd.Series(precios_index, index=precios_index)
    ultimas = serie.resample(freq).last().dropna()
    return set(ultimas.values)


def simular_riqueza(retornos_simples, pesos_objetivo, capital, freq_label=None):
    """retornos_simples: DataFrame de retornos diarios simples (pct_change).
    Si freq_label es None -> Buy & Hold (sin rebalanceo)."""
    pesos_objetivo = np.array(pesos_objetivo, dtype=float)
    fechas = retornos_simples.index
    rebal_dates = _fechas_rebalanceo(fechas, freq_label) if freq_label else set()

    valores_activos = capital * pesos_objetivo
    wealth = []
    for fecha, fila in retornos_simples.iterrows():
        valores_activos = valores_activos * (1 + fila.values)
        total = valores_activos.sum()
        wealth.append(total)
        if freq_label and fecha in rebal_dates:
            valores_activos = total * pesos_objetivo
    return pd.Series(wealth, index=fechas)


def metricas_desempeno(wealth_series, periodos_por_anio=252):
    w = pd.Series(wealth_series).astype(float)
    if len(w) < 2:
        return None
    rets = w.pct_change().dropna()
    retorno_total = w.iloc[-1] / w.iloc[0] - 1
    n_periodos = len(w)
    retorno_anual = (1 + retorno_total) ** (periodos_por_anio / max(n_periodos, 1)) - 1
    vol_anual = rets.std() * np.sqrt(periodos_por_anio)
    sharpe = retorno_anual / vol_anual if vol_anual and vol_anual > 0 else np.nan
    downside = rets[rets < 0].std()
    downside_anual = downside * np.sqrt(periodos_por_anio) if downside and not np.isnan(downside) else np.nan
    sortino = retorno_anual / downside_anual if downside_anual and downside_anual > 0 else np.nan
    drawdown = w / w.cummax() - 1
    max_dd = drawdown.min()
    return {
        "Retorno Total (%)": round(retorno_total * 100, 2),
        "Retorno Anualizado (%)": round(retorno_anual * 100, 2),
        "Volatilidad Anualizada (%)": round(vol_anual * 100, 2),
        "Sharpe Ratio": round(sharpe, 2) if pd.notna(sharpe) else np.nan,
        "Sortino Ratio": round(sortino, 2) if pd.notna(sortino) else np.nan,
        "Max Drawdown (%)": round(max_dd * 100, 2),
        "Riqueza Final (USD)": round(w.iloc[-1], 2),
    }


# ==============================================================================
# FUNCIONES CORE: NSGA-II (DEAP)
# ==============================================================================
def _normalizar_individuo(ind, n):
    arr = np.clip(np.array(ind, dtype=float), 0.0, None)
    s = arr.sum()
    return (arr / s) if s > 0 else np.repeat(1.0 / n, n)


def ejecutar_nsga2(mu, cov, pop_size=200, ngen=100, cxpb=0.7, mutpb=0.3, seed=42):
    n = len(mu)
    mu_arr, cov_arr = mu.values, cov.values
    random.seed(seed)

    # Ajuste a múltiplo de 4 (requisito de selTournamentDCD)
    pop_size = max(8, int(round(pop_size / 4.0)) * 4)

    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluar(ind):
        w = _normalizar_individuo(ind, n)
        ret = float(np.dot(w, mu_arr))
        vol = float(np.sqrt(w @ cov_arr @ w))
        return ret, vol

    toolbox.register("evaluate", evaluar)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20.0, indpb=1.0 / n)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    pop = toolbox.select(pop, len(pop))

    # Punto de referencia fijo para hypervolumen 2D (peor caso dominado)
    ref_ret = min(mu_arr.min(), 0.0) - 0.05
    ref_vol = float(np.sqrt(np.diag(cov_arr)).max()) * 1.2

    def hipervolumen_2d(frente):
        pts = sorted(((ind.fitness.values[0], ind.fitness.values[1]) for ind in frente),
                     key=lambda p: p[1])
        hv, prev_vol = 0.0, ref_vol
        for ret, vol in pts:
            if vol < prev_vol:
                hv += max(0.0, prev_vol - vol) * max(0.0, ret - ref_ret)
                prev_vol = vol
        return hv

    hv_historia = []
    for _gen in range(ngen):
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= cxpb:
                toolbox.mate(c1, c2)
            if random.random() <= mutpb:
                toolbox.mutate(c1)
                toolbox.mutate(c2)
            del c1.fitness.values, c2.fitness.values

        invalidos = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalidos:
            ind.fitness.values = toolbox.evaluate(ind)

        pop = toolbox.select(pop + offspring, pop_size)
        frente0 = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        hv_historia.append(hipervolumen_2d(frente0))

    frente_final = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    resultados = []
    for ind in frente_final:
        w = _normalizar_individuo(ind, n)
        ret, vol = ind.fitness.values
        sharpe = ret / vol if vol > 0 else 0.0
        resultados.append({"weights": w, "ret": ret, "vol": vol, "sharpe": sharpe})
    resultados.sort(key=lambda r: r["vol"])
    return resultados, hv_historia


# ==============================================================================
# FUNCIONES CORE: PROGRAMACIÓN DINÁMICA (BACKWARD INDUCTION / BELLMAN)
# ==============================================================================
def ejecutar_dp_rebalanceo(mu_p, sigma_p, costo_trans_pct, horizonte, grid_step, aversion_riesgo=3.0):
    """Estado escalar s = fracción del capital en el portafolio óptimo (tangente),
    resto en efectivo (retorno 0). Recompensa tipo utilidad media-varianza (Merton).
    Backward induction resuelve J*(t, s) minimizando costos de ajuste."""
    grid = np.round(np.arange(0.0, 1.0 + grid_step, grid_step), 6)
    grid = grid[grid <= 1.0 + 1e-9]
    n_estados = len(grid)

    recompensa = grid * mu_p - 0.5 * aversion_riesgo * (grid ** 2) * (sigma_p ** 2)
    costo_pct = costo_trans_pct / 100.0
    matriz_costos = costo_pct * np.abs(grid.reshape(-1, 1) - grid.reshape(1, -1))  # [origen, destino]

    J = np.zeros((horizonte + 1, n_estados))
    politica = np.zeros((horizonte, n_estados), dtype=int)

    for t in range(horizonte - 1, -1, -1):
        for i in range(n_estados):
            valores = recompensa - matriz_costos[i, :] + J[t + 1, :]
            j_best = int(np.argmax(valores))
            J[t, i] = valores[j_best]
            politica[t, i] = j_best

    return grid, J, politica, matriz_costos


def simular_dp_forward(grid, politica, retornos_periodicos, costo_trans_pct, capital, horizonte):
    horizonte = min(horizonte, len(retornos_periodicos))
    retornos = retornos_periodicos.values[:horizonte]
    fechas = retornos_periodicos.index[:horizonte]
    costo_pct = costo_trans_pct / 100.0

    idx_ini = int(np.argmin(np.abs(grid - 0.5)))

    # Estrategia DP óptima
    wealth_dp, w_cur, idx_cur = [capital], grid[idx_ini], idx_ini
    for t in range(horizonte):
        r = retornos[t]
        idx_next = politica[t, idx_cur]
        w_next = grid[idx_next]
        costo = costo_pct * abs(w_next - w_cur)
        wealth_dp.append(wealth_dp[-1] * (1 + w_cur * r) * (1 - costo))
        w_cur, idx_cur = w_next, idx_next

    # Buy & Hold: exposición inicial fija, jamás se ajusta
    w_fijo = grid[idx_ini]
    wealth_bh = [capital]
    for t in range(horizonte):
        wealth_bh.append(wealth_bh[-1] * (1 + w_fijo * retornos[t]))

    # Siempre rebalanceado a exposición total (w=1) cada periodo
    wealth_full = [capital]
    for t in range(horizonte):
        costo = costo_pct * (0 if t == 0 else 0)  # ya está en el target, sin costo recurrente
        wealth_full.append(wealth_full[-1] * (1 + 1.0 * retornos[t]))

    return fechas, wealth_dp[1:], wealth_bh[1:], wealth_full[1:]


# ==============================================================================
# SIDEBAR / PANEL DE PARÁMETROS GLOBAL
# ==============================================================================
with st.sidebar:
    st.markdown('<div class="mono-text" style="color: #6E7C68;">PROYECTA INNOVA</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="brand-title" style="font-size: 20px; margin-bottom: 20px;">Portafolio Optimizador</h2>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 🛠️ Configuración Global")

    tickers = st.text_input(
        "Tickers de Activos (separados por coma)",
        value="FSM, VOLCABC1.LM, ABX.TO, BVN, BHP"
    )

    col_dates = st.columns(2)
    with col_dates[0]:
        start_date = st.date_input("Fecha Inicio", value=pd.to_datetime("2015-01-01"))
    with col_dates[1]:
        end_date = st.date_input("Fecha Fin", value=pd.to_datetime("2024-12-31"))

    capital = st.number_input("Capital Inicial (USD)", value=100000, step=10000)

    frequency = st.selectbox(
        "Frecuencia de Rebalanceo",
        options=["Semanal", "Mensual", "Trimestral"],
        index=1
    )

    st.markdown("---")

    btn_load = st.button("🔄 Cargar Datos", width="stretch")
    btn_exec = st.button("⚡ Ejecutar Optimización", width="stretch", type="primary")

    st.markdown("<br>" * 5, unsafe_allow_html=True)
    st.markdown('<div class="mono-text" style="text-align: center; color: #9C9384;">MVP v1.0 · 2026</div>', unsafe_allow_html=True)

# Carga de datos: por el botón explícito, o automáticamente si se pide ejecutar sin datos previos
if btn_load or (btn_exec and "log_returns" not in st.session_state):
    try:
        with st.spinner("Descargando datos de Yahoo Finance y calculando estadísticos..."):
            precios, log_returns, mu, cov, tick_list = cargar_datos(tickers, start_date, end_date)
            st.session_state["precios"] = precios
            st.session_state["log_returns"] = log_returns
            st.session_state["retornos_simples"] = precios.pct_change().dropna()
            st.session_state["mu"] = mu
            st.session_state["cov"] = cov
            st.session_state["tick_list"] = tick_list
            st.session_state["data_loaded"] = True
        st.sidebar.success(f"Datos cargados: {len(tick_list)} activos, {len(precios)} sesiones.")
    except Exception as e:
        st.sidebar.error(f"Error al cargar datos: {e}")

datos_listos = st.session_state.get("data_loaded", False)

# ==============================================================================
# CONTROL DE NAVEGACIÓN
# ==============================================================================
tabs = st.tabs([
    "📈 1. Datos y Markowitz",
    "🧬 2. NSGA-II Multiobjetivo",
    "⏱️ 3. Programación Dinámica",
    "📊 4. Comparación de Métodos"
])

# ==============================================================================
# MÓDULO 1: DATOS Y MARKOWITZ
# ==============================================================================
with tabs[0]:
    st.markdown('<div class="mono-text" style="color: #6B4F3B;">EMBUDO CU-01 / CU-02</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="brand-title">Teoría Moderna de Portafolio (Markowitz)</h1>', unsafe_allow_html=True)
    st.markdown("Módulo analítico clásico para la maximización del ratio de Sharpe basado en la media-varianza histórica de los activos.")

    st.markdown("""
    <div class="financial-card">
        <h3>Estructura del Modelo</h3>
        <p>Este lienzo procesa la matriz de covarianza y los retornos esperados de los activos cargados en el sidebar para proyectar el portafolio de mínima varianza y la asignación óptima tangente.</p>
    </div>
    """, unsafe_allow_html=True)

    if not datos_listos:
        st.info("Carga los datos desde el sidebar (🔄 Cargar Datos) para iniciar el análisis.")
    else:
        mu, cov = st.session_state["mu"], st.session_state["cov"]
        tick_list = st.session_state["tick_list"]

        if btn_exec:
            with st.spinner("Optimizando portafolios de Markowitz (SLSQP)..."):
                w_sharpe = optimizar_max_sharpe(mu, cov)
                w_minvar = optimizar_min_varianza(mu, cov)
                objetivos, vols_frontera, _ = frontera_eficiente(mu, cov, n_puntos=200)

                retornos_simples = st.session_state["retornos_simples"]
                w_eq = np.repeat(1.0 / len(tick_list), len(tick_list))

                wealth_bh = simular_riqueza(retornos_simples, w_sharpe, capital, freq_label=None)
                wealth_mkw = simular_riqueza(retornos_simples, w_sharpe, capital, freq_label=frequency)
                wealth_eq = simular_riqueza(retornos_simples, w_eq, capital, freq_label=frequency)

                st.session_state["markowitz"] = {
                    "w_sharpe": w_sharpe, "w_minvar": w_minvar,
                    "objetivos": objetivos, "vols_frontera": vols_frontera,
                    "wealth_bh": wealth_bh, "wealth_mkw": wealth_mkw, "wealth_eq": wealth_eq,
                    "w_eq": w_eq,
                }

        if "markowitz" not in st.session_state:
            st.warning("Presiona ⚡ Ejecutar Optimización en el sidebar para calcular los portafolios.")
        else:
            mk = st.session_state["markowitz"]
            ret_s, vol_s = rendimiento_portafolio(mk["w_sharpe"], mu, cov)
            sharpe_s = ret_s / vol_s if vol_s > 0 else np.nan
            ret_mv, vol_mv = rendimiento_portafolio(mk["w_minvar"], mu, cov)

            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Sharpe Ratio (Máx. Sharpe)", f"{sharpe_s:.2f}", f"Retorno {ret_s*100:.1f}%")
            with m_col2:
                met_bh = metricas_desempeno(mk["wealth_bh"])
                st.metric("Sortino Ratio (Buy & Hold)", f"{met_bh['Sortino Ratio']}", "Análisis de Downside")
            with m_col3:
                st.metric("Volatilidad Portafolio Mín. Var. (σ)", f"{vol_mv*100:.1f}%", "Punto de mínima varianza")

            st.markdown("---")

            g_col1, g_col2 = st.columns(2)
            with g_col1:
                st.markdown("#### Frontera Eficiente de Markowitz")
                fig_ef = px.line(x=mk["vols_frontera"], y=mk["objetivos"],
                                  labels={"x": "Volatilidad (Riesgo)", "y": "Retorno Esperado"})
                fig_ef.add_scatter(x=[vol_s], y=[ret_s], mode="markers", name="Máx. Sharpe",
                                    marker=dict(size=13, color="#C4622D"))
                fig_ef.add_scatter(x=[vol_mv], y=[ret_mv], mode="markers", name="Mín. Varianza",
                                    marker=dict(size=13, color="#3D4F4A"))
                vol_activos = np.sqrt(np.diag(cov.values))
                fig_ef.add_scatter(x=vol_activos, y=mu.values, mode="markers+text", name="Activos",
                                    text=tick_list, textposition="top center",
                                    marker=dict(size=8, color="#7C9473"))
                fig_ef.update_layout(plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_ef, use_container_width=True)

            with g_col2:
                st.markdown("#### Asignación de Pesos Óptimos")
                sub1, sub2 = st.columns(2)
                with sub1:
                    st.caption("Máximo Sharpe")
                    st.plotly_chart(px.pie(names=tick_list, values=mk["w_sharpe"],
                                            color_discrete_sequence=px.colors.sequential.YlGnBu),
                                     use_container_width=True, key="pie_sharpe")
                with sub2:
                    st.caption("Mínima Varianza")
                    st.plotly_chart(px.pie(names=tick_list, values=mk["w_minvar"],
                                            color_discrete_sequence=px.colors.sequential.Sunset),
                                     use_container_width=True, key="pie_minvar")

            st.markdown("#### Evolución de la Riqueza (Backtesting Histórico)")
            fig_wealth = ob.Figure()
            fig_wealth.add_trace(ob.Scatter(x=mk["wealth_bh"].index, y=mk["wealth_bh"].values,
                                             name="Buy & Hold (Máx. Sharpe)", line=dict(color="#B3452F", dash="dash")))
            fig_wealth.add_trace(ob.Scatter(x=mk["wealth_mkw"].index, y=mk["wealth_mkw"].values,
                                             name=f"Rebalanceado ({frequency})", line=dict(color="#7C9473", width=3)))
            fig_wealth.add_trace(ob.Scatter(x=mk["wealth_eq"].index, y=mk["wealth_eq"].values,
                                             name="Equiponderado", line=dict(color="#3D4F4A", dash="dot")))
            fig_wealth.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                      xaxis_title="Fecha", yaxis_title="Valor del Portafolio (USD)")
            st.plotly_chart(fig_wealth, use_container_width=True)

            resumen_csv = pd.DataFrame({
                "Ticker": tick_list,
                "Peso Máx. Sharpe": mk["w_sharpe"],
                "Peso Mín. Varianza": mk["w_minvar"],
            }).to_csv(index=False).encode("utf-8")
            st.download_button("📥 Descargar Reporte de Pesos (CSV)", data=resumen_csv,
                                file_name="markowitz_pesos.csv", mime="text/csv", key="btn_dl_m1")

# ==============================================================================
# MÓDULO 2: NSGA-II MULTIOBJETIVO
# ==============================================================================
with tabs[1]:
    st.markdown('<div class="mono-text" style="color: #6B4F3B;">ALGORITMOS EVOLUTIVOS · CU-03</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="brand-title">Optimización Metaheurística Multiobjetivo (NSGA-II)</h1>', unsafe_allow_html=True)
    st.markdown("Cálculo del frente de Pareto para optimización simultánea de múltiples variables en conflicto (Retorno vs. Volatilidad).")

    st.markdown("### ⚙️ Hiperparámetros del Algoritmo Genético")
    ctrl_col1, ctrl_col2 = st.columns(2)
    with ctrl_col1:
        population = st.slider("Tamaño de la Población", min_value=50, max_value=500, value=200, step=50)
    with ctrl_col2:
        generations = st.slider("Número de Generaciones", min_value=10, max_value=200, value=100, step=10)

    st.markdown("---")

    if not datos_listos:
        st.info("Carga los datos desde el sidebar para habilitar este módulo.")
    else:
        mu, cov = st.session_state["mu"], st.session_state["cov"]
        tick_list = st.session_state["tick_list"]

        if btn_exec:
            with st.spinner(f"Ejecutando NSGA-II ({population} individuos × {generations} generaciones)..."):
                frente, hv_historia = ejecutar_nsga2(mu, cov, pop_size=population, ngen=generations)
                st.session_state["nsga2"] = {"frente": frente, "hv_historia": hv_historia}

        if "nsga2" not in st.session_state:
            st.warning("Presiona ⚡ Ejecutar Optimización en el sidebar para correr el algoritmo genético.")
        else:
            frente = st.session_state["nsga2"]["frente"]
            hv_historia = st.session_state["nsga2"]["hv_historia"]

            g2_col1, g2_col2 = st.columns(2)
            with g2_col1:
                st.markdown("#### Frente No Dominado de Pareto (coloreado por Sharpe)")
                rets = [r["ret"] for r in frente]
                vols = [r["vol"] for r in frente]
                sharpes = [r["sharpe"] for r in frente]
                fig_pareto = px.scatter(x=vols, y=rets, color=sharpes,
                                         labels={"x": "Volatilidad", "y": "Retorno", "color": "Sharpe"},
                                         color_continuous_scale="YlGnBu")
                if "markowitz" in st.session_state:
                    mk = st.session_state["markowitz"]
                    fig_pareto.add_scatter(x=mk["vols_frontera"], y=mk["objetivos"], mode="lines",
                                            name="Frontera Markowitz", line=dict(color="#B3452F", dash="dash"))
                fig_pareto.update_layout(plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_pareto, use_container_width=True)

            with g2_col2:
                st.markdown("#### Evolución del Hypervolumen (Métrica de Convergencia)")
                fig_hv = px.line(x=list(range(1, len(hv_historia) + 1)), y=hv_historia,
                                  labels={"x": "Generación", "y": "Hypervolume Indicator"})
                fig_hv.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig_hv, use_container_width=True)

            st.markdown("#### Distribución de Pesos en Puntos Clave del Frente de Pareto")
            idx_conservador = int(np.argmin(vols))
            idx_agresivo = int(np.argmax(rets))
            idx_balanceado = int(np.argmax(sharpes))

            pie_col1, pie_col2, pie_col3 = st.columns(3)
            with pie_col1:
                st.caption(f"Portafolio A: Conservador (σ={vols[idx_conservador]*100:.1f}%)")
                st.plotly_chart(px.pie(names=tick_list, values=frente[idx_conservador]["weights"], hole=0.3),
                                 use_container_width=True, key="p1")
            with pie_col2:
                st.caption(f"Portafolio B: Balanceado (Sharpe={sharpes[idx_balanceado]:.2f})")
                st.plotly_chart(px.pie(names=tick_list, values=frente[idx_balanceado]["weights"], hole=0.3),
                                 use_container_width=True, key="p2")
            with pie_col3:
                st.caption(f"Portafolio C: Agresivo (Retorno={rets[idx_agresivo]*100:.1f}%)")
                st.plotly_chart(px.pie(names=tick_list, values=frente[idx_agresivo]["weights"], hole=0.3),
                                 use_container_width=True, key="p3")

            df_frente = pd.DataFrame({
                "Retorno": rets, "Volatilidad": vols, "Sharpe": sharpes,
            })
            for i, tck in enumerate(tick_list):
                df_frente[tck] = [r["weights"][i] for r in frente]
            st.download_button("📥 Exportar Datos de Frontera Pareto (.CSV)",
                                data=df_frente.to_csv(index=False).encode("utf-8"),
                                file_name="nsga2_pareto.csv", mime="text/csv", key="btn_dl_m2")

# ==============================================================================
# MÓDULO 3: PROGRAMACIÓN DINÁMICA PARA REBALANCEO
# ==============================================================================
with tabs[2]:
    st.markdown('<div class="mono-text" style="color: #6B4F3B;">OPTIMIZACIÓN TEMPORAL INTERTRAZAS · CU-04</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="brand-title">Rebalanceo Óptimo Multiperiodo vía Programación Dinámica</h1>', unsafe_allow_html=True)
    st.markdown("Resolución recursiva hacia atrás (Bellman) sobre un estado escalar de exposición al portafolio óptimo (tangente) vs. efectivo, minimizando el impacto de los costos de transacción.")

    st.markdown("### 🎛️ Parámetros del Espacio de Estados")
    ctrl3_col1, ctrl3_col2, ctrl3_col3 = st.columns(3)
    with ctrl3_col1:
        cost_trans = st.slider("Costo de Transacción Lineal (%)", min_value=0.0, max_value=2.0, value=0.1, step=0.05)
    with ctrl3_col2:
        horizon = st.slider("Horizonte de Planificación (T en periodos)", min_value=4, max_value=52, value=12, step=4)
    with ctrl3_col3:
        grid_step = st.select_slider("Resolución del Paso de Grilla (Δw)", options=[0.05, 0.02, 0.01, 0.005], value=0.01)

    st.markdown("---")

    if not datos_listos:
        st.info("Carga los datos desde el sidebar para habilitar este módulo.")
    else:
        mu, cov = st.session_state["mu"], st.session_state["cov"]
        log_returns = st.session_state["log_returns"]

        if btn_exec:
            with st.spinner("Resolviendo la ecuación de Bellman (backward induction)..."):
                w_tangente = optimizar_max_sharpe(mu, cov)
                mu_p, sigma_p = rendimiento_portafolio(w_tangente, mu, cov)

                port_log_ret = log_returns.values @ w_tangente
                port_log_ret = pd.Series(port_log_ret, index=log_returns.index)
                monthly_log = port_log_ret.resample("M").sum()
                retornos_periodicos = np.exp(monthly_log) - 1

                grid, J, politica, matriz_costos = ejecutar_dp_rebalanceo(
                    mu_p, sigma_p, cost_trans, horizon, grid_step
                )
                fechas_dp, wealth_dp, wealth_bh_dp, wealth_full_dp = simular_dp_forward(
                    grid, politica, retornos_periodicos, cost_trans, capital, horizon
                )

                st.session_state["dp"] = {
                    "grid": grid, "J": J, "politica": politica, "matriz_costos": matriz_costos,
                    "fechas": fechas_dp, "wealth_dp": wealth_dp,
                    "wealth_bh": wealth_bh_dp, "wealth_full": wealth_full_dp,
                    "mu_p": mu_p, "sigma_p": sigma_p, "w_tangente": w_tangente,
                }

        if "dp" not in st.session_state:
            st.warning("Presiona ⚡ Ejecutar Optimización en el sidebar para resolver el modelo de Bellman.")
        else:
            dp = st.session_state["dp"]

            st.markdown("#### Matriz de Costes de Transición e Intensidad de Ajustes (Heatmap Espacio-Estado)")
            grid_labels = [f"{w:.2f}" for w in dp["grid"]]
            fig_heat = px.imshow(dp["matriz_costos"],
                                  labels=dict(x="Estado Destino (w_t)", y="Estado Origen (w_t-1)", color="Costo Ajuste"),
                                  x=grid_labels, y=grid_labels)
            st.plotly_chart(fig_heat, use_container_width=True)

            g3_col1, g3_col2 = st.columns(2)
            with g3_col1:
                st.markdown("#### Cronología y Decisiones de Ajuste (Timeline de Acción)")
                acciones = []
                for i in range(len(dp["wealth_dp"])):
                    acciones.append("Periodo t=" + str(i + 1))
                estado_actual = int(np.argmin(np.abs(dp["grid"] - 0.5)))
                decisiones = []
                for t in range(min(horizon, dp["politica"].shape[0])):
                    siguiente = dp["politica"][t, estado_actual]
                    delta = dp["grid"][siguiente] - dp["grid"][estado_actual]
                    if abs(delta) < 1e-9:
                        decisiones.append("No Cambiar")
                    elif delta > 0:
                        decisiones.append("Rebalancear Compra")
                    else:
                        decisiones.append("Rebalancear Venta")
                    estado_actual = siguiente
                df_actions = pd.DataFrame({
                    "Periodo": [f"T_{i}" for i in range(len(decisiones))],
                    "Decisión Óptima": decisiones,
                })
                st.dataframe(df_actions, use_container_width=True)

            with g3_col2:
                st.markdown("#### Trayectoria de Riqueza Acumulada vs. Estrategias Alternativas")
                fig_comp_line = ob.Figure()
                fig_comp_line.add_trace(ob.Scatter(x=list(range(len(dp["wealth_dp"]))), y=dp["wealth_dp"],
                                                     name="Estrategia Dinámica DP (Óptima)", line=dict(color="#7C9473", width=3)))
                fig_comp_line.add_trace(ob.Scatter(x=list(range(len(dp["wealth_bh"]))), y=dp["wealth_bh"],
                                                     name="Buy & Hold (Sin Rebalanceo)", line=dict(color="#B3452F", dash="dash")))
                fig_comp_line.add_trace(ob.Scatter(x=list(range(len(dp["wealth_full"]))), y=dp["wealth_full"],
                                                     name="Siempre Rebalanceado (w=1)", line=dict(color="#3D4F4A", dash="dot")))
                fig_comp_line.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                             xaxis_title="Periodo", yaxis_title="Valor del Portafolio (USD)",
                                             margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_comp_line, use_container_width=True)

# ==============================================================================
# MÓDULO 4: COMPARACIÓN DE MÉTODOS
# ==============================================================================
with tabs[3]:
    st.markdown('<div class="mono-text" style="color: #6B4F3B;">CUADRO DE MANDO DIRECTIVO · CU-05</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="brand-title">Comparativa de Modelos Estructurales</h1>', unsafe_allow_html=True)
    st.markdown("Consolidación agregada de las métricas de rendimiento inter-módulo bajo el mismo conjunto de datos históricos.")

    tiene_mk = "markowitz" in st.session_state
    tiene_ns = "nsga2" in st.session_state
    tiene_dp = "dp" in st.session_state

    if not (tiene_mk and tiene_ns and tiene_dp):
        st.info("Ejecuta la optimización (⚡) para poblar la comparativa con resultados reales de los 3 módulos.")
    else:
        mu, cov = st.session_state["mu"], st.session_state["cov"]
        tick_list = st.session_state["tick_list"]
        retornos_simples = st.session_state["retornos_simples"]
        mk = st.session_state["markowitz"]
        ns = st.session_state["nsga2"]
        dp = st.session_state["dp"]

        with st.spinner("Consolidando métricas comparativas..."):
            # Wealth NSGA-II (portafolio balanceado del frente, rebalanceado a la frecuencia global)
            sharpes = [r["sharpe"] for r in ns["frente"]]
            idx_balanceado = int(np.argmax(sharpes))
            w_ns = ns["frente"][idx_balanceado]["weights"]
            wealth_ns = simular_riqueza(retornos_simples, w_ns, capital, freq_label=frequency)

            met_mkw = metricas_desempeno(mk["wealth_mkw"], periodos_por_anio=252)
            met_ns = metricas_desempeno(wealth_ns, periodos_por_anio=252)
            met_dp = metricas_desempeno(dp["wealth_dp"], periodos_por_anio=12)
            met_eq = metricas_desempeno(mk["wealth_eq"], periodos_por_anio=252)

            filas = []
            for nombre, met in [
                ("Markowitz (Media-Varianza)", met_mkw),
                ("NSGA-II (Genético)", met_ns),
                ("Programación Dinámica Temporal", met_dp),
                ("Equiponderado", met_eq),
            ]:
                if met:
                    fila = {"Módulo de Optimización": nombre}
                    fila.update(met)
                    filas.append(fila)
            df_compare = pd.DataFrame(filas).sort_values("Sharpe Ratio", ascending=False).reset_index(drop=True)

        st.markdown("### 📋 Cuadro de Atributos y Eficiencia (ordenado por Sharpe Ratio)")
        st.dataframe(df_compare, use_container_width=True, hide_index=True)

        g4_col1, g4_col2 = st.columns(2)
        with g4_col1:
            st.markdown("#### Ranking de Consistencia de Modelos (Sharpe Ratio)")
            fig_bar = px.bar(df_compare, x="Módulo de Optimización", y="Sharpe Ratio",
                              color="Módulo de Optimización",
                              color_discrete_sequence=['#3D4F4A', '#6B4F3B', '#7C9473', '#B3452F'])
            fig_bar.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig_bar, use_container_width=True)

        with g4_col2:
            st.markdown("#### Evolución de Riqueza Normalizada (Backtesting Superpuesto)")
            fig_all = ob.Figure()
            fig_all.add_trace(ob.Scatter(x=mk["wealth_mkw"].index,
                                          y=mk["wealth_mkw"].values / mk["wealth_mkw"].values[0] * 100,
                                          name="Markowitz"))
            fig_all.add_trace(ob.Scatter(x=wealth_ns.index, y=wealth_ns.values / wealth_ns.values[0] * 100,
                                          name="NSGA-II"))
            fig_all.add_trace(ob.Scatter(x=list(range(len(dp["wealth_dp"]))),
                                          y=np.array(dp["wealth_dp"]) / dp["wealth_dp"][0] * 100,
                                          name="Prog. Dinámica"))
            fig_all.add_trace(ob.Scatter(x=mk["wealth_eq"].index,
                                          y=mk["wealth_eq"].values / mk["wealth_eq"].values[0] * 100,
                                          name="Equiponderado"))
            fig_all.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                   yaxis_title="Riqueza Normalizada (Base 100)",
                                   margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_all, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="framework-note">
            <strong>Nota metodológica:</strong> El módulo de Programación Dinámica opera sobre un estado escalar
            (exposición al portafolio tangente vs. efectivo) para mantener el espacio de estados tratable, por lo
            que sus métricas usan periodicidad mensual mientras que Markowitz, NSGA-II y Equiponderado usan
            periodicidad diaria. La tasa libre de riesgo se asume en 0% para el cálculo del Sharpe Ratio.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button("⚙️ Generar Executive Briefing Report (CSV)",
                            data=df_compare.to_csv(index=False).encode("utf-8"),
                            file_name="executive_briefing.csv", mime="text/csv",
                            key="btn_pdf_m4", type="primary")