"""Run with: streamlit run app.py"""
from __future__ import annotations
import html
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from rl_core import (
    ARROWS, GOAL, GHOST, HORIZON, SIZE, START, WALLS,
    DEFAULT_REWARDS, Maze, Trainer, choose_snapshot_action, evaluate_random,
    evaluate_snapshot, forward_details, from_id, snapshot, snapshot_values, to_id,
)
from matplotlib import colormaps
from matplotlib.colors import TwoSlopeNorm

st.set_page_config(page_title="Mini Pac-Man RL", page_icon="🟡", layout="wide")
st.title("🟡 Mini Pac-Man: laboratorio de Reinforcement Learning")
st.caption("Aprendé por recompensas, compará Q-learning con una red DQN y mirá cómo cambia la política durante el entrenamiento.")


def _value_color(v, vmax):
    """Diverging colour for a value: blue = good (positive), red = bad, grey ≈ 0."""
    rgba = colormaps["RdBu"](0.5 + 0.5 * float(np.clip(v / vmax, -1, 1)))
    r, g, b = (int(255 * x) for x in rgba[:3])
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return f"rgb({r},{g},{b})", ("#10202e" if lum > 150 else "white")


def render_board(position=START, path=None, policy=None, values=None, focus=None, size=48, show_numbers=True):
    """HTML grid. ``values`` (dict pos->float) colours free cells by value and
    prints the number; ``focus`` outlines one cell (the state being updated)."""
    path = path or []
    vmax = max([abs(v) for v in values.values()] + [1e-6]) if values else 1.0
    cells = []
    for r in range(SIZE):
        for c in range(SIZE):
            pos = (r, c)
            sub = ""
            if pos == position:
                symbol, bg, fg = "🟡", "#205975", "white"
            elif pos == GOAL:
                symbol, bg, fg = "🟨", "#e5b72b", "#172330"
            elif pos == GHOST:
                symbol, bg, fg = "👻", "#9d3a4d", "white"
            elif pos in WALLS:
                symbol, bg, fg = "", "#536172", "white"
            else:
                symbol = policy[pos] if policy else "·"
                bg = "#246153" if pos in path else "#1b3047"
                fg = "#d3eff0"
                if values is not None and pos in values:
                    bg, fg = _value_color(values[pos], vmax)
                    sub = f"{values[pos]:+.1f}" if show_numbers else ""
            border = "outline:3px solid #ffd84d;outline-offset:-3px;" if pos == focus else ""
            label = (f'<div style="line-height:1">{html.escape(symbol)}'
                     + (f'<div style="font-size:10px;font-weight:500">{sub}</div>' if sub else "")
                     + "</div>")
            cells.append(
                f'<div style="display:flex;align-items:center;justify-content:center;text-align:center;'
                f'width:{size}px;height:{size}px;background:{bg};color:{fg};border-radius:7px;{border}'
                f'font-size:{int(size / 2)}px;font-weight:700">{label}</div>'
            )
    return (f'<div style="display:grid;grid-template-columns:repeat(5,{size}px);'
            'gap:4px;width:max-content;padding:10px;background:#0c1c2b;border-radius:12px">'
            + ''.join(cells) + '</div>')


def policy_map(snap):
    return {(r, c): ARROWS[choose_snapshot_action(snap, to_id((r, c)))]
            for r in range(SIZE) for c in range(SIZE) if (r, c) not in WALLS}


def value_map(snap):
    """V(s) = max_a Q(s, a): how good the agent believes each cell is."""
    return {(r, c): float(np.max(snapshot_values(snap, to_id((r, c)))))
            for r in range(SIZE) for c in range(SIZE)
            if (r, c) not in WALLS and (r, c) not in (GOAL, GHOST)}


STATE_LABELS = [f"({r},{c})" for r in range(SIZE) for c in range(SIZE)]


def heatmap(a, title, xlabels=None, ylabels=None, xlabel=None, ylabel=None, figsize=(7.2, 3.6)):
    """Diverging heatmap centred on 0: white = 0, blue > 0, red < 0.
    (Centring matters: otherwise 'white' would not mean 'zero weight'.)"""
    a = np.asarray(a, dtype=float)
    vmax = float(np.max(np.abs(a))) or 1e-6
    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    im = ax.imshow(a, aspect="auto", cmap="RdBu", norm=TwoSlopeNorm(0.0, -vmax, vmax))
    ax.set_title(title, fontsize=11)
    if xlabels is not None:
        ax.set_xticks(range(len(xlabels)), xlabels, fontsize=7 if len(xlabels) > 8 else 10,
                      rotation=90 if len(xlabels) > 8 else 0)
    if ylabels is not None:
        ax.set_yticks(range(len(ylabels)), ylabels, fontsize=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def q_bars(before, after=None, chosen=None):
    """Small table-backed bar chart with the 4 action values of one state."""
    data = {"antes": before} if after is not None else {"Q(s, acción)": before}
    if after is not None:
        data["después"] = after
    df = pd.DataFrame(data, index=[f"{i} {ARROWS[i]}" + (" ◀ elegida" if i == chosen else "")
                                   for i in range(4)])
    st.bar_chart(df, stack=False, horizontal=True, height=180)


def make_play():
    return {"pos": START, "path": [START], "steps": 0, "reward": 0.0,
            "done": False, "status": "En curso", "last_action": "—", "last_values": None}


# Distinct learners retain state while switching methods.
if "trainers" not in st.session_state:
    st.session_state.trainers = {}
if "play_rng" not in st.session_state:
    st.session_state.play_rng = random.Random(123)

with st.sidebar:
    st.header("Configuración")
    mode = st.radio("Método", ["Q-learning", "DQN (PyTorch)"],
                    help="Q-learning guarda 25×4 valores; DQN aprende matrices de pesos neuronales.")
    seed = st.number_input("Semilla", min_value=0, max_value=100000, value=42, step=1)
    batch = st.select_slider("Episodios por entrenamiento", options=[25, 50, 100, 200, 300, 600], value=200)
    alpha = st.slider("Alpha · tasa de aprendizaje Q", 0.05, 0.8, 0.2, 0.05,
                      disabled=mode != "Q-learning")
    gamma = st.slider("Gamma · importancia del futuro", 0.5, 0.99, 0.95, 0.01)
    st.caption("Exploración ε: empieza en 0,95 y disminuye hacia 0,04 con los episodios. DQN usa Adam (lr=0,002).")
    st.markdown("**Experimentos con recompensas**")
    goal_reward = st.slider("Premio por moneda", 5, 20, 10)
    ghost_penalty = st.slider("Castigo por fantasma", -30, -3, -15)
    progress_reward = st.slider("Premio por acercarse", 0.0, 0.5, 0.10, 0.05)
    configured_rewards = {"goal": float(goal_reward), "ghost": float(ghost_penalty),
                          "step": -0.05, "wall": -0.55, "progress": float(progress_reward)}
    if st.button("🔁 Reiniciar entrenamiento", width="stretch"):
        st.session_state.trainers[mode] = Trainer(mode, int(seed), configured_rewards)
        st.session_state.play_key = None
        st.session_state.pop(f"trace_{mode}", None)
        st.rerun()

if mode not in st.session_state.trainers:
    st.session_state.trainers[mode] = Trainer(mode, int(seed), configured_rewards)
trainer = st.session_state.trainers[mode]
if configured_rewards != trainer.rewards or int(seed) != trainer.seed:
    st.warning("Cambiaste la semilla o las recompensas. Pulsá «Reiniciar entrenamiento» para aplicarlas limpiamente.")

col_train, col_info = st.columns([1, 2])
with col_train:
    st.subheader("1. Entrenar")
    st.metric("Episodios acumulados", trainer.episode)
    if st.button(f"▶ Entrenar {batch} episodios", type="primary", width="stretch"):
        with st.spinner("Entrenando la política..."):
            trainer.train(episodes=batch, alpha=alpha, gamma=gamma)
        st.session_state.pop(f"trace_{mode}", None)
        st.success(f"Se agregaron {batch} episodios.")
        st.rerun()
with col_info:
    st.subheader("Resultados evaluados sin exploración")
    initial = evaluate_snapshot(trainer.checkpoints[0])
    current = evaluate_snapshot(trainer.checkpoints[-1])
    rand = evaluate_random()
    m1, m2, m3 = st.columns(3)
    m1.metric("Aleatorio", f"{rand:.0%}", help="300 partidas, inicios al azar")
    m2.metric("Sin entrenar", f"{initial['success_rate']:.0%}", help="Política del episodio 0")
    m3.metric("Entrenado", f"{current['success_rate']:.0%}",
              delta=f"{(current['success_rate'] - initial['success_rate'])*100:+.0f} pp",
              help=f"{current['wins']} de {current['attempts']} posiciones iniciales válidas")
    st.caption("Éxito al alcanzar la moneda en ≤30 pasos desde 20 posiciones iniciales. El entorno es fijo; los porcentajes son didácticos, no significancia estadística.")

# ---------------------------------------------------------------------------
st.divider()
st.subheader("🔬 Aprender paso a paso: mirá una actualización por dentro")
st.caption("Cada clic juega **un movimiento de entrenamiento** (con exploración ε) y aplica "
           "exactamente la misma actualización que el entrenamiento por lotes. Así se ve qué número "
           "cambia, por qué y cuánto.")
trace_key = f"trace_{mode}"
s1, s2, s3 = st.columns(3)
with s1:
    if st.button("👣 Dar 1 paso de aprendizaje", width="stretch"):
        st.session_state[trace_key] = trainer.learn_step(alpha, gamma, detail=True)
with s2:
    if st.button("⏩ 10 pasos", width="stretch"):
        for _ in range(10):
            st.session_state[trace_key] = trainer.learn_step(alpha, gamma, detail=True)
with s3:
    if st.button("⏭ Terminar episodio en curso", width="stretch",
                 disabled=trainer.live is None):
        while trainer.live is not None:
            st.session_state[trace_key] = trainer.learn_step(alpha, gamma, detail=True)
        if trainer.checkpoints[-1]["episode"] != trainer.episode:
            trainer.checkpoints.append(snapshot(trainer))

tr = st.session_state.get(trace_key)
if tr is None:
    st.info("Pulsá «Dar 1 paso de aprendizaje» para ver la primera actualización. "
            "Tip: probalo con el agente sin entrenar y después de 200-300 episodios.")
else:
    s, a, sp = from_id(tr["state"]), tr["action"], from_id(tr["next_state"])
    tcol, ecol = st.columns([1, 2])
    with tcol:
        st.markdown(render_board(position=sp, path=[from_id(x) for x in tr["path"]], focus=s),
                    unsafe_allow_html=True)
        st.caption("Recuadro amarillo: casilla cuyo valor se actualiza. Verde: recorrido de este episodio.")
    with ecol:
        kind = ("🎲 **exploró** (movimiento al azar)" if tr["explored"]
                else "🧠 **explotó** (eligió la acción de mayor Q)")
        st.markdown(f"**Paso {tr['episode_step']} del episodio {trainer.episode + (0 if tr.get('episode_finished') else 1)}** · "
                    f"desde {s} movió **{ARROWS[a]}** y llegó a {sp} · {kind} (ε = {tr['eps']:.2f})")
        outcome = ("🏆 moneda" if tr["info"]["success"] else "👻 fantasma" if tr["info"]["ghost"]
                   else "⌛ fin por límite de pasos" if tr["done"] else "sigue")
        st.markdown(f"Recompensa inmediata **r = {tr['reward']:+.2f}** · resultado: {outcome} · "
                    f"recompensa acumulada del episodio: {tr['episode_total']:+.2f}")
        if tr["mode"] == "Q-learning":
            fut = "0 (terminó el episodio)" if tr["done"] else f"{tr['gamma']:.2f} × {tr['next_max']:.2f}"
            st.markdown("**Regla de Q-learning (Bellman)**")
            st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha\,\big[\underbrace{r + \gamma \max_{a'} Q(s',a')}_{\text{objetivo}} - Q(s,a)\big]")
            st.markdown(
                f"- Objetivo = r + γ·max Q(s') = {tr['reward']:+.2f} + {fut} = **{tr['target']:.3f}**\n"
                f"- Error TD = objetivo − Q viejo = {tr['target']:.3f} − {tr['old']:.3f} = **{tr['td']:+.3f}**\n"
                f"- Q nuevo = {tr['old']:.3f} + {tr['alpha']:.2f} × {tr['td']:+.3f} = **{tr['new']:.3f}**")
            st.caption("Solo cambia **un número** de la tabla: Q(s, a). El error TD dice si el movimiento "
                       "resultó mejor (+) o peor (−) de lo que el agente esperaba; α decide cuánto corregir.")
        else:
            upd = tr["update"]
            fut = "0 (terminó)" if tr["done"] else f"{tr['gamma']:.2f} × {tr['next_max']:.2f}"
            st.markdown(f"**1. Guardar la experiencia** (s, a, r, s') en la memoria de repetición "
                        f"→ {tr['memory']} transiciones.")
            st.markdown(f"**2. Objetivo para esta transición** (red objetivo congelada): "
                        f"r + γ·max Q̂(s') = {tr['reward']:+.2f} + {fut} = **{tr['target']:.3f}** "
                        f"· la red predecía Q(s,a) = {tr['old']:.3f}")
            if upd is None:
                st.warning(f"Todavía no hay 64 experiencias guardadas ({tr['memory']}/64): la red no se ajusta aún.")
            else:
                st.markdown(f"**3. Paso de gradiente** sobre un lote al azar de 64 recuerdos "
                            f"(no sobre este paso en particular): pérdida Huber = **{upd['loss']:.4f}** · "
                            f"‖∇W1‖ = {upd['grad_w1']:.4f} · ‖∇W2‖ = {upd['grad_w2']:.4f}")
                st.markdown(f"**4. Efecto:** Q(s,a) pasó de {tr['old']:.3f} a **{tr['new']:.3f}**. "
                            "Como los pesos son compartidos, también cambian los valores de otras casillas.")
                if upd["target_sync"]:
                    st.success("🔄 En este paso se copiaron los pesos a la red objetivo (cada 125 actualizaciones).")
        q_bars(tr["values_before"], tr["values_after"], chosen=a)
        st.caption("Valores Q de la casilla de partida antes y después de esta actualización.")
    if tr["mode"] != "Q-learning" and tr.get("update"):
        upd = tr["update"]
        st.markdown("**¿Qué pesos se movieron en este paso de gradiente?**")
        d1, d2 = st.columns(2)
        with d1:
            show_fig(heatmap(upd["delta_w1"], "ΔW1 en este paso (32 neuronas × 25 casillas)",
                             xlabels=STATE_LABELS, ylabel="Neurona oculta", xlabel="Casilla de entrada"))
        with d2:
            show_fig(heatmap(upd["delta_w2"], "ΔW2 en este paso (4 acciones × 32 neuronas)",
                             ylabels=ARROWS, xlabel="Neurona oculta"))
        batch_cells = ", ".join(STATE_LABELS[i] for i in upd["batch_states"])
        st.caption("Como la entrada es one-hot, el gradiente de W1 solo existe en las **columnas de las casillas "
                   f"presentes en el lote** ({batch_cells}) y, en W2, solo en las neuronas activas (ReLU > 0). "
                   "Los cambios tenues en otras columnas vienen del *momentum* de Adam (arrastra la dirección de "
                   "pasos anteriores). Las columnas de muros nunca cambian: el agente nunca está ahí.")

# ---------------------------------------------------------------------------
st.divider()
st.subheader("2. Ver aprendizaje y recompensas")
if trainer.history:
    df = pd.DataFrame(trainer.history).set_index("episode")
    df["recompensa_media_50"] = df["reward"].rolling(50, min_periods=1).mean()
    df["exito_medio_50"] = df["success"].rolling(50, min_periods=1).mean() * 100
    df["error_medio_50"] = df["error"].rolling(50, min_periods=1).mean()
    chart1, chart2 = st.columns(2)
    with chart1:
        st.markdown("**Recompensa por episodio**")
        st.line_chart(df[["reward", "recompensa_media_50"]], x_label="Episodios", y_label="Recompensa", height=250)
    with chart2:
        st.markdown("**Éxito en entrenamiento (media móvil 50)**")
        st.line_chart(df[["exito_medio_50"]], x_label="Episodios", y_label="Éxito móvil (%)", height=250)
    chart3, chart4, chart5 = st.columns(3)
    err_name = "|error TD| medio" if mode == "Q-learning" else "pérdida media (Huber)"
    with chart3:
        st.markdown(f"**Error de predicción: {err_name}**")
        st.line_chart(df[["error_medio_50"]], x_label="Episodios", y_label=err_name, height=200)
        st.caption("Qué tan equivocadas estaban las predicciones del agente. Es la señal que mueve los pesos: "
                   "si baja, el agente ya «se sorprende» menos.")
    with chart4:
        st.markdown("**Cuánto se movieron los pesos por episodio**")
        st.line_chart(df[["weight_change"]], x_label="Episodios", y_label="‖Δ parámetros‖", height=200)
        st.caption("Norma L2 del cambio de todos los números aprendibles en cada episodio. "
                   "Grande al principio; tiende a estabilizarse cuando la política converge.")
    with chart5:
        st.markdown("**Exploración ε**")
        st.line_chart(df[["epsilon"]], x_label="Episodios", y_label="ε", height=200)
        st.caption("Probabilidad de mover al azar. Mucha exploración al inicio para descubrir la moneda; "
                   "luego el agente explota lo aprendido.")
    st.download_button("Exportar historial CSV", df.to_csv().encode("utf-8"),
                       file_name=f"historial_{'q' if mode == 'Q-learning' else 'dqn'}.csv",
                       mime="text/csv")
else:
    st.info("Todavía no entrenaste. Los valores Q comienzan en cero y los pesos DQN se inicializan al azar.")

# ---------------------------------------------------------------------------
st.divider()
st.subheader("3. Abrir un checkpoint y mirar la política")
checkpoints = trainer.checkpoints
choices = [point["episode"] for point in checkpoints]
selected_ep = st.select_slider("Episodio guardado", options=choices, value=choices[-1])
sel_index = choices.index(selected_ep)
selected = checkpoints[sel_index]
previous = checkpoints[sel_index - 1] if sel_index > 0 else None

mapcol, valcol, visitcol = st.columns(3)
with mapcol:
    st.markdown("**Política: acción preferida**")
    st.markdown(render_board(position=(-1, -1), policy=policy_map(selected)), unsafe_allow_html=True)
    performance = evaluate_snapshot(selected)
    st.caption(f"Checkpoint {selected_ep} · {performance['wins']}/{performance['attempts']} inicios exitosos")
with valcol:
    st.markdown("**Valor de cada casilla V(s) = max Q(s, a)**")
    st.markdown(render_board(position=(-1, -1), policy=policy_map(selected), values=value_map(selected)),
                unsafe_allow_html=True)
    st.caption("Azul = el agente espera ganar desde ahí; rojo = espera perder. Moviendo el slider "
               "se ve cómo el valor de la moneda **se propaga hacia atrás** casilla por casilla.")
with visitcol:
    st.markdown("**Visitas durante el entrenamiento**")
    visits = selected["visits"]
    vis = {(r, c): float(visits[to_id((r, c))]) for r in range(SIZE) for c in range(SIZE)
           if (r, c) not in WALLS and (r, c) not in (GOAL, GHOST)}
    vmax_vis = max(vis.values()) or 1.0
    st.markdown(render_board(position=(-1, -1),
                             policy={p: f"{int(v)}" for p, v in vis.items()},
                             values={p: v / vmax_vis for p, v in vis.items()}, show_numbers=False),
                unsafe_allow_html=True)
    st.caption("Cuántas veces salió el agente desde cada casilla. Donde casi no pasó, sus valores "
               "están poco entrenados (y las flechas pueden ser poco fiables).")

if mode == "Q-learning":
    tab_now, tab_delta = st.tabs(["Tabla Q del checkpoint", "Qué cambió desde el checkpoint anterior"])
    with tab_now:
        show_fig(heatmap(selected["q"].T, "Valores Q(s, acción)", xlabels=STATE_LABELS, ylabels=ARROWS,
                         xlabel="Casilla (estado)", figsize=(10, 2.6)))
        st.caption("25 estados × 4 movimientos. Cada celda es un número independiente: no hay red neuronal. "
                   "Blanco = 0 (todavía sin aprender), azul = positivo, rojo = negativo.")
    with tab_delta:
        if previous is None:
            st.info("El episodio 0 es el punto de partida: todos los valores Q valen 0.")
        else:
            show_fig(heatmap(selected["q"].T - previous["q"].T,
                             f"ΔQ entre episodio {previous['episode']} y {selected_ep}",
                             xlabels=STATE_LABELS, ylabels=ARROWS, xlabel="Casilla (estado)", figsize=(10, 2.6)))
            st.caption("Solo cambian las celdas (estado, acción) que el agente probó en esos episodios.")
else:
    tab_now, tab_delta, tab_fwd = st.tabs(["Pesos del checkpoint", "Qué cambió desde el checkpoint anterior",
                                           "Cómo decide la red en una casilla"])
    with tab_now:
        a1, b1 = st.columns(2)
        with a1:
            show_fig(heatmap(selected["w1"], "W1: 32 neuronas × 25 casillas",
                             xlabels=STATE_LABELS, ylabel="Neurona oculta", xlabel="Casilla de entrada"))
        with b1:
            show_fig(heatmap(selected["w2"], "W2: 4 acciones × 32 neuronas", ylabels=ARROWS,
                             xlabel="Neurona oculta"))
        init = checkpoints[0]
        diff1 = np.mean(np.abs(selected["w1"] - init["w1"]))
        diff2 = np.mean(np.abs(selected["w2"] - init["w2"]))
        st.caption(f"Cambio promedio absoluto desde episodio 0: W1 = {diff1:.4f}, W2 = {diff2:.4f}. "
                   "Columna j de W1 = cómo «ve» la red la casilla j. Fila a de W2 = qué neuronas empujan la acción a.")
    with tab_delta:
        if previous is None:
            st.info("El episodio 0 tiene pesos aleatorios iniciales. Elegí un checkpoint posterior.")
        else:
            a2, b2 = st.columns(2)
            with a2:
                show_fig(heatmap(selected["w1"] - previous["w1"],
                                 f"ΔW1 (episodio {previous['episode']} → {selected_ep})",
                                 xlabels=STATE_LABELS, ylabel="Neurona oculta", xlabel="Casilla de entrada"))
            with b2:
                show_fig(heatmap(selected["w2"] - previous["w2"],
                                 f"ΔW2 (episodio {previous['episode']} → {selected_ep})",
                                 ylabels=ARROWS, xlabel="Neurona oculta"))
            col_change = np.abs(selected["w1"] - previous["w1"]).sum(axis=0)
            top = np.argsort(col_change)[::-1][:3]
            st.caption("Casillas cuyos pesos de entrada más cambiaron: "
                       + ", ".join(STATE_LABELS[i] for i in top)
                       + ". Las columnas de muros nunca cambian porque el agente nunca puede estar ahí.")
    with tab_fwd:
        free = [(r, c) for r in range(SIZE) for c in range(SIZE) if (r, c) not in WALLS]
        cell = st.selectbox("Casilla", free, format_func=lambda p: f"{p}", index=0)
        fd = forward_details(selected, to_id(cell))
        active = int((fd["hidden"] > 0).sum())
        st.markdown(f"1. **Entrada one-hot**: solo la casilla {cell} vale 1, así que la red toma la columna "
                    f"{to_id(cell)} de W1 (+ sesgo).\n"
                    f"2. **ReLU**: {active} de 32 neuronas quedan activas (>0); las demás se apagan y no influyen.\n"
                    f"3. **Salida**: Q(s,a) = Σ W2[a,j]·h[j] + b2[a]. Abajo, el aporte de cada neurona a cada acción.")
        f1, f2 = st.columns([1, 2])
        with f1:
            hid = pd.DataFrame({"activación h": fd["hidden"]}, index=[f"n{j:02d}" for j in range(32)])
            st.bar_chart(hid, height=260)
        with f2:
            show_fig(heatmap(fd["contrib"], "Aporte W2[a,j]·h[j] de cada neurona a cada acción",
                             ylabels=ARROWS, xlabel="Neurona oculta", figsize=(7.2, 2.6)))
        st.dataframe(pd.DataFrame({"acción": ARROWS, "Q(s, acción)": fd["q"],
                                   "elegida": ["◀" if i == int(np.argmax(fd["q"])) else "" for i in range(4)]}),
                     hide_index=True, width="stretch")
        st.caption("Las neuronas son compartidas entre casillas: ajustar una para mejorar una casilla "
                   "también cambia otras. Eso permite generalizar, pero también hace al DQN menos estable.")

with st.expander("Inspeccionar las cuatro acciones de una casilla"):
    row = st.select_slider("Fila", options=list(range(5)), value=0)
    col = st.select_slider("Columna", options=list(range(5)), value=0)
    values = snapshot_values(selected, to_id((row, col)))
    st.dataframe(pd.DataFrame({"acción": ARROWS, "Q(s, acción)": values}),
                 hide_index=True, width="stretch")

st.divider()
st.subheader("4. Ver cómo juega: sin entrenar vs. entrenado")
agent = st.radio("Política de la partida", ["Aleatorio", "Sin entrenar", "Checkpoint elegido", "Entrenado actual"],
                 horizontal=True)
play_key = (mode, agent, selected_ep if agent == "Checkpoint elegido" else None)
if st.session_state.get("play_key") != play_key:
    st.session_state.play_key = play_key
    st.session_state.play = make_play()


def play_one():
    p = st.session_state.play
    if p["done"]:
        return
    env = Maze(rewards=trainer.rewards)
    env.pos = p["pos"]
    env.steps = p["steps"]
    state = to_id(env.pos)
    if agent == "Aleatorio":
        action = st.session_state.play_rng.randrange(4)
        p["last_values"] = None
    else:
        snap = {"Sin entrenar": checkpoints[0], "Checkpoint elegido": selected}.get(agent, checkpoints[-1])
        p["last_values"] = snapshot_values(snap, state)
        action = int(np.argmax(p["last_values"]))
    _, reward, done, info = env.step(action)
    p["pos"] = env.pos
    p["steps"] = env.steps
    p["reward"] += reward
    p["path"].append(env.pos)
    p["last_action"] = ARROWS[action]
    p["done"] = done
    if info["success"]:
        p["status"] = "🏆 Llegó a la moneda"
    elif info["ghost"]:
        p["status"] = "👻 Chocó con el fantasma"
    elif done:
        p["status"] = "⌛ Se quedó sin pasos"

boardcol, controlscol = st.columns([1, 2])
with boardcol:
    board = st.empty()
with controlscol:
    p = st.session_state.play
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Un paso", disabled=p["done"], width="stretch"):
            play_one()
    with b2:
        auto = st.button("Jugar solo", disabled=p["done"], width="stretch")
    with b3:
        if st.button("Reiniciar partida", width="stretch"):
            st.session_state.play = make_play()
            st.rerun()
    if auto:
        while not st.session_state.play["done"]:
            play_one()
            board.markdown(render_board(position=st.session_state.play["pos"],
                                       path=st.session_state.play["path"]),
                           unsafe_allow_html=True)
            time.sleep(0.11)
    p = st.session_state.play
    st.metric("Recompensa de esta partida", f"{p['reward']:.2f}")
    st.markdown(f"**Estado:** {p['status']}  |  **Pasos:** {p['steps']}/{HORIZON}  |  **Última acción:** {p['last_action']}")
    if p.get("last_values") is not None:
        q_bars(p["last_values"], chosen=ARROWS.index(p["last_action"]))
        st.caption("Por qué eligió ese movimiento: valores Q de la casilla desde la que salió. "
                   "Sin exploración, siempre toma la barra más larga.")
    st.caption("El juego es completamente determinista. Compará la misma posición inicial con diferentes checkpoints. Las recompensas mostradas son las del entrenamiento activo.")
board.markdown(render_board(position=st.session_state.play["pos"],
                           path=st.session_state.play["path"]), unsafe_allow_html=True)

with st.expander("Qué está aprendiendo y qué cambiar después"):
    st.markdown("""
- **Q-learning** guarda un valor para cada par `(estado, acción)`. El mejor movimiento es el de mayor valor en esa fila.
- **DQN** aproxima esos valores con una red de `25 entradas → 32 neuronas → 4 salidas`. Aquí sí estás viendo matrices de pesos reales.
- **Cómo se ajustan los números**: en ambos métodos, el agente compara lo que predecía con lo que pasó (`r + γ·max Q(s')`). Esa diferencia (error TD) es la que mueve el valor Q (Q-learning) o, vía gradiente, los pesos (DQN). Usá «Aprender paso a paso» para verlo con números.
- **Propagación del valor**: al principio solo la casilla junto a la moneda aprende que es buena; episodio a episodio, ese valor se transmite hacia atrás (mirá el mapa V(s) en distintos checkpoints).
- La recompensa no es el peso: es una señal escalar que guía la actualización de la política.
- Probá primero 0, 100, 300 y 800 episodios. Reiniciá y cambiá gamma para observar cómo cambia el aprendizaje.
- Esta demo entrena con posiciones iniciales aleatorias y evita usar el resultado final como señal de entrenamiento.
- Para aproximarse a un sistema industrial, el siguiente escalón sería añadir estados que cambian en el tiempo, presupuesto de llamadas y recompensas diferidas.
""")

st.caption("Mini Pac-Man RL · Proyecto educativo autocontenido · sin LLM, sin APIs y sin control de equipos reales")
