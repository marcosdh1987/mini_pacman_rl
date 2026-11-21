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
    evaluate_snapshot, from_id, snapshot_values, to_id,
)

st.set_page_config(page_title="Mini Pac-Man RL", page_icon="🟡", layout="wide")
st.title("🟡 Mini Pac-Man: laboratorio de Reinforcement Learning")
st.caption("Aprendé por recompensas, compará Q-learning con una red DQN y mirá cómo cambia la política durante el entrenamiento.")


def render_board(position=START, path=None, policy=None):
    path = path or []
    cells = []
    for r in range(SIZE):
        for c in range(SIZE):
            pos = (r, c)
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
            cells.append(
                f'<div style="display:flex;align-items:center;justify-content:center;'
                f'width:48px;height:48px;background:{bg};color:{fg};border-radius:7px;'
                f'font-size:24px;font-weight:700">{html.escape(symbol)}</div>'
            )
    return ('<div style="display:grid;grid-template-columns:repeat(5,48px);'
            'gap:4px;width:max-content;padding:10px;background:#0c1c2b;border-radius:12px">'
            + ''.join(cells) + '</div>')


def policy_map(snap):
    return {(r, c): ARROWS[choose_snapshot_action(snap, to_id((r, c)))]
            for r in range(SIZE) for c in range(SIZE) if (r, c) not in WALLS}


def heatmap(a, title, xlabels=None, ylabels=None):
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=110)
    im = ax.imshow(a, aspect="auto", cmap="coolwarm")
    ax.set_title(title, fontsize=11)
    if xlabels is not None:
        ax.set_xticks(range(len(xlabels)), xlabels)
    if ylabels is not None:
        ax.set_yticks(range(len(ylabels)), ylabels, fontsize=6)
    else:
        ax.set_ylabel("Neuronas (filas)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def make_play():
    return {"pos": START, "path": [START], "steps": 0, "reward": 0.0,
            "done": False, "status": "En curso", "last_action": "—"}


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

st.divider()
st.subheader("2. Ver aprendizaje y recompensas")
if trainer.history:
    df = pd.DataFrame(trainer.history).set_index("episode")
    df["recompensa_media_50"] = df["reward"].rolling(50, min_periods=1).mean()
    df["exito_medio_50"] = df["success"].rolling(50, min_periods=1).mean() * 100
    chart1, chart2 = st.columns(2)
    with chart1:
        st.line_chart(df[["reward", "recompensa_media_50"]], x_label="Episodios", y_label="Recompensa")
    with chart2:
        st.line_chart(df[["exito_medio_50"]], x_label="Episodios", y_label="Éxito móvil (%)")
    st.caption(f"Exploración ε actual: {trainer.epsilon():.3f}. La variación durante el entrenamiento es esperable: el agente todavía explora.")
    st.download_button("Exportar historial CSV", df.to_csv().encode("utf-8"),
                       file_name=f"historial_{'q' if mode == 'Q-learning' else 'dqn'}.csv",
                       mime="text/csv")
else:
    st.info("Todavía no entrenaste. Los valores Q comienzan en cero y los pesos DQN se inicializan al azar.")

st.divider()
st.subheader("3. Abrir un checkpoint y mirar la política")
checkpoints = trainer.checkpoints
choices = [point["episode"] for point in checkpoints]
selected_ep = st.select_slider("Episodio guardado", options=choices, value=choices[-1])
selected = next(x for x in checkpoints if x["episode"] == selected_ep)

mapcol, weightscol = st.columns([1, 2])
with mapcol:
    st.markdown("**Flechas: acción preferida en cada casilla**")
    st.markdown(render_board(position=(-1, -1), policy=policy_map(selected)), unsafe_allow_html=True)
    performance = evaluate_snapshot(selected)
    st.caption(f"Checkpoint {selected_ep} · {performance['wins']}/{performance['attempts']} inicios exitosos")
with weightscol:
    if mode == "Q-learning":
        st.markdown("**Tabla Q: 25 estados × 4 movimientos**")
        labels = [f"({r},{c})" for r in range(SIZE) for c in range(SIZE)]
        fig = heatmap(selected["q"], "Valores Q(s, acción)", ARROWS, labels)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Cada celda numérica almacena el valor de una acción desde un estado. No hay red neuronal.")
    else:
        st.markdown("**Pesos reales de la red neuronal DQN**")
        a, b = st.columns(2)
        with a:
            fig = heatmap(selected["w1"], "W1: 32 × 25")
            st.pyplot(fig)
            plt.close(fig)
        with b:
            fig = heatmap(selected["w2"], "W2: 4 × 32")
            st.pyplot(fig)
            plt.close(fig)
        init = checkpoints[0]
        diff1 = np.mean(np.abs(selected["w1"] - init["w1"]))
        diff2 = np.mean(np.abs(selected["w2"] - init["w2"]))
        st.caption(f"Cambio promedio absoluto desde episodio 0: W1 = {diff1:.4f}, W2 = {diff2:.4f}. Los colores muestran pesos, pero no son explicaciones directas de la decisión.")

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
    elif agent == "Sin entrenar":
        action = choose_snapshot_action(checkpoints[0], state)
    elif agent == "Checkpoint elegido":
        action = choose_snapshot_action(selected, state)
    else:
        action = choose_snapshot_action(checkpoints[-1], state)
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
    st.caption("El juego es completamente determinista. Compará la misma posición inicial con diferentes checkpoints. Las recompensas mostradas son las del entrenamiento activo.")
board.markdown(render_board(position=st.session_state.play["pos"],
                           path=st.session_state.play["path"]), unsafe_allow_html=True)

with st.expander("Qué está aprendiendo y qué cambiar después"):
    st.markdown("""
- **Q-learning** guarda un valor para cada par `(estado, acción)`. El mejor movimiento es el de mayor valor en esa fila.
- **DQN** aproxima esos valores con una red de `25 entradas → 32 neuronas → 4 salidas`. Aquí sí estás viendo matrices de pesos reales.
- La recompensa no es el peso: es una señal escalar que guía la actualización de la política.
- Probá primero 0, 100, 300 y 800 episodios. Reiniciá y cambiá gamma para observar cómo cambia el aprendizaje.
- Esta demo entrena con posiciones iniciales aleatorias y evita usar el resultado final como señal de entrenamiento.
- Para aproximarse a un sistema industrial, el siguiente escalón sería añadir estados que cambian en el tiempo, presupuesto de llamadas y recompensas diferidas.
""")

st.caption("Mini Pac-Man RL · Proyecto educativo autocontenido · sin LLM, sin APIs y sin control de equipos reales")
