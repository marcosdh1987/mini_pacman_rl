.PHONY: setup run test clean

# Crea/actualiza el entorno virtual (.venv) e instala dependencias con uv.
setup:
	uv sync

# Levanta la app de Streamlit (ejecuta `make setup` primero si hace falta).
run: setup
	uv run streamlit run app.py

# Corre una verificación rápida de humo: entrena unos pocos episodios de
# cada modo (Q-learning y DQN) sin levantar la UI, para validar que el
# entorno quedó bien instalado.
test: setup
	uv run python -c "\
import rl_core; \
t = rl_core.Trainer('Q-learning', seed=1); t.train(episodes=5); \
d = rl_core.Trainer('DQN (PyTorch)', seed=1); d.train(episodes=5); \
print('smoke test ok: Q-learning y DQN entrenaron sin errores')"

clean:
	rm -rf .venv __pycache__
