# Mini Pac-Man RL — laboratorio interactivo

Proyecto didáctico en Python/Streamlit para comparar **Q-learning tabular** y **Deep Q-Network (DQN)** (red neuronal de dos capas).

## Ejecución

Con [uv](https://docs.astral.sh/uv/) (recomendado):

```bash
uv sync                    # crea .venv con Python 3.12 e instala dependencias
uv run streamlit run app.py
```

Alternativa con pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

En Windows, para activar el entorno virtual: `.venv\Scripts\activate`.

## Experimento sugerido

1. Elegí **Q-learning**, dejá el episodio 0 y probá «Sin entrenar».
2. Entrená 100 episodios; observá cómo cambian recompensas, Q-valores y flechas.
3. Entrená 300-600 episodios adicionales y compará con la política sin entrenar.
4. Elegí **DQN (PyTorch)**; entrená de a 200 episodios e inspeccioná los pesos `W1`/`W2` a diferentes checkpoints. El aprendizaje de la red puede ser menos estable.
5. Compará la trayectoria resultante en la misma grilla.
6. Cambiá **alpha** (Q-learning), **gamma**, el premio de la moneda, el castigo al fantasma o el shaping y reiniciá para repetir. La tasa de aprendizaje DQN es fija en esta primera demo para controlar variables.

## Entorno y recompensas

- Grilla 5×5, movimientos arriba/abajo/izquierda/derecha.
- Moneda `(4,4)` +10; fantasma `(3,2)` -15, con final del episodio.
- Cada paso -0.05; chocar con un muro/borde -0.55 extra.
- Acercarse a la moneda +0.10; alejarse -0.10 (reward shaping).
- Máximo 30 pasos por episodio.
- Los episodios de entrenamiento comienzan en posiciones aleatorias válidas; la partida visual inicia en `(0,0)`.

## Qué significan los gráficos

- **Recompensa por episodio**: éxito temprano es mejor, pero hay ruido debido a exploración y distinta posición inicial.
- **Éxito (%)**: promedio móvil sobre las últimas 50 partidas de entrenamiento; los datos no corresponden al test.
- **Política aprendida**: la flecha indica `argmax_a Q(s,a)` para cada casilla.
- **Q-learning**: la tabla Q tiene `25 × 4` números; son valores aprendidos, no pesos de una red neuronal.
- **DQN**: `W1 (32 × 25)` y `W2 (4 × 32)` son matrices de pesos del modelo. Compará checkpoints para ver cómo cambian.
- **Evaluación**: una política congelada se prueba desde todas las posiciones iniciales válidas. No hay aleatoriedad del entorno en esas pruebas; es una prueba didáctica, no un benchmark real.

## Notas metodológicas

- Este juego está inspirado en Pac-Man, **no es el entorno original**.
- El Q-learning y el DQN parten de un entorno conocido y pequeño. El estado es completamente observable, no hace falta LLM ni sensores reales.
- No usar información de evaluación para elegir una política y después presentarla como prueba independiente; esta demo es para aprender.
- La penalización por acercarse al fantasma, la recompensa y el shaping son decisiones de diseño. Probar otras funciones de recompensa cambia el comportamiento.
- Una buena clasificación de anomalías en datos etiquetados usualmente se resuelve más naturalmente con aprendizaje supervisado; RL es especialmente útil cuando la secuencia de decisiones y costos importan.
