# Mini Pac-Man RL — laboratorio interactivo

Proyecto didáctico en Python/Streamlit para comparar **Q-learning tabular** y **Deep Q-Network (DQN)** (red neuronal de dos capas).

## Ejecución

### Con Makefile (recomendado)

Requiere [uv](https://docs.astral.sh/uv/) instalado (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
make run     # crea/actualiza .venv con Python 3.12, instala deps y levanta Streamlit
```

Otros targets disponibles:

```bash
make setup   # solo crea el entorno e instala dependencias, sin levantar la app
make test    # smoke test: entrena unos pocos episodios de Q-learning y DQN sin UI
make test-unit  # tests unitarios con pytest (tests/)
make clean   # borra .venv y cachés de Python
```

La app queda disponible en `http://localhost:8501`.

### Manual con uv

```bash
uv sync                    # crea .venv con Python 3.12 e instala dependencias
uv run streamlit run app.py
```

### Alternativa con pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

En Windows, para activar el entorno virtual: `.venv\Scripts\activate`.

> Nota: `pyproject.toml`/`uv.lock` fijan `torch` a la build CPU-only (este demo
> entrena una red chica de 2 capas, no hace falta GPU). Si instalás con `pip`
> en Linux y querés evitar las dependencias CUDA de PyTorch, instalá antes el
> wheel CPU-only siguiendo https://pytorch.org/get-started/locally/.

## Experimento sugerido

1. Elegí **Q-learning**, dejá el episodio 0 y probá «Sin entrenar».
2. Entrená 100 episodios; observá cómo cambian recompensas, Q-valores y flechas.
3. Entrená 300-600 episodios adicionales y compará con la política sin entrenar.
4. Elegí **DQN (PyTorch)**; entrená de a 200 episodios e inspeccioná los pesos `W1`/`W2` a diferentes checkpoints. El aprendizaje de la red puede ser menos estable.
5. Compará la trayectoria resultante en la misma grilla.
6. Elegí un **patrón de fantasma** móvil, reiniciá y repetí: hacen falta más episodios (más estados), y con el selector de fase podés ver cómo la política cambia según dónde esté el fantasma (esperar, rodearlo, pasar detrás).
7. Cambiá **alpha** (Q-learning), **gamma**, el premio de la moneda, el castigo al fantasma o el shaping y reiniciá para repetir. La tasa de aprendizaje DQN es fija en esta primera demo para controlar variables.

## Entorno y recompensas

- Grilla 5×5, movimientos arriba/abajo/izquierda/derecha.
- Moneda `(4,4)` +10; fantasma -15, con final del episodio.
- Fantasma configurable (barra lateral → *Movimiento del fantasma*):
  - **Estático**: quieto en `(3,2)` (comportamiento original).
  - **Patrulla vertical**: sube y baja por la columna derecha `(1,4)→(2,4)→(3,4)→(2,4)`.
  - **Patrulla horizontal**: va y viene por la fila 3 `(3,0)→(3,1)→(3,2)→(3,1)`, que todo camino a la moneda debe cruzar.
  - **Ronda circular**: gira por el centro `(1,2)→(1,3)→(1,4)→(2,4)→(2,3)→(2,2)`.
  - **Personalizado**: escribís la ruta, p. ej. `2,2 2,3 2,4 3,4 2,4 2,3` (casillas vecinas, cíclica, máx. 12).
- El fantasma avanza una casilla de su ruta por cada paso del agente y repite el ciclo. El agente ve
  `(su casilla, fase del fantasma)`, por eso el patrón es **aprendible**: hay `25 × largo del ciclo` estados.
  Ver `docs/ghost-movement.md`.
- Cada paso -0.05; chocar con un muro/borde -0.55 extra.
- Acercarse a la moneda +0.10; alejarse -0.10 (reward shaping).
- Máximo 30 pasos por episodio.
- Los episodios de entrenamiento comienzan en posiciones (y fases del fantasma) aleatorias válidas; la partida visual y la evaluación arrancan con el fantasma en la fase `f0`.

## Qué significan los gráficos

- **Recompensa por episodio**: éxito temprano es mejor, pero hay ruido debido a exploración y distinta posición inicial.
- **Éxito (%)**: promedio móvil sobre las últimas 50 partidas de entrenamiento; los datos no corresponden al test.
- **Política aprendida**: la flecha indica `argmax_a Q(s,a)` para cada casilla.
- **Q-learning**: la tabla Q tiene `25 × 4` números; son valores aprendidos, no pesos de una red neuronal.
- **DQN**: `W1 (32 × 25)` y `W2 (4 × 32)` son matrices de pesos del modelo. Compará checkpoints para ver cómo cambian.
- **Evaluación**: una política congelada se prueba desde todas las posiciones iniciales válidas. No hay aleatoriedad del entorno en esas pruebas; es una prueba didáctica, no un benchmark real.

## Entender cómo aprende (y cómo se ajustan los pesos)

- **🔬 Aprender paso a paso**: juega un único movimiento de entrenamiento (con exploración ε) y muestra la cuenta completa:
  - *Q-learning*: objetivo `r + γ·max Q(s')`, error TD y el nuevo `Q(s,a)` con los números reales; solo cambia una celda de la tabla.
  - *DQN*: experiencia guardada en memoria, objetivo de la red objetivo, pérdida Huber del lote de 64, normas de gradiente y mapas **ΔW1 / ΔW2** de ese paso de gradiente.
  - Barras con los 4 valores Q de la casilla antes y después de la actualización.
- **Curvas de aprendizaje extra**: error de predicción (|error TD| medio o pérdida DQN), cuánto se movieron los parámetros en cada episodio (‖Δθ‖) y ε.
- **Mapa de valor V(s) = max Q(s,a)**: al mover el slider de checkpoints se ve cómo el valor de la moneda se propaga hacia atrás.
- **Mapa de visitas**: dónde estuvo el agente durante el entrenamiento (donde casi no pasó, lo aprendido es poco fiable).
- **Qué cambió desde el checkpoint anterior**: ΔQ o ΔW1/ΔW2 entre dos checkpoints consecutivos.
- **Cómo decide la red en una casilla** (DQN): activaciones ReLU de las 32 neuronas y el aporte `W2[a,j]·h[j]` de cada neurona a cada acción.
- **Durante la partida**: barras con los valores Q que explican cada movimiento.
- Todos los mapas de calor usan una escala divergente centrada en 0 (blanco = 0, azul = positivo, rojo = negativo).

## Notas metodológicas

- Este juego está inspirado en Pac-Man, **no es el entorno original**.
- El Q-learning y el DQN parten de un entorno conocido y pequeño. El estado es completamente observable, no hace falta LLM ni sensores reales.
- No usar información de evaluación para elegir una política y después presentarla como prueba independiente; esta demo es para aprender.
- La penalización por acercarse al fantasma, la recompensa y el shaping son decisiones de diseño. Probar otras funciones de recompensa cambia el comportamiento.
- Una buena clasificación de anomalías en datos etiquetados usualmente se resuelve más naturalmente con aprendizaje supervisado; RL es especialmente útil cuando la secuencia de decisiones y costos importan.
