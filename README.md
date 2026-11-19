# Examen 2 - Modelos Matemáticos y Simulación

**Visión artificial (Ultralytics YOLO) para detectar derrames de líquido y alimentar un modelo matemático de seguridad.**

Carrera de Ingeniería de Software · Universidad Iberoamericana del Ecuador (UNIBE) · Docente: Ing. Jorge Morales, MSc.

> Integrantes: Helton Quezada, Steven Chinchin

## 1. Idea del proyecto

Un derrame de líquido en un piso es un riesgo (resbalones, daño a equipos). Se entrena un detector YOLO que localiza el líquido en video, y de cada cuadro se extrae una **variable cuantitativa**: la fracción del cuadro cubierta por líquido, `A(t)`. Esa serie alimenta dos modelos matemáticos ejecutados en Python:

| Modelo | Qué representa |
|---|---|
| **EDO por fases** | Fase 1 (derrame): `dA/dt = q - k1·A`. Fase 2 (limpieza/secado): `dA/dt = -k2·A`. |
| **Cadena de Markov** | Estados Seguro (A < 5 %), Precaución (5–25 %), Crítico (> 25 %). Matriz de transición estimada del video, distribución estacionaria y tiempo medio hasta Crítico. |

Sobre la cadena se simulan (Monte Carlo) escenarios de fuga mayor y de limpieza más rápida.

## 2. Dataset

- **cls-liquid** (Ultralytics Platform): charcos, fugas y derrames en interiores y pavimento. Detección con bounding boxes, **1 clase** (`liquid`).
- 803 imágenes de 800×800 y 1.775 cajas. El dataset original trae train/val; se separó un 10 % como **test**: **643 train / 80 val / 80 test**.
- El archivo `cls-liquid.ndjson` está en el repositorio; `prepare` descarga las imágenes y crea el formato YOLO. Las imágenes no se suben al repositorio.

## 3. Entrenamiento (Fase II)

- Modelo: `yolo11n.pt` (Ultralytics 8.4), GPU NVIDIA RTX 5060 Laptop (8 GB).
- Hiperparámetros: `epochs=50`, `imgsz=640`, `batch=16`, `lr0=0.01`, `patience=15`, `seed=42`. Se eligió la variante *n* y 640 px por los 8 GB de VRAM; `patience` corta el entrenamiento si deja de mejorar.
- Curvas de entrenamiento (loss, precisión, recall, mAP) en `entrenamiento/`.

**Métricas en el conjunto de test (80 imágenes):**

| Precisión | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|
| 0.876 | 0.856 | 0.929 | 0.750 |

Pesos entrenados: `resultados/best.pt`.

## 4. Resultados de la simulación (Fases III y IV)

Video de prueba procesado a 5 muestras/s con media móvil de 5 puntos (119 muestras). Área media detectada: 7,7 %.

- **EDO fase 1:** crecimiento casi lineal, `dA/dt ≈ 0.122` (la fase dura ~3,6 s, `k1` no es identificable). **Fase 2:** `dA/dt = -0.418·A`.
- **Markov:** distribución estacionaria Seguro 61 % / Precaución 25 % / Crítico 14 %. Tiempo medio desde Seguro hasta Crítico: 8,9 s.

| Escenario (horizonte 15 s) | P(llegar a Crítico) | Tiempo medio (s) |
|---|---|---|
| Base | 0.832 | 5.99 |
| Fuga ×1.5 | 0.958 | 4.62 |
| Fuga ×2 | 0.988 | 3.57 |
| Limpieza ×2 | 0.740 | 6.06 |
| Fuga ×2 + limpieza ×2 | 0.974 | 4.10 |

- **Sensibilidad a la visión** (recall 0.856, precisión 0.876, Monte Carlo): tiempo en Crítico entre 12,9 % y 15,8 % (IC 95 %).
- **Validación:** la EDO por fases (ajustada con el 70 % de cada fase) mejora al predictor ingenuo en un 58 % (RMSE 0,073 vs 0,175).

Gráficas: `resultados/resultados.png`. Video anotado: `resultados/video_anotado.mp4`.

## 5. Limitaciones

- El video de prueba es corto y contiene un solo evento; la fase de derrame dura pocos segundos, por lo que los parámetros son indicativos.
- La cadena de Markov asume un proceso estacionario, pero el video no lo es (derrame → pico → limpieza). La distancia de variación total en la validación fue 0,32.
- El sMAPE es alto (150 %) porque el área tiende a ~0 en la fase de limpieza; se usa RMSE como métrica principal.
- Falsos positivos, reflejos, líquidos transparentes, oclusiones e iluminación afectan `A(t)`. Se propagó el error del detector (recall y precisión) con Monte Carlo.
- Se aplicó un filtro de media móvil a `A(t)` para reducir el parpadeo de las detecciones entre cuadros.

## 6. Cómo reproducirlo (PowerShell, Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Para GPU NVIDIA RTX serie 50 (Blackwell) se necesita PyTorch con CUDA 12.8+:
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

python examen2_liquid.py prepare --ndjson cls-liquid.ndjson
python examen2_liquid.py train --data dataset/data.yaml --device 0
python examen2_liquid.py run --weights resultados\best.pt --fuente "video.mp4" --recall 0.856 --precision 0.876 --show
```

## 7. Estructura

```
examen2_liquid.py      # prepare / train / run (Fases I-IV)
cls-liquid.ndjson      # anotaciones y URLs del dataset
requirements.txt
resultados/            # best.pt, metricas_test.json, resumen.json, escenarios.csv, serie_vision.csv, resultados.png, video_anotado.mp4
entrenamiento/         # curvas y matriz de confusión del entrenamiento
```
