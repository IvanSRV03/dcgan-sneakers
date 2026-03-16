# DCGAN — Generador de Tenis 👟

Proyecto de Deep Learning: entrenamiento de una **DCGAN** para generar imágenes sintéticas de tenis a partir de un dataset de ~1000 imágenes (StockX, fondo blanco, 128×128px).

---

## Estructura del Proyecto

```
dcgan-sneakers/
├── config.py              ← Todos los hiperparámetros en un solo lugar
├── requirements.txt
├── data/
│   └── raw/               ← PON AQUÍ tus imágenes de tenis (.jpg/.png)
├── outputs/
│   ├── samples/           ← Grillas de imágenes generadas por epoch
│   └── checkpoints/       ← Pesos del modelo (.pt)
├── src/
│   ├── dataset.py         ← DataLoader + transformaciones
│   ├── model.py           ← Arquitectura Generator y Discriminator
│   ├── train.py           ← Loop de entrenamiento principal
│   ├── generate.py        ← Generación e interpolación latente
│   └── utils.py           ← Seeds, checkpoints, visualizaciones
└── notebooks/             ← Experimenta aquí (exploración, visualizaciones)
```

---

## Setup

### 1. Clonar el repo e instalar dependencias

```bash
git clone https://github.com/TU_USUARIO/dcgan-sneakers.git
cd dcgan-sneakers
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> ⚡ **RTX 3070**: usa CUDA 12.1 (`cu121`). Verifica con `python -c "import torch; print(torch.cuda.is_available())"` → debe ser `True`.

### 2. Preparar el dataset

Copia todas tus imágenes de tenis en:
```
data/raw/
├── sneaker_001.jpg
├── sneaker_002.jpg
└── ...
```

No necesitas subcarpetas. El `DataLoader` las encontrará automáticamente.

---

## Entrenamiento

```bash
python src/train.py
```

Durante el entrenamiento verás en consola:
- **D_loss**: pérdida del Discriminador (idealmente ~0.5–1.0, estable)
- **G_loss**: pérdida del Generador (tiende a subir al principio, normal)
- **D(x)**: qué tan "real" clasifica el Discriminador las imágenes reales (idealmente ~0.8–0.9)
- **D(G)**: qué tan "real" clasifica las imágenes falsas (idealmente subirá con el tiempo)

Cada 10 epochs se guarda una grilla de muestra en `outputs/samples/`.

### Reanudar entrenamiento

```bash
RESUME_CHECKPOINT=outputs/checkpoints/checkpoint_epoch_0150.pt python src/train.py
```

---

## Generación de imágenes nuevas

### Imágenes aleatorias

```bash
python src/generate.py --checkpoint outputs/checkpoints/checkpoint_epoch_0300.pt --n 64
```
→ Guarda `outputs/generadas.png`

### Interpolación entre dos tenis

Genera una secuencia que muestra cómo el espacio latente "mezcla" características visuales entre dos puntos:

```bash
python src/generate.py --checkpoint outputs/checkpoints/checkpoint_epoch_0300.pt --interpolate --steps 10
```
→ Guarda `outputs/interpolacion.png`

---

## Hiperparámetros clave (`config.py`)

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `IMAGE_SIZE` | 128 | Resolución de entrada/salida |
| `LATENT_DIM` | 128 | Dimensión del vector z |
| `BATCH_SIZE` | 32 | Batches de entrenamiento |
| `NUM_EPOCHS` | 300 | Epochs totales |
| `LEARNING_RATE` | 0.0002 | LR para G y D (igual, Adam) |
| `BETA1` | 0.5 | Recomendado en paper DCGAN |
| `G_FEATURES / D_FEATURES` | 64 | Feature maps base |

---

## Señales de entrenamiento saludable

- D_loss ≈ 0.5–1.5 → el Discriminador no colapsa ni se vuelve perfecto
- G_loss baja gradualmente a lo largo de las epochs
- Las muestras de `outputs/samples/` muestran estructura reconocible antes de la epoch 50

## Problemas comunes

| Síntoma | Causa probable | Solución |
|---------|----------------|----------|
| Todas las imágenes iguales | Mode collapse | Bajar LR, aumentar `LATENT_DIM` |
| D_loss → 0 rápido | D demasiado fuerte | Reducir `D_FEATURES` o aumentar `G_FEATURES` |
| Imágenes con ruido puro siempre | G no aprende | Verificar normalización del dataset |
| OOM en GPU | Batch muy grande | Reducir `BATCH_SIZE` a 16 |

---

## Referencia

Radford et al., *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks* (2015). [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
