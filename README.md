# DCGAN — Generador de Tenis

Experimento para el proyecto de Deep Learning. La idea era entrenar una DCGAN con fotos de tenis de StockX y ver si el modelo podía aprender a generar tenis nuevos por su cuenta.

El dataset son ~1220 imágenes en JPG, todas con fondo blanco, lo cual ayudó bastante porque el modelo no tenía que lidiar con fondos complicados.

---

## Qué hace

Entrena una red generativa adversarial (DCGAN) con imágenes de tenis. El Generator aprende a crear tenis falsos que el Discriminator no pueda distinguir de los reales. Después de 300 epochs el modelo ya genera tenis reconocibles con forma, suela y color.

También incluye interpolación en el espacio latente — básicamente mezclar dos puntos del espacio de ruido para ver cómo el modelo transiciona entre un estilo de tenis y otro.

---

## Resultados

| Epoch | Qué se ve |
|-------|-----------|
| 1 | Ruido puro, nada reconocible |
| 20 | Siluetas borrosas, colores básicos |
| 170 | Formas reconocibles, variedad de estilos |
| 300 | Tenis con forma, suela y color definidos |

La interpolación más interesante que salió fue un Yeezy Slide transformándose en zapatilla de fútbol con tacos — se ve cómo el modelo mezcla características visuales entre dos puntos del espacio latente.

---

## Estructura

```
dcgan-sneakers/
├── config.py           # Hiperparámetros
├── src/
│   ├── model.py        # Generator y Discriminator
│   ├── dataset.py      # Carga de imágenes
│   ├── train.py        # Entrenamiento
│   ├── generate.py     # Generación e interpolación
│   └── utils.py        # Checkpoints, visualizaciones
├── data/raw/           # Imágenes de tenis (no incluidas en el repo)
└── outputs/
    ├── samples/        # Grillas por epoch
    └── checkpoints/    # Pesos del modelo
```

---

## Cómo correrlo

```bash
# Instalar dependencias
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Entrenar
python src/train.py

# Generar tenis nuevos
python src/generate.py --checkpoint outputs/checkpoints/checkpoint_epoch_0300.pt

# Interpolación entre dos tenis
python src/generate.py --checkpoint outputs/checkpoints/checkpoint_epoch_0300.pt --interpolate --steps 10
```

---

## Setup usado

- GPU: NVIDIA RTX 3070 (8GB VRAM)
- Python 3.11 + PyTorch 2.5.1 con CUDA 12.1
- ~2 horas de entrenamiento para 300 epochs

---

## Notas

El dataset pequeño (1220 imágenes) se nota en los resultados — algunos tenis generados tienen artefactos de color o formas un poco raras. Con más datos probablemente mejoraría bastante la nitidez. Las imágenes de fondo blanco fueron una ventaja real, el modelo aprendió el fondo desde muy temprano y se concentró en la forma del tenis.
