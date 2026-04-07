# DCGAN — Generador de Tenis 👟

Proyecto para la materia de Deep Learning. Entrené una DCGAN con ~1,220 fotos de tenis de StockX para que el modelo aprendiera a generar tenis nuevos por su cuenta.

El dataset son imágenes JPG con fondo blanco uniforme, lo cual ayudó bastante porque el modelo no tuvo que lidiar con fondos complicados y se concentró en aprender la forma del tenis desde temprano.

---

## Resultados

| Epoch | Qué se ve |
|-------|-----------|
| 1 | Ruido puro, nada reconocible |
| 20 | Siluetas borrosas, colores básicos emergentes |
| 170 | Formas reconocibles, variedad de estilos |
| 300 | Tenis con forma, suela y color definidos |
| 500 | Mayor detalle, texturas, variedad de estilos |

La interpolación más interesante: una bota alta Nike transformándose en un cleat de fútbol americano en 10 pasos — la caña baja, aparecen los tacos, la forma se alarga. El modelo lo aprendió solo, sin etiquetas.

---

## Arquitectura

DCGAN siguiendo el paper de Radford et al. (2015), adaptada para 128×128px en lugar de las 64×64 originales.

**Generator** — toma un vector de ruido `z` (dim=128) y lo expande con 6 capas ConvTranspose2d hasta llegar a una imagen RGB 128×128. Usa BatchNorm + ReLU en cada capa y Tanh al final. ~13.2M parámetros.

**Discriminador** — recibe una imagen 128×128 y la comprime con Conv2d stride=2 hasta producir un único logit real/falso. Usa BatchNorm + LeakyReLU(0.2). ~11.1M parámetros.

---

## Decisiones técnicas

- **BCEWithLogitsLoss** — más estable numéricamente que BCE + Sigmoid por separado
- **Label Smoothing (0.9)** — evita que el Discriminador se vuelva demasiado confiado
- **Adam con β1=0.5** — recomendado en el paper original para estabilizar el entrenamiento
- **LR Scheduler** — reduce el LR a la mitad en epoch 250 y 400 para refinar detalles
- **Data Augmentation** — flip horizontal, random crop y variaciones de brillo/contraste

---

## Experimentación

Se probaron tres variantes durante el proyecto:

| Variante | Resultado |
|----------|-----------|
| DCGAN 300 epochs | Baseline, tenis reconocibles |
| WGAN + weight clipping | Loss explotó, descartado |
| WGAN-GP | Estable pero resultados visuales inferiores con este dataset |
| **DCGAN + Augmentation + LR Scheduler 500 epochs** | **Versión final, mejores resultados** |

---

## Estructura

```
dcgan-sneakers/
├── config.py           # Hiperparámetros
├── src/
│   ├── model.py        # Generator y Discriminador
│   ├── dataset.py      # Carga de imágenes + augmentation
│   ├── train.py        # Loop de entrenamiento
│   ├── generate.py     # Generación e interpolación SLERP
│   └── utils.py        # Checkpoints, visualizaciones
├── data/raw/           # Imágenes de tenis (no incluidas)
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
python src/generate.py --checkpoint outputs/checkpoints/checkpoint_epoch_0500.pt

# Interpolación entre dos tenis
python src/generate.py --checkpoint outputs/checkpoints/checkpoint_epoch_0500.pt --interpolate --steps 10
```

---

## Setup

- GPU: NVIDIA RTX 3070 (8.6 GB VRAM)
- Python 3.11 + PyTorch 2.5.1 con CUDA 12.1
- ~4 horas de entrenamiento para 500 epochs

---

## Notas

Con 1,220 imágenes el modelo aprende bien la estructura global pero le cuesta el detalle fino — logos, agujetas, texturas específicas. Con más datos y más epochs probablemente mejoraría bastante. Las imágenes de fondo blanco fueron una ventaja real.

Se experimentó con WGAN y WGAN-GP pero el DCGAN con augmentation y más epochs dio mejores resultados visuales para este dataset en particular.

---

## Referencia

Radford et al., *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks* (2015). [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
