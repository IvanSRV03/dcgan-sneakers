# config.py — Configuración central del proyecto DCGAN

# ── Dataset ──────────────────────────────────────────────────────────────────
DATA_DIR       = "data/raw"          # Carpeta con tus imágenes originales
PROCESSED_DIR  = "data/processed"    # Imágenes preprocesadas (opcional)
IMAGE_SIZE     = 128                 # Resolución cuadrada de entrada
NUM_CHANNELS   = 3                   # RGB

# ── Entrenamiento ─────────────────────────────────────────────────────────────
BATCH_SIZE     = 32
NUM_EPOCHS     = 300
LEARNING_RATE  = 0.0002
BETA1          = 0.5                 # Adam β1 (recomendado para GANs)
BETA2          = 0.999               # Adam β2

# ── Arquitectura ──────────────────────────────────────────────────────────────
LATENT_DIM     = 128                 # Dimensión del vector de ruido (z)
G_FEATURES     = 64                  # Feature maps base del Generator
D_FEATURES     = 64                  # Feature maps base del Discriminator

# ── Outputs ───────────────────────────────────────────────────────────────────
OUTPUT_DIR         = "outputs"
SAMPLES_DIR        = "outputs/samples"
CHECKPOINTS_DIR    = "outputs/checkpoints"
SAMPLE_INTERVAL    = 10              # Guardar imágenes cada N epochs
CHECKPOINT_INTERVAL = 50             # Guardar checkpoint cada N epochs

# ── Misc ──────────────────────────────────────────────────────────────────────
SEED           = 42
NUM_WORKERS    = 4                   # DataLoader workers (ajusta según tu CPU)
