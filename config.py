# config.py — DCGAN mejorado

DATA_DIR       = "data/raw"
IMAGE_SIZE     = 128
NUM_CHANNELS   = 3

BATCH_SIZE     = 32
NUM_EPOCHS     = 500
LEARNING_RATE  = 0.0002

BETA1          = 0.5
BETA2          = 0.999

REAL_LABEL     = 0.9
FAKE_LABEL     = 0.0

LATENT_DIM     = 128
G_FEATURES     = 64
D_FEATURES     = 64

OUTPUT_DIR          = "outputs"
SAMPLES_DIR         = "outputs/samples"
CHECKPOINTS_DIR     = "outputs/checkpoints"
SAMPLE_INTERVAL     = 10
CHECKPOINT_INTERVAL = 50

SEED           = 42
NUM_WORKERS    = 4
