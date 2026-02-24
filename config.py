"""Configuration parameters for the project."""

# Time step (ms)
DT = 1.0

# Matching tolerance (ms)
MATCH_TOLERANCE = 2.0

# Data paths
TRAIN_DATA_PATH = "./data/train"
TEST_DATA_PATH = "./data/test"
OUTPUT_PATH = "./predictions"

# Training parameters
N_TRAIN_FILES = 32  # Total training files
N_VAL_FILES = 6     # Files to use for validation
MAX_ROWS = None     # Set to limit rows for faster testing (e.g., 500)

# Optimization parameters
OPTIMIZATION_METHOD = 'random'  # 'random' or 'grid'
OPTIMIZATION_ITERATIONS = 30    # For random search

# Available models (to easily switch between them)
AVAILABLE_MODELS = [
    'LIF',
    'Izhikevich_RS',
    'Izhikevich_FS',
    'AdEx',
    'SRM',
    'RateBased'
]

# Default model to use
DEFAULT_MODEL = 'LIF'