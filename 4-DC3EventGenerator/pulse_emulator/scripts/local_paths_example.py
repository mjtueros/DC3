"""Template for machine-specific locations.

Copy this file to ``local_paths.py`` (same folder) and fill in the values:

    cp scripts/local_paths_example.py scripts/local_paths.py

``local_paths.py`` is gitignored. The scripts import it after adding ``scripts/`` to
``sys.path``, so run them from the repository root:

    sys.path.append('scripts/')
    from local_paths import BASE_MODEL_PATH
"""
BASE_MODEL_PATH = "/sps/grand/aferrier/MODELS/generative_models/"   # trained models,         e.g. "/home/me/MODELS/generative_models"
BASE_DATA_PATH = ""    # datasets,               e.g. "/home/me/DATA/generative_data"
BASE_ROOT = "/sps/grand/DC2.1rc4/GP300ZHAireS-NJ"        # simulation ROOT files
RF_PARAMS_CONFIG_FILE = "/sps/grand/aferrier/MODELS/antenna/RF_params_new_leffs.json"  # RF-chain parameter JSON
LFMAP_DIR = ""         # galactic noise maps LFmap*.npy
