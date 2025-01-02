# utils/directory_setup.py

import os

def setup_directories(config):
    """
    Creates the necessary directories for saving data and results.

    Parameters:
    - config: The configuration object/module that contains directory paths.
    """
    directories = [config.PREPARED_DATA_DIR, 'univariant', 'bivariant', 'assumptions', 'initial_model', 'reduced_model']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory created: {directory}")
