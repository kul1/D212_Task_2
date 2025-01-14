import importlib
import os

def load_config(config_file='config.py'):
    """
    Load the configuration dynamically based on the config file.

    Parameters:
    - config_file (str): Path to the main configuration file.

    Returns:
    - config (module): The dynamically loaded configuration module with validated attributes.
    """
    config_module_name = None

    # Ensure the config file exists
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The config file '{config_file}' was not found.")

    try:
        # Parse the CONFIG_TYPE from the file
        with open(config_file, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith("CONFIG_TYPE"):
                    config_type_value = line.split('=')[1].split('#')[0].strip().strip('"').strip("'")
                    config_module_name = f"config_{config_type_value}"
                    break

        if not config_module_name:
            raise ValueError(f"No valid CONFIG_TYPE found in '{config_file}'")

        # Ensure the configuration module exists
        config_module_path = f"{config_module_name}.py"
        if not os.path.exists(config_module_path):
            raise FileNotFoundError(f"The configuration module '{config_module_path}' does not exist.")

        # Dynamically import the specified configuration module
        config = importlib.import_module(config_module_name)

        # Initialize derived attributes if not present
        initialize_config_attributes(config)

    except Exception as e:
        raise Exception(f"Error loading config module from '{config_file}': {e}")

    return config


def initialize_config_attributes(config):
    """
    Initialize or validate computed attributes in the config module.

    Parameters:
    - config: The loaded configuration module.

    Modifies:
    - Adds `RESULTS_DIR` and `VISUALS_DIR` attributes if they are missing.
    """
    if not hasattr(config, "RESULTS_DIR"):
        raise AttributeError("Missing required attribute 'RESULTS_DIR' in config.")
    if not hasattr(config, "VISUALS_DIR"):
        config.VISUALS_DIR = os.path.join(config.RESULTS_DIR, "visuals")

    # Ensure directories exist
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.VISUALS_DIR, exist_ok=True)


def get_flow_function(config):
    """
    Dynamically returns the flow function based on the CONFIG_TYPE.

    Parameters:
    - config: The loaded configuration module.

    Returns:
    - str: The full path of the flow function.
    """
    flow_function = f"flow.flow_{config.CONFIG_TYPE}.run_analysis"
    return flow_function


def import_flow_function(flow_function):
    """
    Dynamically imports the specified flow function.

    Parameters:
    - flow_function (str): The full path to the flow function.

    Returns:
    - function: The dynamically imported function object.
    """
    module_name, function_name = flow_function.rsplit('.', 1)
    try:
        flow_module = importlib.import_module(module_name)
        return getattr(flow_module, function_name)
    except ImportError as e:
        raise ImportError(f"Error importing module '{module_name}': {e}")
    except AttributeError as e:
        raise AttributeError(f"Function '{function_name}' not found in module '{module_name}': {e}")
