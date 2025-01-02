import importlib
import os

def load_config(config_file='config.py'):
    """
    Load the configuration dynamically based on the config file.

    Parameters:
    - config_file (str): Path to the main configuration file.

    Returns:
    - module: The dynamically loaded configuration module.
    """
    config_module_name = None

    # Ensure the config file exists
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The config file '{config_file}' was not found.")

    try:
        # Parse the config type from the file
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
        config_module = importlib.import_module(config_module_name)

        # Validate required attributes
        validate_config_attributes(config_module)

    except Exception as e:
        raise Exception(f"Error loading config module from '{config_file}': {e}")

    return config_module


def validate_config_attributes(config):
    """
    Validate that the configuration module contains required attributes.

    Parameters:
    - config: The dynamically loaded configuration module.

    Raises:
    - AttributeError: If a required attribute is missing.
    """
    required_attributes = ["CONFIG_TYPE", "RESULTS_DIR", "VISUALS_DIR"]
    for attr in required_attributes:
        if not hasattr(config, attr):
            raise AttributeError(f"Config module is missing required attribute: {attr}")


def ensure_directories(config):
    """
    Ensure that the directories specified in the config exist.

    Parameters:
    - config: The loaded configuration module.
    """
    directories = [config.RESULTS_DIR, config.VISUALS_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory exists: {directory}")


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
