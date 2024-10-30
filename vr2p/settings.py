from pathlib import Path

import yaml
from ipyfilechooser import FileChooser
from IPython.display import display


def select_settings_file(start_path: Path | None = None) -> FileChooser:
    """Opens a GUI dialog that allows selecting the .yml settings file.

    This function is designed to be used from python notebooks or interactive Python shell. Do not use this function
    with non-interactive source code.

    Args:
        start_path: The path to the directory that should be opened when file chooser dialog starts.

    Returns:
        Reference to FileChooser GUI window.
    """
    # If starting path is provided, uses it as the initial path.
    if start_path:
        file_chooser = FileChooser(str(start_path))

    # Otherwise, defaults to the current working directory supplied by the OS module.
    else:
        file_chooser = FileChooser()

    # Enhances the search to specifically look for the .yml files.
    file_chooser.use_dir_icons = True
    file_chooser.filter_pattern = "*.yml"

    display(file_chooser)
    return file_chooser


def parse_settings(file: Path | str) -> Dict[str, Any]:
    """Parses settings from a configuration YAML file.

    Args:
        file: Path to the YAML settings file.

    Returns:
        Dictionary containing the parsed configuration settings.

    Raises:
        FileNotFoundError: If the specified .yml file doesn't exist.
        NameError: If required configuration parameters (keys) are not present in the loaded file.
    """
    file_path = Path(file)
    if not file_path.is_file():
        raise FileNotFoundError(f"Settings file not found at path {file_path}.")

    required_keys = {"server", "cell_detection", "registration", "clustering", "demix"}

    # Using FullLoader to support custom Python objects and YAML constructors
    with open(file_path) as data_file:
        settings = yaml.load(data_file, Loader=yaml.FullLoader)

    # Checks for missing required keys
    missing_keys = required_keys - settings.keys()
    if missing_keys:
        raise NameError(f"Missing required keys in settings file: {', '.join(missing_keys)}")

    return settings


def parse_data_info(file: Path | str) -> Dict[str, Any]:
    """Parses data ID information from a YAML file.

    Args:
        file: Path to the data information YAML file.

    Returns:
        Dictionary containing the ID information about the data and the animal(s) that produced it.

    Raises:
        FileNotFoundError: If the specified .yml file doesn't exist.
        NameError: If required data fields (keys) are not present in the loaded file.
    """
    file_path = Path(file)
    if not file_path.is_file():
        raise FileNotFoundError(f"Information file not found at path {file_path}.")

    required_keys = {"data", "animal"}

    # Using FullLoader to support custom Python objects and YAML constructors
    with open(file_path) as data_file:
        information = yaml.load(data_file, Loader=yaml.FullLoader)

        # Checks for missing required keys
        missing_keys = required_keys - information.keys()
        if missing_keys:
            raise NameError(f"Missing required keys in information file: {', '.join(missing_keys)}")

        return information
