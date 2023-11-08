from ipyfilechooser import FileChooser
import yaml
from pathlib import Path

def select_settings_file(start_path = None):
    """Uses ipyfilechooser to open a gui for selecting a yml file.

    Returns:
        [type]: reference to ipyfilechooser window.
    """
    if start_path:
        fc = FileChooser(start_path)
    else:
        fc = FileChooser()
    fc.use_dir_icons = True
    fc.filter_pattern = '*.yml'
    display(fc)
    return fc

def parse_settings(file):
    file=Path(file)
    if file.is_file():
        # read yml
        with open(file) as data_file:
            settings = yaml.load(data_file, Loader=yaml.FullLoader)
        # check for important keywords.
        for key in ['server','cell_detection','registration','clustering','demix']:
            if key not in settings:
                raise NameError(f"Could not find key '{key}' in settings file")
        return settings

def parse_data_info(file):
    file=Path(file)
    if file.is_file():
        # read yml
        with open(file) as data_file:
            settings = yaml.load(data_file, Loader=yaml.FullLoader)
        # check for important keywords.
        for key in ['data','animal']:
            if key not in settings:
                raise NameError(f"Could not find key '{key}' in settings file")
        return settings