import yaml
from sculpgen import Logger


class RYaml:
    """
    Experimental
    """

    _config: dict

    def __init__(self):
        """
        experimental
        """
        self._log = Logger().log

    def read_yaml(self, yaml_file_path: str):
        """Experimental"""
        try:
            with open(file=yaml_file_path, mode="r", encoding="utf-8") as yaml_file:
                self._config = yaml.safe_load(yaml_file)
        except FileNotFoundError:
            print(f"No file found in specified path {yaml_file}")
            return
        except yaml.YAMLError:
            print("Error reading the contents of yaml file specified.")
            return
