import yaml
from sculpgen import Logger


class RYaml:
    """
    Class for reading and writing YAML files.
    This class provides methods to read a YAML file and store its contents in a dictionary.
    It also handles exceptions related to file not found and YAML parsing errors.
    The read_yaml method reads the contents of a YAML file and stores it in the _config attribute.
    It prints an error message if the file is not found or if there is an error
    while reading the YAML file.
    """

    _config: dict

    def __init__(self):

        self._log = Logger().log

    def read_yaml(self, yaml_file_path: str):
        """
        This method reads a YAML file from the specified path and stores its contents in the _config attribute.
        It handles exceptions for file not found and YAML parsing errors.
        """
        try:
            with open(file=yaml_file_path, mode="r", encoding="utf-8") as yaml_file:
                self._config = yaml.safe_load(yaml_file)
        except FileNotFoundError:
            print(f"No file found in specified path {yaml_file}")
            return
        except yaml.YAMLError:
            print("Error reading the contents of yaml file specified.")
            return
