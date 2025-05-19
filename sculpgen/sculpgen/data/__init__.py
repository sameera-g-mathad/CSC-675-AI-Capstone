import os
from datasets import load_dataset, DatasetDict
from PIL import Image
from sculpgen import RYaml


class Data(RYaml):
    """Experimental"""

    def __init__(self, yaml_file_path: str):
        super().__init__()
        self.read_yaml(yaml_file_path=yaml_file_path)

    def rename(self):
        """Experimental"""
        try:
            directory = self._config["rename"]["directory"]
            rename = self._config["rename"]["rename"]
            rename_prefix = self._config["rename"]["rename_prefix"]
            if not rename:
                self._log.info("Rename flag is %s and files cannot be renamed", rename)
                return
            self._log.info("Reading files from %s", directory)
            for filenum, filename in enumerate(os.listdir(directory)):
                _, ext = os.path.splitext(filename)
                old_path = os.path.join(directory, filename)
                new_name = f"{rename_prefix}{filenum + 1}{ext}"
                new_path = os.path.join(directory, new_name)
                os.rename(old_path, new_path)
                self._log.info("Successfully renamed %s to %s", filename, new_name)
        except FileNotFoundError as e:
            self._log.error(e)

    def download_from_hf_datasets(self):
        """Experimental"""
        try:
            is_download = self._config["huggingface"]["download"]
            if not is_download:
                self._log.info(
                    "Download flag is %s and files cannot be renamed", is_download
                )
                return
            os.makedirs(self._config["huggingface"]["save_dir"], exist_ok=True)

            self._log.info(
                "Downloading files from %s",
                self._config["huggingface"]["huggingface_repo"],
            )
            df = load_dataset(path=self._config["huggingface"]["huggingface_repo"])
            if isinstance(df, DatasetDict):
                for image_num, image in enumerate(df["train"]["image"]):
                    if isinstance(image, Image.Image):
                        new_file = f"{self._config["huggingface"]["save_prefix"]}{image_num + 1}{self._config["huggingface"]["extension"]}"
                        file_path = os.path.join(
                            self._config["huggingface"]["save_dir"], new_file
                        )
                        image = image.convert("RGB")
                        image.save(file_path, format="JPEG")
                        self._log.info(
                            "Successfully downloaded %s to %s",
                            self._config["huggingface"]["huggingface_repo"],
                            file_path,
                        )
        except FileNotFoundError as e:
            self._log.error(e)


# data = Data("/home/smathad/ai_capstone/sculpgen/yaml/config.yaml")
# data.rename(
#     "/Users/sameergururajmathad/Documents/CSC - 675/AI Capstone/sculpgen/data/abstract"
# )
# data.download_from_hf_datasets(
#     repo_name="Durgas/Indian_sculptures",
#     output_dir="/Users/sameergururajmathad/Documents/CSC - 675/AI Capstone/sculpgen/data/sculptures",
# )
