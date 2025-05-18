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
            directory = self._content["data"]["directory"]
            rename = self._content["data"]["rename"]
            rename_prefix = self._content["data"]["rename_prefix"]
            if not rename:
                self._log.info("Rename is %s and files cannot be renamed", rename)
                return
            self._log.info("Reading files from %s", directory)
            for filenum, filename in enumerate(os.listdir(directory)):
                _, ext = os.path.splitext(filename)
                old_path = os.path.join(directory, filename)
                new_name = f"{rename_prefix}{filenum + 1}{ext}"
                new_path = os.path.join(directory, new_name)
                os.rename(old_path, new_path)
            self._log.info("Successfully renamed all files!!!")
        except FileNotFoundError as e:
            self._log.error(e)

    # def download_from_hf_datasets(self, repo_name: str, output_dir: str):
    #     """Experimental"""
    #     df = load_dataset(path=repo_name)
    #     if isinstance(df, DatasetDict):
    #         for image_num, image in enumerate(df["train"]["image"]):
    #             if isinstance(image, Image.Image):
    #                 new_file = f"sculpture_{image_num}.jpg"
    #                 file_path = os.path.join(output_dir, new_file)
    #                 image = image.resize((500, 500))
    #                 image = image.convert("RGB")
    #                 image.save(file_path, format="JPEG")


# data = Data("/home/smathad/ai_capstone/sculpgen/yaml/config.yaml")
# data.rename(
#     "/Users/sameergururajmathad/Documents/CSC - 675/AI Capstone/sculpgen/data/abstract"
# )
# data.download_from_hf_datasets(
#     repo_name="Durgas/Indian_sculptures",
#     output_dir="/Users/sameergururajmathad/Documents/CSC - 675/AI Capstone/sculpgen/data/sculptures",
# )
