import os
from datasets import load_dataset, DatasetDict
from PIL import Image


class Data:

    def rename(self, directory: str):
        """Experimental"""
        for filenum, filename in enumerate(os.listdir(directory)):
            _, ext = os.path.splitext(filename)
            old_path = os.path.join(directory, filename)
            with Image.open(old_path) as image:
                new_name = f"abstract_{filenum + 1}{ext}"
                new_path = os.path.join(directory, new_name)
                image = image.resize((500, 500))
                image.save(new_path, format="JPEG")

    def download_from_hf_datasets(self, repo_name: str, output_dir: str):
        """Experimental"""
        df = load_dataset(path=repo_name)
        if isinstance(df, DatasetDict):
            for image_num, image in enumerate(df["train"]["image"]):
                if isinstance(image, Image.Image):
                    new_file = f"sculpture_{image_num}.jpg"
                    file_path = os.path.join(output_dir, new_file)
                    image = image.resize((500, 500))
                    image = image.convert("RGB")
                    image.save(file_path, format="JPEG")


data = Data()
data.rename(
    "/Users/sameergururajmathad/Documents/CSC - 675/AI Capstone/sculpgen/data/abstract"
)
data.download_from_hf_datasets(
    repo_name="Durgas/Indian_sculptures",
    output_dir="/Users/sameergururajmathad/Documents/CSC - 675/AI Capstone/sculpgen/data/sculptures",
)
