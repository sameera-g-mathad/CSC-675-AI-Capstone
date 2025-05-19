import os
import torch
import random
import time
from tqdm import tqdm
from torchvision.transforms import transforms
from torchvision.utils import save_image
from PIL import Image
from datasets import load_dataset, DatasetDict
from sculpgen import RYaml, VGG19


class Model(RYaml):
    """Experimental"""

    def __init__(self, yaml_file_path: str):
        super().__init__()
        self.read_yaml(yaml_file_path=yaml_file_path)

        random_seed = self._content["train"]["random_seed"]
        torch.manual_seed(random_seed)
        # random.seed(random_seed) # You can uncomment if you want similar images.

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self._styles_len = len(
                os.listdir(self._content["train"]["abstract_folder"])
            )

        except FileNotFoundError as e:
            self._log.error(e)

        self._model = VGG19()
        self._model.to(self._device)
        self._model.eval()

        self._df = load_dataset(self._content["train"]["huggingfaace_repo"])

        self._save_path = self._content["train"]["save_path"]
        os.makedirs(self._save_path, exist_ok=True)

        resize = int(self._content["train"]["resize"])
        self._transforms = transforms.Compose(
            [
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),  # ImageNet stats.
            ]
        )
        self._denormalize = transforms.Normalize(
            (-2.12, -2.04, -1.80), (4.37, 4.46, 4.44)
        )

    def content_loss(self, content: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Experimental"""
        content_loss = torch.mean((content - target) ** 2)
        return content_loss

    def gram_matrix(self, image: torch.Tensor) -> torch.Tensor:
        """Experimental"""
        channel, width, height = image.shape
        image = image.view(channel, width * height)
        gram_mat = (image @ image.T) / (channel * width * height)

        return gram_mat

    def style_loss(self, style: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Experimental"""
        style_gram = self.gram_matrix(style)
        target_gram = self.gram_matrix(target)
        style_loss = torch.mean((style_gram - target_gram) ** 2)

        return style_loss

    def train(
        self,
        content_image: torch.Tensor,
        style_image: torch.Tensor,
        target_image: torch.Tensor,
        *args,
    ):
        """Experimental"""
        optimizer = torch.optim.Adam(
            [target_image], lr=self._content["train"]["learning_rate"]
        )
        epoch = self._content["train"]["epochs"]

        content_image = content_image.to(self._device)
        style_image = style_image.to(self._device)
        target_image = target_image.to(self._device)

        target_image.requires_grad = True
        content_image_num, style_image_num = args

        for _ in tqdm(range(epoch), colour="blue"):
            content_features = self._model(content_image)
            print("content", content_features[0].requires_grad)
            style_features = self._model(style_image)
            target_features = self._model(target_image)
            optimizer.zero_grad()

            style_loss = 0.0
            content_loss = 0.0

            for content_feature, style_feature, target_feature in zip(
                content_features, style_features, target_features
            ):
                content_loss += self.content_loss(
                    content=content_feature, target=target_feature
                )
                style_loss += self.style_loss(
                    style=style_feature, target=target_feature
                )

            loss: torch.Tensor = (
                self._content["train"]["alpha"] * content_loss
                + self._content["train"]["beta"] * style_loss
            )
            loss.backward()

            optimizer.step()

        target_image = self._denormalize(target_image)
        save_image(
            target_image,
            fp=f"{self._save_path}/{self._content['train']['save_prefix']}content_{content_image_num}_style_{style_image_num}{self._content['train']['ext']}",
        )

    def convert_nst(self):
        """Experimental"""
        if isinstance(self._df, DatasetDict):
            for content_image_num, image in enumerate(self._df["train"]["image"][:1]):
                image = self._transforms(image)
                for _ in range(self._content["train"]["images_per_content"]):
                    style_image_num = random.randint(a=1, b=self._styles_len)

                    selected_filename = os.path.join(
                        self._content["train"]["abstract_folder"],
                        f"{self._content['train']['abstract_prefix']}{style_image_num}{self._content['train']['ext']}",
                    )

                    style_image = Image.open(selected_filename).convert("RGB")

                    style_image = self._transforms(style_image)

                    target_image = image.clone().detach()

                    self.train(
                        image,
                        style_image,
                        target_image,
                        content_image_num,
                        style_image_num,
                    )
