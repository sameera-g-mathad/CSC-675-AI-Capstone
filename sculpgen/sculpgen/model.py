import os
import torch
from tqdm import tqdm
from torchvision.transforms import transforms
from torchvision.utils import save_image
from PIL import Image
from datasets import load_dataset, DatasetDict
from sculpgen import RYaml, VGG19


class Model(RYaml):
    """Experimental"""

    def __init__(self):
        super().__init__()
        self._styles_len = len(
            os.listdir(
                "/Users/sameergururajmathad/Documents/CSC - 675/AI Capstone/sculpgen/data/abstract"
            )
        )
        self._model = VGG19()
        self._model.eval()
        self._df = load_dataset("Durgas/Indian_sculptures")
        self._transforms = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),  # ImageNet stats.
            ]
        )

    def content_loss(self, content: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Experimental"""
        return torch.mean((content - target) ** 2)

    def gram_matrix(self, image: torch.Tensor) -> torch.Tensor:
        """Experimental"""
        channel, width, height = image.shape
        image = image.view(channel, width * height)
        return image @ image.T

    def style_loss(self, style: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Experimental"""
        style_gram = self.gram_matrix(style)
        target_gram = self.gram_matrix(target)
        return torch.mean((style_gram - target_gram) ** 2)

    def train(self):
        """Experimental"""
        if isinstance(self._df, DatasetDict):
            for image in self._df["train"]["image"][:2]:
                image = self._transforms(image)
                style_image = Image.open(
                    "/Users/sameergururajmathad/Documents/CSC - 675/AI Capstone/sculpgen/data/abstract/abstract_87.jpg"
                )
                style_image = self._transforms(style_image)
                target_image = image.clone()
                target_image.requires_grad = True

                # print(image.shape, style_image.shape, target_image.shape)
                optimizer = torch.optim.Adam([target_image], lr=0.01)
                epoch = 200
                for _ in tqdm(range(epoch), colour="blue"):
                    conent_features = self._model(image)
                    style_features = self._model(style_image)
                    target_features = self._model(target_image)

                    optimizer.zero_grad()

                    style_loss = 0.0
                    content_loss = 0.0

                    for content_feature, style_feature, target_feature in zip(
                        conent_features, style_features, target_features
                    ):
                        content_loss += self.content_loss(
                            content=content_feature, target=target_feature
                        )
                        style_loss += self.style_loss(
                            style=style_feature, target=target_feature
                        )

                    loss: torch.Tensor = torch.Tensor(
                        1 * content_loss + 10000 * style_loss
                    )
                    loss.backward()
                    optimizer.step()

                target_image = transforms.Normalize(
                    (-2.12, -2.04, -1.80), (4.37, 4.46, 4.44)
                )(target_image)
                save_image(
                    target_image,
                    fp="/Users/sameergururajmathad/Documents/CSC - 675/AI Capstone/sculpgen/data/nst/output.jpg",
                )
