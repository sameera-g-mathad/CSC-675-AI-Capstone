from typing import Optional
import torch
from torchvision.models import vgg19, VGG19_Weights
from torch import nn


class VGG19(nn.Module):
    """
    @note: Comments are generated with the help of co-pilot.

    This class implements a VGG19 model for feature extraction.
    It uses the pretrained VGG19 model from torchvision and extracts features
    from specific layers defined in the _layers attribute.
    The model is initialized with a random seed for reproducibility.
    The forward method takes an image as input and passes it through the model,
    collecting the output from the specified layers.
    """

    def __init__(self, random_seed: Optional[int] = 42):
        """
        The constructor initializes the VGG19 model for feature extraction.
        It sets the random seed for reproducibility and defines the layers
        from which features will be extracted.

        :param random_seed: The random seed for reproducibility.
        :type random_seed: Optional[int]

        """
        super().__init__()
        torch.manual_seed(random_seed)

        # VGG19 Conv layers [7, (7 + 3), (7 + 3 + 4), (7 + 3 + 4 + 5), (7 + 3 + 4 + 5 + 6)]
        self._layers = ["7", "10", "14", "19", "25"]
        self._model = vgg19(weights=VGG19_Weights.DEFAULT).features

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        """
        The forward method takes an image as input and passes it through the VGG19 model,
        collecting the output from the specified layers.

        :param image: The input image tensor.
        :type image: torch.Tensor
        :return: A list of feature tensors from the specified layers.
        :rtype: list[torch.Tensor]
        """
        feature_arr: list[torch.Tensor] = []
        for name, layer in self._model.named_children():
            # image should pass through each layer be it conv, relu etc.
            image = layer(image)
            if name in self._layers:
                feature_arr.append(image)

        return feature_arr
