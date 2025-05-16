import torch
from torchvision import models
from torch import nn

torch.manual_seed(123)


class VGG19(nn.Module):

    def __init__(self):
        super().__init__()
        self._layers = ["7", "10", "14", "19", "25"]
        self._model = models.vgg19(pretrained=True).features

    def forward(self, image):
        feature_arr: list[list] = []
        for name, layer in self._model.named_children():
            # image should pass through each layer be it conv, relu etc.
            image = layer(image)
            if name in self._layers:
                feature_arr.append(image)

        return feature_arr
