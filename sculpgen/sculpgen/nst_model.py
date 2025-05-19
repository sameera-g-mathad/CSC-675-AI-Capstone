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


class NSTModel(RYaml):
    """
    @note: Comments are generated with the help of co-pilot.

    This class implements a Neural Style Transfer (NST) model using VGG19.
    It inherits from the RYaml class which is used to read the configuration file.
    The class is used to generate stylized images using a content image and a style image.
    The class uses the VGG19 model to extract features from the content and style images.
    The class uses the Adam optimizer to update the target image to minimize the content and style loss.
    The class uses the PyTorch library to perform the computations on the GPU if available.
    """

    def __init__(self, yaml_file_path: str):
        # Initialize the parent class
        super().__init__()

        # Read the YAML configuration file
        self.read_yaml(yaml_file_path=yaml_file_path)
        self._log.info("YAML file read successfully.")

        # Set the random seed for reproducibility
        random_seed = self._config["train"]["random_seed"]
        torch.manual_seed(random_seed)
        # random.seed(random_seed) # You can uncomment if you want similar images.

        # Set the device to GPU if available, otherwise CPU
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._log.info("Using device: %s.", self._device)

        # Check if the abstract folder exists to get the number of style images.
        try:
            self._styles_len = len(os.listdir(self._config["train"]["abstract_folder"]))

        except FileNotFoundError as e:
            self._log.error(e)

        # Initialize the VGG19 model and load it to the device
        # and set it to evaluation mode
        self._log.info("Loading VGG19 model.")
        self._model = VGG19()
        self._model.to(self._device)
        self._model.eval()
        self._log.info("VGG19 model loaded successfully.")

        # Load the dataset from Hugging Face
        self._log.info("Loading dataset from Hugging Face.")
        self._log.info(
            "Hugging Face repo: %s", self._config["train"]["huggingfaace_repo"]
        )
        self._df = load_dataset(self._config["train"]["huggingfaace_repo"])
        self._log.info("Dataset loaded successfully.")

        # Create the save path for the generated images
        self._log.info("Creating save path for generated images.")
        self._log.info("Save path: %s", self._config["train"]["save_path"])
        self._save_path = self._config["train"]["save_path"]
        # Create the save path if it doesn't exist
        os.makedirs(self._save_path, exist_ok=True)

        # Initialize the transforms for the images
        self._log.info("Initializing transforms for images.")
        resize = int(self._config["train"]["resize"])
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
        )  # ImageNet stats.

    def _content_loss(
        self, content: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        This method calculates the content loss between the content and target images.
        The content loss is defined as the mean squared error between the content and target images.

        @param content: The content image tensor.
        @param target: The target image tensor.
        @return: The content loss as a tensor.
        @rtype: torch.Tensor
        """
        content_loss = torch.mean((content - target) ** 2)
        return content_loss

    def _gram_matrix(self, image: torch.Tensor) -> torch.Tensor:
        """
        This method calculates the Gram matrix of an image tensor.
        The Gram matrix is used to capture the style of the image.
        It is calculated by reshaping the image tensor and performing matrix multiplication.

        @param image: The image tensor.
        @return: The Gram matrix as a tensor.
        @rtype: torch.Tensor
        """
        channel, width, height = image.shape
        image = image.view(channel, width * height)
        gram_mat = torch.mm(image, image.T)

        return gram_mat

    def _style_loss(self, style: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        This method calculates the style loss between the style and target images.
        The style loss is defined as the mean squared error between the Gram matrices of the style and target images.

        @param style: The style image tensor.
        @param target: The target image tensor.
        @return: The style loss as a tensor.
        @rtype: torch.Tensor

        @note: The Gram matrix is used to capture the style of the image.
        """
        style_gram = self._gram_matrix(style)
        target_gram = self._gram_matrix(target)
        style_loss = torch.mean((style_gram - target_gram) ** 2)
        return style_loss

    def _train(
        self,
        content_image: torch.Tensor,
        style_image: torch.Tensor,
        target_image: torch.Tensor,
    ) -> torch.Tensor:
        """
        This method trains the model to generate a stylized image.
        It uses the content and style images to compute the content and style loss.
        The target image is updated using the optimizer to minimize the loss.

        @param content_image: The content image tensor.
        @param style_image: The style image tensor.
        @param target_image: The target image tensor.
        @return: The generated stylized image tensor.
        @rtype: torch.Tensor

        @note: The content and style images are passed through the VGG19 model to extract features.
        """

        # Move the images to the device
        content_image = content_image.to(self._device)
        style_image = style_image.to(self._device)
        target_image = target_image.to(self._device)
        # Set the target image to require gradients which allows the optimizer to update it
        # during training.
        target_image.requires_grad = True

        # Set the optimizer to update the target image. This is done after
        # moiving the image to the device and setting it to require gradients.
        optimizer = torch.optim.Adam(
            [target_image], lr=self._config["train"]["learning_rate"]
        )

        # Set the number of epochs for training
        epoch = self._config["train"]["epochs"]

        # Set the number of iterations for training
        for _ in tqdm(range(epoch), colour="red"):
            # Clear the gradients of the optimizer
            optimizer.zero_grad()

            # Pass the content, style, and target images through the VGG19 model
            # to extract features.
            content_features = self._model(content_image)
            style_features = self._model(style_image)
            target_features = self._model(target_image)

            style_loss = 0.0
            content_loss = 0.0

            # Iterate through the features and calculate the content and style loss
            # for each layer of the VGG19 model.
            for content_feature, style_feature, target_feature in zip(
                content_features, style_features, target_features
            ):
                content_loss += self._content_loss(
                    content=content_feature, target=target_feature
                )
                style_loss += self._style_loss(
                    style=style_feature, target=target_feature
                )

            # Calculate the total loss as a weighted sum of the content and style loss
            # using the alpha and beta parameters from the config file.
            loss: torch.Tensor = (
                self._config["train"]["alpha"] * content_loss
                + self._config["train"]["beta"] * style_loss
            )

            # Backpropagate the loss to update the target image
            loss.backward()
            # Update the target image using the optimizer.
            optimizer.step()

        return target_image

    def convert_nst(self):
        """
        This method is used as a wrapper to call the train method.
        It iterates through the dataset and generates stylized images using the content and style images.
        It saves the generated images to the specified save path.

        @note: The content and style images are passed through the train method to generate the stylized image.
        """
        if isinstance(self._df, DatasetDict):

            # If the dataset is a DatasetDict, we need to get the train split
            self._df = self._df["train"]
            # Iterate through the dataset and generate stylized images. self._df has images and labels in them
            # You can check the dataset 'https://huggingface.co/datasets/Durgas/Indian_sculptures'.

            for content_image_num, image in enumerate(
                tqdm(self._df["image"][:1], colour="yellow")
            ):
                # Transform the image to a tensor and normalize it
                content_image = self._transforms(image)

                # Iterate for the number of images per content to generate
                # multiple stylized images for each content image
                for _ in range(self._config["train"]["images_per_content"]):

                    # Randomly select a style image from the abstract folder
                    style_image_num = random.randint(a=1, b=self._styles_len)

                    selected_filename = os.path.join(
                        self._config["train"]["abstract_folder"],
                        (
                            f"{self._config['train']['abstract_prefix']}"
                            f"{style_image_num}{self._config['train']['ext']}"
                        ),
                    )

                    style_image = Image.open(selected_filename).convert("RGB")
                    style_image = torch.Tensor(self._transforms(style_image))

                    # Copy/Clone the content image to a tensor
                    target_image = content_image.clone().detach()

                    # Train the model to generate a stylized image.
                    target_image = self._train(
                        content_image,
                        style_image,
                        target_image,
                    )

                    # Denormalize the target image and save it to the specified save path
                    target_image = self._denormalize(target_image)
                    save_image(
                        target_image,
                        fp=(
                            f"{self._save_path}/{self._config['train']['save_prefix']}"
                            f"content_{content_image_num}"
                            f"_style_{style_image_num}{self._config['train']['ext']}"
                        ),
                    )
