from typing import Optional
import torch
import chathist
from chathist import GPT2


class Model:
    """
    Experimental
    """

    _tokenizer: chathist.tokenizer.Tokenizer
    _model: GPT2
    _device: str

    _train_loader: torch.utils.data.DataLoader
    _val_loader: torch.utils.data.DataLoader | None = None

    _epochs: int = 2
    _optimizer: torch.optim.Optimizer
    _loss: torch.nn.CrossEntropyLoss
    _learning_rate: float

    def __init__(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
    ):
        self._tokenizer = chathist.config.tokenizer
        self._device = chathist.config.device
        self._learning_rate = chathist.config.lr
        self._epochs = chathist.config.epochs

        self._model = GPT2()
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._learning_rate
        )
        self._loss = torch.nn.CrossEntropyLoss()

        self._model.load_weights()
        self._train_loader = train_loader
        if val_loader is not None:
            self._val_loader = val_loader

    def train(self):
        """
        Experimental
        """
        self._model.to(self._device)

        for _ in range(self._epochs):
            self._model.train()
            for i, (inputs, targets) in enumerate(self._train_loader):
                print(i, inputs, targets)

            #     inputs.to(self._device)
            #     targets.to(self._device)

            #     self._optimizer.zero_grad()

            #     logits = self._model(inputs)
            #     loss: torch.Tensor = self._loss(logits, targets)

            #     loss.backward()
            #     self._optimizer.step()

            # self._model.eval()
            # with torch.inference_mode():

    def generate(self, prompt: str) -> str:
        """Experimental"""
        token_ids = self._tokenizer.encode_text(prompt)
        token_ids = torch.unsqueeze(token_ids, dim=0)
        self._model.eval()
        for _ in range(30):
            with torch.inference_mode():
                logits = self._model(token_ids)  # (1, token_len, emb_dim)
                last_token = logits[:, -1, :]  # Get the last embedding
                last_token = torch.softmax(
                    last_token, dim=-1
                )  # Making sure that all the logits are between 0 and 1.
                token = torch.argmax(last_token, dim=-1, keepdim=True)
                # print(token.shape, token_ids.shape)
                token_ids = torch.cat((token_ids, token), dim=-1)

        return self._tokenizer.decode_ids(token_ids.squeeze(dim=0))
