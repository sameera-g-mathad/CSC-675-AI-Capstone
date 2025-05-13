from typing import Optional
import logging
import torch
import pandas as pd
import chathist
from chathist import GPT2
from .instruction_styling import Instruction


class Model:
    """
    Experimental
    """

    _device: str
    _endoftext: int
    _epochs: int = 2
    _learning_rate: float
    _log: logging.Logger
    _loss: torch.nn.CrossEntropyLoss
    _model: GPT2
    _optimizer: torch.optim.Optimizer
    _tokenizer: chathist.tokenizer.Tokenizer

    def __init__(self, style: Instruction):
        self._tokenizer = chathist.config.tokenizer
        self._device = chathist.config.device
        self._learning_rate = chathist.config.lr
        self._endoftext = chathist.config.endoftext
        self._epochs = chathist.config.epochs
        self._log = chathist.config.log

        self._style = style
        self._model = GPT2()
        self._model.load_weights()

        self._optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self._learning_rate, weight_decay=0.1
        )

        self._model.to(self._device)
        self._loss = torch.nn.CrossEntropyLoss()

    def calc_loss(self, inputs: torch.Tensor, targets: torch.Tensor):
        """Experiment"""
        inputs = inputs.to(self._device)
        targets = targets.to(self._device)
        logits = self._model(inputs)

        # pylint: disable=no-member
        return self._loss(logits.flatten(0, 1), targets.flatten().long())

    def evaluate(self, _loader: torch.utils.data.DataLoader):
        """Experimental"""
        self._model.eval()
        total_loss = 0.0
        # Since index starts from 0 below
        batches = 0
        with torch.no_grad():
            for inputs, targets in _loader:
                loss: torch.Tensor = self.calc_loss(inputs, targets)
                total_loss += loss.item()
                batches += 1
        self._model.train()
        return total_loss / batches

    def generate(self, prompt: str) -> str:
        """Experimental"""
        if not self._style.is_format(prompt):
            self._log.info(
                "Formatting prompt into %s style", chathist.config.style_name
            )
            prompt = self._style.format(prompt)

        token_ids = self._tokenizer.encode_text(prompt)
        token_ids = torch.unsqueeze(token_ids, dim=0).to(self._device)
        self._model.eval()
        response = []
        for _ in range(30):
            with torch.inference_mode():
                logits = self._model(token_ids)  # (1, token_len, emb_dim)
                last_token = logits[:, -1, :]  # Get the last embedding
                # Making sure that all the logits are between 0 and 1.
                last_token = torch.softmax(last_token, dim=-1)

                token = torch.argmax(last_token, dim=-1, keepdim=True)

                if token == self._endoftext:
                    break

                response.append(token.item())
                token_ids = torch.cat((token_ids, token), dim=-1)

        return self._tokenizer.decode_ids(torch.tensor(response))

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
    ):
        """
        Experimental
        """
        train_loss: list = []
        val_loss: list | None = None if val_loader is None else []

        for epoch in range(self._epochs):
            self._model.train()
            self._log.info("Epoch %s", epoch + 1)
            total_loss = 0
            batches = 0
            for i, (inputs, targets) in enumerate(train_loader):
                # self._log.info("Batch %s", i + 1)

                self._optimizer.zero_grad()

                loss: torch.Tensor = self.calc_loss(inputs=inputs, targets=targets)

                total_loss += loss.item()
                batches += 1
                loss.backward()

                self._optimizer.step()

                # if (i + 1) % 10 == 0:
                #     self._log.info("Batch: %s, Loss: %s", (i+1), loss.item())
            self._log.info("Epoch: %s, Loss: %s", epoch, total_loss / batches)

        return train_loss, val_loss

    def trainable_params(self, verbose: bool = True) -> Optional[pd.DataFrame]:
        """
        Experimental
        """
        param_info: dict = self._model.get_model_params(verbose)
        self._log.info("Total model parameters: %s", param_info["total_params"])
        self._log.info("Total trainable parameters: %s", param_info["total_trainable"])
        if verbose:
            return param_info["df"]
