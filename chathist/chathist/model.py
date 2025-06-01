from typing import Optional, Generator, Any
import os
import logging
from tqdm import tqdm
import torch
import pandas as pd
import chathist
from chathist import GPT2
from .instruction_styling import InstructionStyle


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
    _train_model: bool = True
    _tokenizer: chathist.tokenizer.Tokenizer

    def __init__(self):
        self._tokenizer = chathist.config.tokenizer
        self._device = chathist.config.device
        self._learning_rate = chathist.config.lr
        self._endoftext = chathist.config.endoftext
        self._epochs = chathist.config.epochs
        self._log = chathist.config.log
        self._save_path = chathist.config.save_path
        self._context_length = chathist.config.gpt_flavor["context_length"]

        self._log.info("Device Selected: %s", self._device)
        self._style = InstructionStyle.load()
        self._style_name = chathist.config.style_name
        self._model = GPT2()

        if os.path.exists(self._save_path):
            self._log.info(
                "Loading saved model on path %s\n",
                self._save_path,
            )
            self._train_model = False

            checkpoint = torch.load(self._save_path, map_location=self._device)

            self._model.load_state_dict(checkpoint)

        else:
            self._model.load_weights()
            self._optimizer = torch.optim.AdamW(
                self._model.parameters(), lr=self._learning_rate, weight_decay=0.1
            )
            self._loss = torch.nn.CrossEntropyLoss()

        self._model.to(self._device)

    def _calc_loss(self, inputs: torch.Tensor, targets: torch.Tensor):
        """Experiment"""
        inputs = inputs.to(self._device)
        targets = targets.to(self._device)
        logits = self._model(inputs)

        # pylint: disable=no-member
        return self._loss(logits.flatten(0, 1), targets.flatten().long())

    def _evaluate(self, _loader: torch.utils.data.DataLoader):
        """Experimental"""
        self._model.eval()
        total_loss = 0.0
        # Since index starts from 0 below
        batches = 0
        with torch.no_grad():
            for inputs, targets in tqdm(_loader, colour="blue"):
                loss: torch.Tensor = self._calc_loss(inputs, targets)
                total_loss += loss.item()
                batches += 1

        self._log.info("Validation Loss: %s", total_loss / batches)
        self._model.train()

    def generate(
        self, prompt: str, top_k: int = 0, temperature: float = 0.0
    ) -> Generator[str, Any, Any]:
        """Experimental"""
        if not self._style.is_format(prompt):
            self._log.info("Formatting prompt into %s style", self._style_name)
            prompt = self._style.format(prompt)

        token_ids = self._tokenizer.encode_text(prompt)
        token_ids = torch.unsqueeze(token_ids, dim=0).to(self._device)
        self._model.eval()
        # response = []
        token = 0  # Assign random value for now
        for _ in range(self._context_length):
            with torch.inference_mode():
                logits = self._model(token_ids)  # (1, token_len, emb_dim)
                last_token = logits[:, -1, :]  # Get the last embedding
                # Making sure that all the logits are between 0 and 1.
                last_token = torch.softmax(last_token, dim=-1)

                if top_k > 0:
                    top_logits, _ = torch.topk(last_token, top_k)
                    min_value = top_logits[:, -1]
                    last_token = torch.where(
                        last_token < min_value,
                        torch.tensor(float("-inf")).to(self._device),
                        last_token,
                    )
                if temperature > 0.0:
                    last_token = last_token / temperature
                    probs = torch.softmax(last_token, dim=-1)
                    token = torch.multinomial(probs, num_samples=1)
                else:
                    token = torch.argmax(last_token, dim=-1, keepdim=True)

                if token == self._endoftext:
                    break

                # response.append(token.item())
                token_ids = torch.cat((token_ids, token), dim=-1)

            yield self._tokenizer.decode_ids(token)

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

        if self._train_model:
            for epoch in range(self._epochs):
                self._model.train()
                self._log.info("Epoch %s", epoch + 1)
                total_loss = 0
                batches = 0
                for inputs, targets in tqdm(train_loader, colour="green"):
                    # self._log.info("Batch %s", i + 1)

                    self._optimizer.zero_grad()

                    loss: torch.Tensor = self._calc_loss(inputs=inputs, targets=targets)

                    total_loss += loss.item()
                    batches += 1
                    loss.backward()

                    self._optimizer.step()

                self._log.info("Train Loss: %s", total_loss / batches)
                if isinstance(val_loader, torch.utils.data.DataLoader):
                    self._evaluate(val_loader)
            torch.save(self._model.state_dict(), self._save_path)
        else:
            self._log.warning(
                "Trained model exists on saved path %s\n Skipping Training",
                self._save_path,
            )
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
