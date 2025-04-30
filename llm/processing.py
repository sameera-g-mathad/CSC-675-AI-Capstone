import pandas as pd
from abc import ABC, abstractmethod
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class Tokenizer:
    """
    Wrapper for tiktoken package.
    """

    def __init__(self, encoding: str = "gpt2") -> None:
        """
        Constructor for the tokenizer.
        :param str encoding: Encoding name to be passed. By default, gpt2 is used.

        >>> tk = Tokenizer()
        """
        self.tokenizer = tiktoken.get_encoding(encoding)

    def encode_text(self, text: str, allowed_special="<|endoftext|>") -> list[int]:
        """
        Method to encode the text passed using the tokenizer initialized.
        Uses tiktoken.encode() method to achieve encoding

        :param str text: Text that needs to be encoded.
        :returns: List of encoded ids.
        :rtype: list[int]

        >>> tk.encode_text('what is the significance of indian history')
        >>> # Output: [10919, 318, 262, 12085, 286, 773, 666, 2106]
        """

        return self.tokenizer.encode(text, allowed_special={allowed_special})

    def decode_ids(self, ids: list[int]) -> str:
        """
        Method to decode the ids passed using the tokenizer initialized
        to get the encoded text back.
        Uses tiktoken.decode() method to achieve decoding.

        :param list[int] text: ids that needs to be decoded.
        :returns: retrieved string.
        :rtype: str

        >>> tk.decode_ids([10919, 318, 262, 12085, 286, 773, 666, 2106])
        >>> # Output: 'what is the significance of indian history'
        """

        return self.tokenizer.decode(ids)


class InstructionDataset(Dataset):
    """
    This class extends the Dataset class provided by pytorch.
    """

    def __init__(self, data: list, tokenizer: Tokenizer):
        self.data = data
        self.encoded_data = []

        for instance in self.data:
            self.encoded_data.append(tokenizer.encode_text(instance))

    def __getitem__(self, index):
        return self.encoded_data[index]

    def __len__(self):
        return len(self.encoded_data)


class InstructionLoader:
    """Experimental"""

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: Tokenizer,
    ):
        self.data = data
        self.tokenizer = tokenizer

    def custom_collate(
        self, batch, pad_token_id=50256
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This is a custom function given to pytorch's DataLoader
        for creating batches. This is becuase, we want to take the
        input and use this input as x and y (shifted by 1 word).
        Also, each batch need to be of equal length and is padded by
        `pad_token_id` here. This is in constrast with y, since these
        padding tokens shouldn't be considered for loss calculation.

        :returns: Tuple of x and y.
        :rtype: tuple[torch.Tensor, torch.Tensor].
        """
        max_len = max([len(x) for x in batch])

        inputs_x, target_y = [], []

        for item in batch:
            new_item = item.copy()
            new_item += [pad_token_id]

            inputs = new_item[:-1]
            input_len = len(inputs)
            targets = new_item[1:]
            target_len = len(targets)

            assert input_len == target_len, "Length of both input and output must match"

            if input_len < max_len:
                diff = max_len - input_len
                inputs += [pad_token_id] * diff
                targets += [-100] * diff

            inputs_x.append(torch.tensor(inputs))
            target_y.append(torch.tensor(targets))

        return torch.stack(inputs_x).to("cpu"), torch.stack(target_y).to("cpu")

    def load(self, val_size: float, batch_size: int):
        """Experimental"""
        # train_size = 1 - (test_size + val_size)
        train_size = 1 - val_size
        total_len = len(self.data)
        train_len = int(total_len * train_size)
        # val_len = train_len + int(total_len * val_size)

        train_data = InstructionDataset(
            self.data[:train_len]["instruct_data"].tolist(), self.tokenizer
        )
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.custom_collate,
        )
        val_data = InstructionDataset(
            self.data[train_len:]["instruct_data"].tolist(), self.tokenizer
        )
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.custom_collate,
        )
        # test_data = InstructionDataset(
        #     self.data[val_len:]["instruct_data"].tolist(), self.tokenizer
        # )

        # test_loader = DataLoader(
        #     test_data,
        #     batch_size=batch_size,
        #     shuffle=False,
        #     drop_last=False,
        #     collate_fn=self.custom_collate,
        # )

        return train_loader, val_loader


class InstructionStyle:
    """
    Class for creating instruction styles either alpaca or phi3.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        input_col: str,
        response_col: str | None = None,
        new_col: str = "instruct_data",
    ):
        """
        :param str input_col: This must be the column name that will be used as input query.
        """
        self.data = data
        self.input_col = input_col
        self.response_col = response_col
        self.new_col = new_col

    def convert(
        self,
        convert_type: str,
        input_query: str,
        response_query: str,
        prompt: str = "",
    ) -> pd.DataFrame:
        """

        :param str prompt: This must be the prompt that will be used as an instruction for the LLM. Only supported for alpaca
        :param str input_query: This must be a string for identifying the input.
        :param str response_query: This must be a string for identifying the response.
        :param str endoftext: Padding token to append endoftext for each input.

        :rtype:pd.DataFrame
        :returns: A new dataframe containing a column for the instrution dataset.
        """
        match (convert_type):
            case "alpaca":
                convert_func = self.alpaca
            case "phi3":
                convert_func = self.phi3
            case "_":
                convert_func = self.alpaca

        return pd.DataFrame(
            {
                self.new_col: self.data.apply(
                    lambda row: convert_func(
                        prompt,
                        input_query,
                        row[self.input_col],
                        response_query,
                        (
                            row[self.response_col]
                            if self.response_col is not None
                            else ""
                        ),
                    ),
                    axis=1,
                )
            }
        )

    def alpaca(
        self,
        prompt: str,
        input_query: str,
        _input: str,
        response_query: str,
        _response: str = "",
    ):
        """Experimental"""
        return f"{prompt}" f"{input_query}{_input}" f"{response_query}{_response}"

    def phi3(
        self,
        _prompt: str,
        input_query: str,
        _input: str,
        response_query: str,
        _response: str = "",
    ):
        """Experimental"""
        return f"{input_query}{_input}" f"\n" f"{response_query}{_response}"


class Model:
    """Experimental"""

    def __init__(self, device: str) -> None:
        self.device = device

    def generate(
        self,
        model,
        idx,
        max_new_tokens=10,
        context_size=256,
        temperature=0.0,
        top_k=None,
        eos_id=None,
    ):

        # For-loop is the same as before: Get logits, and only focus on last time step
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = model(idx_cond)
            logits = logits[:, -1, :]

            # New: Filter logits with top_k sampling
            if top_k is not None:
                # Keep only top_k values
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float("-inf")).to(logits.device),
                    logits,
                )

            # New: Apply temperature scaling
            if temperature > 0.0:
                logits = logits / temperature

                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            # Otherwise same as before: get idx of the vocab entry with the highest logits value
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

            if (
                idx_next == eos_id
            ):  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
                break

            # Same as before: append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

        return idx

    def calc_batch_loss(self, x, y, model):
        x, y = x.to(self.device), y.to(self.device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
        return loss

    def calc_loss_loader(self, data_loader, model):
        total_loss = 0
        num_batches = 0
        for x, y in data_loader:
            num_batches += 1
            loss = self.calc_batch_loss(x, y, model)
            total_loss += loss.item()

        return total_loss / num_batches

    def train(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
        optimizer: torch.optim.Optimizer,
    ):
        train_loss = []
        val_loss = []
        model.to(self.device)
        print(f"Model is on {next(model.parameters()).device}")
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}:")
            model.train()
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                print(f"Batch number: {i+1}")
                optimizer.zero_grad()
                loss = self.calc_batch_loss(x, y, model)
                loss.backward()
                optimizer.step()
                # Explicitly delete x and y to free GPU memory
                del x, y, loss

                # Force garbage collection
                # gc.collect()

                # Clear PyTorch's CUDA memory cache
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            model.eval()
            with torch.no_grad():
                train_loss.append(self.calc_loss_loader(train_loader, model))
                val_loss.append(self.calc_loss_loader(val_loader, model))
        return model, train_loss, val_loss
