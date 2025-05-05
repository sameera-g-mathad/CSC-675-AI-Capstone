from functools import partial
import torch
from torch.utils.data import DataLoader, Dataset
import chathist


class InstructionDataset(Dataset):
    """
    Custom class for creating a dataset.
    """

    def __init__(self, data: list) -> None:
        self._tokenizer = chathist.config.tokenizer
        self._encoded = []
        for instance in data:
            self._encoded.append(self._tokenizer.encode_text(instance))

    def __getitem__(self, index):
        return self._encoded[index]

    def __len__(self):
        return len(self._encoded)


class InstructionDataLoader:
    """
    Class for creating pytorch DataLoaders.
    """

    def __init__(self):
        self._ignore_index = chathist.config.ignore_index
        self._endoftext = chathist.config.endoftext
        self._response_query = chathist.config.response_query

    def _custom_collate(self, batch, _endoftext=50256, mask_input: bool = False):
        """
        Experimental
        """
        _max_length = max([len(item) for item in batch])
        _inputs, _targets = [], []
        for _item in batch:
            _padding_len = _max_length - len(_item)
            _item = torch.cat(
                (
                    _item,
                    torch.tensor(
                        [self._endoftext],
                        requires_grad=False,
                        dtype=chathist.config.dtype,
                    ),
                )
            )
            _input = torch.cat(
                (
                    _item[:-1],
                    torch.tensor(
                        [self._endoftext] * _padding_len,
                        requires_grad=False,
                        dtype=chathist.config.dtype,
                    ),
                )
            )
            _target = torch.cat(
                (
                    _item[1:],
                    torch.tensor(
                        [self._ignore_index] * _padding_len,
                        requires_grad=False,
                        dtype=chathist.config.dtype,
                    ),
                )
            )

            assert len(_input) == len(
                _target
            ), "Input and targets should have equal length"

            if mask_input:
                print("Not supported yet")
            #     self._mask(_target, _tokenizer.encode_text(_response_query).numpy().tolist(),)
            _inputs.append(_input)
            _targets.append(_target)

        return torch.stack(_inputs), torch.stack(_targets)

    def load(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool,
        shuffle: bool,
        mask_input: bool,
    ):
        """
        Experimental
        """
        _collate_fn = partial(self._custom_collate, mask_input=mask_input)
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            collate_fn=_collate_fn,
        )

    # def _mask(self, source: list, mask: list):
