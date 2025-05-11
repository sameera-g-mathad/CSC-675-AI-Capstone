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
        self._response_ids = chathist.config.response_ids

    def _custom_collate(self, batch, mask_input: bool = False):
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
                _target = self._mask(_target, self._response_ids)
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

    def _mask(self, source: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        This method is used to mask all the input values including response ids
        to ignore_index so that the values are not considered for loss calculation
        and LLM does not generate redundant values

        :param torch.Tensor source: The source ids that needs to be masked.
        :param torch.Tensor mask: The mask ids used as a reference. Since this is used
        to mask until response query, this values are ids for response query.

        :rtype: torch.Tensor
        :returns: Returns the source tensor
        """
        i = 0  # iterator for source
        j = 0  # iterator for mask
        match_found = False

        # This line loops through all values in source
        while i < len(source):
            count = 0

            # This loop runs until there is no match found between source and mask,
            # this happens when the iterator i is at the start of response ids in the source
            while (
                i < len(source) and j < len(mask) and source[i].item() == mask[j].item()
            ):
                count += 1
                # All the items that are matching, i.e, response ids of source and mask ids
                # are matched and response ids is set to _ignore_index
                source[i] = self._ignore_index
                i += 1
                j += 1

                # if all the response ids was matched, then the loop should be terminated
                if count == len(mask):
                    match_found = True
                    break
            if match_found:
                break
            # If there is no match, then the values of source is set to ignore_index.
            source[i] = self._ignore_index
            i += 1
            j = 0
        return source
