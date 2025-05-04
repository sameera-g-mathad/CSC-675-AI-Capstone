import torch
import tiktoken


class Tokenizer:
    """
    Wrapper for tiktoken package.
    """

    def __init__(self, encoding: str = "gpt2") -> None:
        """
        Constructor for the tokenizer.
        :param str encoding: Encoding name to be passed. By default, gpt2 is used.

        :examples:
        >>> tk = Tokenizer()
        """
        self.tokenizer = tiktoken.get_encoding(encoding)

    def encode_text(
        self, text: str, allowed_special: set | None = None
    ) -> torch.Tensor:
        """
        Method to encode the text passed using the tokenizer initialized.
        Uses tiktoken.encode() method to achieve encoding

        :param str text: Text that needs to be encoded.
        :returns: List of encoded ids.
        :rtype: list[int]

        :examples:
        >>> tk.encode_text('what is the significance of indian history')
        >>> # Output: [10919, 318, 262, 12085, 286, 773, 666, 2106]
        """

        if allowed_special is None:
            allowed_special = set("<|endoftext|>")

        return torch.tensor(
            self.tokenizer.encode(text, allowed_special=set(allowed_special)),
            dtype=torch.int16,
        )

    def decode_ids(self, ids: torch.Tensor) -> str:
        """
        Method to decode the ids passed using the tokenizer initialized
        to get the encoded text back.
        Uses tiktoken.decode() method to achieve decoding.

        :param list[int] text: ids that needs to be decoded.
        :returns: retrieved string.
        :rtype: str

        :examples:
        >>> tk.decode_ids([10919, 318, 262, 12085, 286, 773, 666, 2106])
        >>> # Output: 'what is the significance of indian history'
        """

        return self.tokenizer.decode(ids.numpy().tolist())
