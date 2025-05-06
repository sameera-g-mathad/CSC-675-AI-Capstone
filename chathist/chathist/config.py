from .tokenizer import Tokenizer
import torch


class Config:
    """
    Class that holds all the configuration within
    the chathist folder. This includes setting tokenizer,
    flavor of gpt(ex: 124M, 1.5Betc), login into huggingface
    to name a few.
    """

    _tokenizer: Tokenizer | None = None
    _response_query: str | None = None
    _ignore_index: int = -100
    _endoftext: int = 50256
    _dtype = torch.uint16
    _device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # By default the repo is set as the intention of this
    # project is to build gpt2 model.
    _hugginface_user = "openai-community"
    _gpt_config: dict = {
        "gpt2": {
            "vocab_size": 50257,  # Vocabulary size
            "context_length": 1024,  # Context length
            "emb_dim": 768,  # Embedding dimension
            "n_heads": 12,  # Number of attention heads
            "n_layers": 12,  # Number of layers
            "drop_rate": 0.1,  # Dropout rate
            # "qkv_bias": False,
        },
        "gpt2-medium": {},
        "gpt2-large": {},
        "gpt2-xl": {
            "vocab_size": 50257,  # Vocabulary size
            "context_length": 1024,  # Context length
            "emb_dim": 1600,  # Embedding dimension
            "n_heads": 25,  # Number of attention heads
            "n_layers": 48,  # Number of layers
            "drop_rate": 0.1,
        },
    }
    _gpt_flavor: str = "gpt2-xl"

    def set_gpt_flavor(self, flavor: str = "gpt2-xl") -> None:
        """
        This method is to set the gpt2 flavor that instantiates
        all the weights as needed.

        :param str flavor: Flavor to retrieve. `["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]`
        are supported values as they serve as the repo names on hugginface as well.
        """
        if flavor not in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
            raise ValueError(
                f'{flavor} is not accpted!!! Please pass any one of ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]'
            )
        self._gpt_flavor = flavor

    @property
    def device(self):
        """
        Returns the device on which the data can be loaded and trained on.
        'cuda' or 'cpu' is returned depending on the availability
        """
        return self._device

    @property
    def gpt_flavor(self):
        """
        Returns the gpt config correponding to the set flavor.
        """
        return self._gpt_config[self._gpt_flavor]

    @property
    def huggingface_repo(self):
        """
        Property to return the default huggingface repo name,
        which is then used to download available weights.
        """
        if self._gpt_flavor is None:
            raise ValueError("Please set the flavor before.")
        return f"{self._hugginface_user}/{self._gpt_flavor}"

    def set_tokenizer(self, tokenizer: Tokenizer) -> None:
        """
        This method is used to set the tokenizer
        that will be used within all the files
        that requires tokenizer for converting
        vocab to tokens and vice-versa.

        :param chathist.Tokenizer tokenizer: The instance of tokenizer
        to be used throughout the training and inference.

        :rtype: None
        :returns: None
        """
        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> Tokenizer:
        """
        This method is used to return the
        set tokenizer wherever required
        within the project.

        :rtype: Tokenizer
        """
        if self._tokenizer is None:
            return Tokenizer()
        return self._tokenizer

    def _set_response_query(self, response_query: str) -> None:
        """
        This method sets the response query that is inputted
        in InstructionStyle while creating propmt styles.
        This is done so that it can be used everywhere, including
        in dataloaders whether to mask all the input and instructions
        including response query, so that the model only predicts the
        response and not the input again.

        :param str response_query: Response query that is inputed into
        Instruction Style.

        :rtype: None

        :examples:

        `Alpaca`: "### Response"

        `Phi3`: "<|assistant|>"
        """
        self._response_query = response_query

    @property
    def response_query(self) -> str:
        """
        Returns the stored response query back.
        """
        if self._response_query is None:
            raise ValueError("Response query is not set.")
        return self._response_query

    @property
    def response_ids(self) -> torch.Tensor:
        """
        Returns the ids of stored response query back.
        """
        if self._response_query is None:
            raise ValueError("Response query is not set.")

        return self.tokenizer.encode_text(self._response_query)

    def set_ingore_index(self, ignore_index):
        """
        Sets ignore index that is padded to response or targets
        so that it is excluded during loss calculation.
        This method is not needed to be honest, as the ignore_index
        has limited values that are supported by pytorch.

        `Use with caution!!!.` Please read the docstring and assign
        pytorch allowed ignore indeces.
        """
        self._ignore_index = ignore_index

    @property
    def ignore_index(self):
        """
        Returns the stored or default ignore index back.
        """
        return self._ignore_index

    def set_dtype(self, dtype: torch.dtype):
        """
        To set the datatype of tensors created across
        project. This is to ensure all the tensors have
        common datatype to prevent errors. By default,
        dtype = torch.unit16.

        :param torch.dtype dtype: Dtype to set tensors with.
        """
        self._dtype = dtype

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the set or default dtype to use.
        """
        return self._dtype

    def set_endoftext(self, endoftext: int):
        """
        This method is supposed to set the endoftext.
        There is no need to set this, as it is assumed
        that tiktoken with encoding of gpt2 is used and
        50256 is set by default. Adding here to match
        consistency as with previous methods.

        :param int endoftext: The value of endoftext to be
        used
        """
        self._endoftext = endoftext

    @property
    def endoftext(self):
        """
        Returns the set or default endoftext value.
        """
        return self._endoftext
