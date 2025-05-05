from chathist.tokenizer import Tokenizer
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
    def response_query(self):
        """
        Returns the stored response query back.
        """
        return self._response_query

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
