import logging
from omegaconf import DictConfig, OmegaConf
from .tokenizer import Tokenizer
import torch
from hydra import initialize, compose

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Config:
    """
    Class that holds all the configuration within
    the chathist folder. This includes setting tokenizer,
    flavor of gpt(ex: 124M, 1.5Betc), login into huggingface
    to name a few.
    """

    _log: logging.Logger
    _tokenizer: Tokenizer | None = None
    _ignore_index: int = -100
    _endoftext: int = 50256
    _dtype = torch.int32
    _device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Config settings
    _model_name: str
    _model_path: str
    _save_as: str
    _save_path: str

    # Lora settings
    _use_lora: bool
    _lora_rank: int
    _lora_alpha: float

    # Training settings
    _epochs: int = 3
    _learning_rate: float = 0.01

    # DataLoader settings
    _batch_size: int = 16
    _shuffle: bool = False
    _drop_last: bool = True
    _mask_input: bool = True

    # Style settings
    _style_name: str = "alpaca"
    _input_query: str = ""
    _response_query: str = ""
    _prompt: str = ""
    _input_col: str = ""
    _response_col: str = ""
    _output_col: str = ""
    _new_df: bool = True

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
        },
        "gpt2-medium": {
            "vocab_size": 50257,
            "context_length": 1024,
            "emb_dim": 1024,
            "n_heads": 16,
            "n_layers": 24,
            "drop_rate": 0.1,
        },
        "gpt2-large": {
            "vocab_size": 50257,
            "context_length": 1024,
            "emb_dim": 1280,
            "n_heads": 20,
            "n_layers": 36,
            "drop_rate": 0.1,
        },
        "gpt2-xl": {
            "vocab_size": 50257,
            "context_length": 1024,
            "emb_dim": 1600,
            "n_heads": 25,
            "n_layers": 48,
            "drop_rate": 0.1,
        },
    }
    _gpt_flavor: str = "gpt2-xl"

    def __init__(
        self,
        config_path: str = "conf",
        config_name: str = "config",
        defaults: str = "train",
    ):
        """
        Experimental
        """
        self._log = logging.getLogger(__name__)
        with initialize(config_path=config_path, version_base=None):
            self._cfg = compose(config_name=config_name)

        if defaults in self._cfg:
            OmegaConf.set_struct(self._cfg, False)
            self._cfg.update(self._cfg.pop(defaults))
            OmegaConf.set_struct(self._cfg, True)

        self._initialize()

    def _compare_dict(
        self, dictionary: DictConfig, key: str, raise_error: bool = False
    ) -> bool:
        """
        Experimental
        """
        if key not in dictionary:
            if raise_error:
                raise ValueError(
                    f"{key} is not present in the config file which is required!!!"
                )
            self._log.warning(
                "%s is not present in the config file, falling back to defaults!!!",
                key,
            )
            return False
        return True

    def _initialize(self):
        """
        Experimental
        """
        if self._compare_dict(self._cfg, "config"):
            if self._compare_dict(self._cfg["config"], "ignore_index"):
                self._ignore_index = self._cfg["config"]["ignore_index"]

            if self._compare_dict(self._cfg["config"], "endoftext"):
                self._endoftext = self._cfg["config"]["endoftext"]

            # if self._compare_dict(self._cfg["config"], "dtype"):
            #     self._dtype = self._cfg["config"]["dtype"]

            # if self._compare_dict(self._cfg["config"], "device"):
            #     self._device = self._cfg["config"]["device"]

            if self._compare_dict(self._cfg["config"], "gpt_flavor"):
                self._gpt_flavor = self._cfg["config"]["gpt_flavor"]

            if self._compare_dict(self._cfg["config"], "model_name", raise_error=True):
                self._model_name = self._cfg["config"]["model_name"]

            if self._compare_dict(self._cfg["config"], "model_path", raise_error=True):
                self._model_path = self._cfg["config"]["model_path"]

            if self._compare_dict(self._cfg["config"], "save_as", raise_error=True):
                self._save_as = self._cfg["config"]["save_as"]

            if self._compare_dict(self._cfg["config"], "save_path", raise_error=True):
                self._save_path = self._cfg["config"]["save_path"]

        if self._compare_dict(self._cfg, "lora"):
            if self._compare_dict(self._cfg["lora"], "use_lora"):
                self._use_lora = self._cfg["lora"]["use_lora"]

            if self._compare_dict(self._cfg["lora"], "rank"):
                self._lora_rank = self._cfg["lora"]["rank"]

            if self._compare_dict(self._cfg["lora"], "alpha"):
                self._lora_alpha = self._cfg["lora"]["alpha"]

        if self._compare_dict(self._cfg, "train"):
            if self._compare_dict(self._cfg["train"], "epochs", raise_error=True):
                self._epochs = self._cfg["train"]["epochs"]

            if self._compare_dict(
                self._cfg["train"], "learning_rate", raise_error=True
            ):
                self._learning_rate = self._cfg["train"]["learning_rate"]

        if self._compare_dict(self._cfg, "data"):
            if self._compare_dict(self._cfg["data"], "batch_size", raise_error=True):
                self._batch_size = self._cfg["data"]["batch_size"]

            if self._compare_dict(self._cfg["data"], "shuffle", raise_error=True):
                self._shuffle = self._cfg["data"]["shuffle"]

            if self._compare_dict(self._cfg["data"], "drop_last", raise_error=True):
                self._drop_last = self._cfg["data"]["drop_last"]

            if self._compare_dict(self._cfg["data"], "mask_input", raise_error=True):
                self._mask_input = self._cfg["data"]["mask_input"]

        if self._compare_dict(self._cfg, "style"):
            if self._compare_dict(self._cfg["style"], "style_name", raise_error=True):
                self._style_name = self._cfg["style"]["style_name"]

            if self._compare_dict(self._cfg["style"], "prompt", raise_error=True):
                self._prompt = self._cfg["style"]["prompt"]

            if self._compare_dict(self._cfg["style"], "input_query", raise_error=True):
                self._input_query = self._cfg["style"]["input_query"]

            if self._compare_dict(
                self._cfg["style"], "response_query", raise_error=True
            ):
                self._response_query = self._cfg["style"]["response_query"]

            if self._compare_dict(self._cfg["style"], "input_col", raise_error=True):
                self._input_col = self._cfg["style"]["input_col"]

            if self._compare_dict(self._cfg["style"], "response_col", raise_error=True):
                self._response_col = self._cfg["style"]["response_col"]

            if self._compare_dict(self._cfg["style"], "output_col", raise_error=True):
                self._output_col = self._cfg["style"]["output_col"]

            if self._compare_dict(self._cfg["style"], "new_df", raise_error=True):
                self._new_df = self._cfg["style"]["new_df"]

    @property
    def log(self):
        """
        Experimental
        """
        return self._log

    @property
    def tokenizer(self) -> Tokenizer:
        """
        This method is used to return the
        set tokenizer wherever required
        within the project.

        :rtype: Tokenizer
        """
        if self._tokenizer is None:
            self._tokenizer = Tokenizer()
        return self._tokenizer

    @property
    def device(self) -> str:
        """
        Returns the device on which the data can be loaded and trained on.
        'cuda' or 'cpu' is returned depending on the availability
        """
        return self._device

    @property
    def gpt_flavor(self) -> dict:
        """
        Returns the gpt config correponding to the set flavor.
        """
        return self._gpt_config[self._gpt_flavor]

    @property
    def huggingface_repo(self) -> str:
        """
        Property to return the default huggingface repo name,
        which is then used to download available weights.
        """
        return f"{self._hugginface_user}/{self._gpt_flavor}"

    @property
    def outdir(self) -> str:
        """
        Experimental
        """
        return f"{self._model_path}/{self._model_name}"

    @property
    def response_ids(self) -> torch.Tensor:
        """
        Returns the ids of stored response query back.
        """
        if self._response_query is None:
            raise ValueError("Response query is not set.")

        return self.tokenizer.encode_text(self._response_query)

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the set or default dtype to use.
        """
        return self._dtype

    @property
    def endoftext(self) -> int:
        """
        Returns the set or default endoftext value.
        """
        return self._endoftext

    @property
    def ignore_index(self) -> int:
        """
        Returns the stored or default ignore index back.
        """
        return self._ignore_index

    @property
    def use_lora(self):
        """
        Experimental
        """
        return self._use_lora

    @property
    def lora_alpha(self):
        """
        Experimental
        """
        return self._lora_alpha

    @property
    def lora_rank(self):
        """
        Experimental
        """
        return self._lora_rank

    @property
    def lr(self):
        """Experiment"""
        return self._learning_rate

    @property
    def epochs(self):
        """Experiment"""
        return self._epochs

    @property
    def batch_size(self):
        """Experiment"""
        return self._batch_size

    @property
    def drop_last(self):
        """Experiment"""
        return self._drop_last

    @property
    def shuffle(self):
        """Experiment"""
        return self._shuffle

    @property
    def mask_input(self):
        """Experiment"""
        return self._mask_input

    @property
    def style_name(self):
        """Experiment"""
        return self._style_name

    @property
    def prompt(self):
        """Experiment"""
        return self._prompt

    @property
    def input_query(self):
        """Experiment"""
        return self._input_query

    @property
    def response_query(self):
        """Experiment"""
        return self._response_query

    @property
    def input_col(self):
        """Experiment"""
        return self._input_col

    @property
    def response_col(self):
        """Experiment"""
        return self._response_col

    @property
    def output_col(self):
        """Experiment"""
        return self._output_col

    @property
    def new_df(self):
        """Experiment"""
        return self._new_df

    def set_dtype(self, dtype: torch.dtype):
        """
        To set the datatype of tensors created across
        project. This is to ensure all the tensors have
        common datatype to prevent errors. By default,
        dtype = torch.unit16.

        :param torch.dtype dtype: Dtype to set tensors with.
        """
        self._dtype = dtype

    # def _set_response_query(self, response_query: str) -> None:
    #     """
    #     This method sets the response query that is inputted
    #     in InstructionStyle while creating propmt styles.
    #     This is done so that it can be used everywhere, including
    #     in dataloaders whether to mask all the input and instructions
    #     including response query, so that the model only predicts the
    #     response and not the input again.

    #     :param str response_query: Response query that is inputed into
    #     Instruction Style.

    #     :rtype: None

    #     :examples:

    #     `Alpaca`: "### Response"

    #     `Phi3`: "<|assistant|>"
    #     """
    #     self._response_query = response_query

    # def set_gpt_flavor(self, flavor: str = "gpt2-xl") -> None:
    #     """
    #     This method is to set the gpt2 flavor that instantiates
    #     all the weights as needed.

    #     :param str flavor: Flavor to retrieve. `["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]`
    #     are supported values as they serve as the repo names on hugginface as well.
    #     """
    #     if flavor not in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
    #         raise ValueError(
    #             f'{flavor} is not accpted!!! Please pass any one of ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]'
    #         )
    #     self._gpt_flavor = flavor
    # @DeprecationWarning
    # def set_tokenizer(self, tokenizer: Tokenizer) -> None:
    #     """
    #     This method is used to set the tokenizer
    #     that will be used within all the files
    #     that requires tokenizer for converting
    #     vocab to tokens and vice-versa.

    #     :param chathist.Tokenizer tokenizer: The instance of tokenizer
    #     to be used throughout the training and inference.

    #     :rtype: None
    #     :returns: None
    #     """
    #     self._tokenizer = tokenizer

    # @DeprecationWarning
    # def set_ingore_index(self, ignore_index):
    #     """
    #     Sets ignore index that is padded to response or targets
    #     so that it is excluded during loss calculation.
    #     This method is not needed to be honest, as the ignore_index
    #     has limited values that are supported by pytorch.

    #     `Use with caution!!!.` Please read the docstring and assign
    #     pytorch allowed ignore indeces.
    #     """
    #     self._ignore_index = ignore_index

    # @DeprecationWarning

    # @DeprecationWarning
    # def set_endoftext(self, endoftext: int):
    #     """
    #     This method is supposed to set the endoftext.
    #     There is no need to set this, as it is assumed
    #     that tiktoken with encoding of gpt2 is used and
    #     50256 is set by default. Adding here to match
    #     consistency as with previous methods.

    #     :param int endoftext: The value of endoftext to be
    #     used
    #     """
    #     self._endoftext = endoftext
