import os
from abc import ABC, abstractmethod
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers.models.gpt2 import GPT2Model
import chathist


class EModule(nn.Module, ABC):
    """
    Extended Module class that is used to create
    custom modules for the model.
    """

    def __init__(self) -> None:
        super().__init__()
        # Set the random seed for reproducibility
        torch.manual_seed(123)

        # Set the configuration parameters.
        self._device = chathist.config.device
        self._gpt_flavor = chathist.config.gpt_flavor
        self._log = chathist.config.log
        self._model = chathist.config.huggingface_repo
        self._outdir = chathist.config.outdir
        self._use_lora = chathist.config.use_lora

        # The number of layers in the model.
        self._layers = self._gpt_flavor["n_layers"]

    def assign_nn_parameter(self, weights: torch.Tensor) -> torch.nn.Parameter:
        """
        Assigns the weights to a nn.Parameter. This method is used to copy the weights
        from the pretrained model to the custom module.

        :param torch.Tensor weights: weights to be assigned
        :return: nn.Parameter
        :rtype: torch.nn.Parameter
        """
        weights = nn.Parameter(weights.clone().detach())
        if self._use_lora:
            weights.requires_grad = False
        return weights

    @abstractmethod
    def load_weights(self, weights: OrderedDict | None):
        """
        This abstract method is implemented as needed by
        sub-classes. This method is used to load the weights
        from the pretrained model to the custom module.

        :param OrderedDict | None weights: weights to be loaded
        :raises ValueError: if weights is None
        :raises NotImplementedError: if the method is not implemented
        :return: None
        :rtype: None

        """
        raise NotImplementedError("Should be implemented by the subclasses.")


class LayerNorm(EModule):
    """
    This class is used for normalizing layers across columns
    (dim = -1) that was used in gpt2.
    Standardization = (x - x_bar) / std
    Creates scale and shift parameters, as the name
    indicates this is used to scale and shift inputs
    once normalization is done.
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This abstract method is implemented as needed by
        `nn.Module`.

        Normalization is calculated here.
        :param torch.Tensor x: inputs
        """
        x_mean = torch.mean(x, dim=-1, keepdim=True)
        x_std = torch.std(x, dim=-1, keepdim=True)
        x_norm = (x - x_mean) / (x_std + self.eps)

        # Scale and shift inputs by learning patterns which are best.
        return self.scale * x_norm + self.shift

    def load_weights(self, weights: OrderedDict | None) -> None:
        """
        Abstract method implemented as needed by `EModule`.
        This method is used to load the weights from the
        pretrained Linear Layer to the custom Linear Layer module.

        :param OrderedDict | None weights: weights to be loaded
        :raises ValueError: if weights is None
        :raises NotImplementedError: if the method is not implemented
        :return: None
        :rtype: None
        """

        if weights is None:
            raise ValueError("Weights passed should be an instance of OrderedDict")
        self.scale = self.assign_nn_parameter(weights["scale"])
        self.shift = self.assign_nn_parameter(weights["shift"])


class GeLU(nn.Module):
    """
    GeLU - Gaussian Error Linear Unit. This layer is used as the
    activation function in MLP (FF) layers inside the trasformer
    layers.
    GeLU(x) = 0.5 * x * [ 1 + tanh(√(2.0 / pi) * (x + 0.044715 * x ^ 3))]
    """

    def forward(self, x) -> torch.Tensor:
        """
        This abstract method is implemented as needed by
        `nn.Module`.

        Computes GeLU activation.
        """
        return (0.5 * x) * (
            1
            + torch.tanh(
                (
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class LoRA(nn.Module):
    """
    LoRA = Lower Rank Adaptation. This class is used
    for adding addtional weights to the existing
    model that can be finetuned. That is instead of
    finetuning the weight parameters of larger models
    that are large in shape (could be in million), additional
    smaller weights (could be in thousand) are added to the model
    and finetuned. This is helpful as finetuning a layer of the
    model would mean storing gradients of the same size in gpu.
    Instead smaller weights (here matrix A and B) are stored and trained.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()

        # LoRA rank and alpha are set in the config file.
        rank = chathist.config.lora_rank
        alpha = chathist.config.lora_alpha

        # Matrix A and B are initialized here.
        self.a_matrix = nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(
            self.a_matrix, a=torch.sqrt(torch.tensor(5)).item()
        )
        self.b_matrix = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This abstract method is implemented as needed by
        `nn.Module`.

        LoRA multiplication is done here.
        """
        return self.alpha * (x @ self.a_matrix @ self.b_matrix)


class LinearLoRA(EModule):
    """
    This layer replaces the regular linear layers in the model,
    that adds additional loRa weights.
    """

    def __init__(self, linear_layer: nn.Linear):
        super().__init__()

        self.linear_layer = linear_layer

        # self.linear_layer.weight.requires_grad = False
        # if self.linear_layer.bias is not None:
        #     self.linear_layer.bias.requires_grad = False

        in_features, out_features = linear_layer.in_features, linear_layer.out_features
        self.lora = LoRA(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This abstract method is implemented as needed by
        `nn.Module`.

        Final output = x * weight_linear + x * A @ B
        """
        return self.linear_layer(x) + self.lora(x)

    def load_weights(self, weights: OrderedDict | None):
        """
        Abstract method implemented as needed by `EModule`.
        This method is used to load the weights from the
        pretrained Linear Layer to the custom Linear Layer module.

        :param OrderedDict | None weights: weights to be loaded
        :raises ValueError: if weights is None
        :raises NotImplementedError: if the method is not implemented
        :return: None
        """
        if weights is None:
            raise ValueError("Weights passed should be an instance of OrderedDict")
        self.linear_layer.weight = self.assign_nn_parameter(weights["weights"])
        self.linear_layer.bias = self.assign_nn_parameter(weights["bias"])


class MLPGPT2(EModule):
    """
    The Gpt2 feedforward layer present in the transformer layers.
    In gpt2 the feedforward network has input and output layer
    of dimenision (emb_dim, emb_dim * 4) and (emb_dim * 4, emb_dim)
    and GeLU activation is mainly applied here to the input layers.
    """

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        out_dim = emb_dim * 4

        # This is the feedforward layer that is used in the
        # transformer layers.
        # The input layer is of dimension (emb_dim, emb_dim * 4)
        # and the output layer is of dimension (emb_dim * 4, emb_dim)
        self.layer1 = nn.Linear(emb_dim, out_dim)
        self.gelu = GeLU()
        self.layer2 = nn.Linear(out_dim, emb_dim)

        # This runs if LoRA is used in the model.
        if self._use_lora:
            self.use_lora()

    def use_lora(self):
        """
        To replace the linear layers with LoRA layers.
        This is used to finetune the model with LoRA weights.
        """
        if self._use_lora:
            self._log.info("Using LoRA weights for ff layers!!")

            if isinstance(self.layer1, nn.Linear):
                self.layer1 = LinearLoRA(self.layer1)

            if isinstance(self.layer2, nn.Linear):
                self.layer2 = LinearLoRA(self.layer2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This abstract method is implemented as needed by
        `nn.Module`.

        The inputs pass through this feedforward layers and
        returns the outputs of same dimensions as that of inputs.
        """
        return self.layer2(self.gelu(self.layer1(x)))

    def load_weights(self, weights: OrderedDict | None):
        """
        Abstract method implemented as needed by `EModule`.
        This method is used to load the weights from the
        pretrained MLPGPT2 layers to the custom MLPGPT2 module.

        :param OrderedDict | None weights: weights to be loaded
        :raises ValueError: if weights is None
        :raises NotImplementedError: if the method is not implemented
        :return: None
        :rtype: None
        """
        if weights is None:
            raise ValueError("Weights passed should be an instance of OrderedDict")

        # This runs if LoRA is used in the model. This is becuase,
        # we need to load the weights according to the config of HuggingFace
        # model.
        if (
            self._use_lora
            and isinstance(self.layer1, LinearLoRA)
            and isinstance(self.layer2, LinearLoRA)
        ):
            self.layer1.load_weights(
                OrderedDict(
                    {"weights": weights["c_fc_weights"], "bias": weights["c_fc_bias"]}
                )
            )

            self.layer2.load_weights(
                OrderedDict(
                    {
                        "weights": weights["c_proj_weights"],
                        "bias": weights["c_proj_bias"],
                    }
                )
            )
        # This runs for the saved model to load the weights.
        elif isinstance(self.layer1, nn.Linear) and isinstance(self.layer2, nn.Linear):
            self.layer1.load_state_dict(
                {
                    "weight": self.assign_nn_parameter(weights["c_fc_weights"]),
                    "bias": self.assign_nn_parameter(weights["c_fc_bias"]),
                }
            )
            self.layer2.load_state_dict(
                {
                    "weight": self.assign_nn_parameter(weights["c_proj_weights"]),
                    "bias": self.assign_nn_parameter(weights["c_proj_bias"]),
                }
            )


class MultiHeadAttention(EModule):
    """
    This class is for multi-head attention.
    The inputs pass through query, key and values
    to compute attention weights and final outputs
    where the model learns about attending other inputs
    in the sequence.
    """

    def __init__(self, _gpt_flavor: dict) -> None:
        super().__init__()
        self.emb_dim = _gpt_flavor["emb_dim"]

        # Initializing the weights for query, key and value weights.
        self.wq = nn.Linear(
            _gpt_flavor["emb_dim"],
            _gpt_flavor["emb_dim"],
        )
        #  bias=_gpt_flavor["qkv_bias"])
        self.wk = nn.Linear(
            _gpt_flavor["emb_dim"],
            _gpt_flavor["emb_dim"],
        )
        #  bias=_gpt_flavor["qkv_bias"])
        self.wv = nn.Linear(
            _gpt_flavor["emb_dim"],
            _gpt_flavor["emb_dim"],
        )
        #  bias=_gpt_flavor["qkv_bias"])

        # The output layer is of dimension (emb_dim, emb_dim)
        self.out_form = nn.Linear(_gpt_flavor["emb_dim"], _gpt_flavor["emb_dim"])

        # The number of heads is set in the config file.
        self.heads = _gpt_flavor["n_heads"]

        # The head dimension is emb_dim / n_heads
        self.head_dim = _gpt_flavor["emb_dim"] // self.heads

        self.dropout = nn.Dropout(_gpt_flavor["drop_rate"])

        # The mask is used to prevent the model from attending
        # to the future tokens in the sequence.
        self.mask = torch.triu(
            torch.ones(
                _gpt_flavor["context_length"],
                _gpt_flavor["context_length"],
                dtype=torch.bool,
                requires_grad=False,
                device=self._device,
            ),
            diagonal=1,
        )

        # This runs if LoRA is used in the model.
        if self._use_lora:
            self.use_lora()

    def use_lora(self):
        """
        To replace the linear layers with LoRA layers.
        This is used to finetune the model with LoRA weights.
        """
        if self._use_lora:
            self._log.info("Using LoRA weights for multi head layers!!")
            if isinstance(self.wq, torch.nn.modules.linear.Linear):
                self.wq = LinearLoRA(self.wq)
            if isinstance(self.wk, nn.Linear):
                self.wk = LinearLoRA(self.wk)
            if isinstance(self.wv, nn.Linear):
                self.wv = LinearLoRA(self.wv)
            if isinstance(self.out_form, nn.Linear):
                self.out_form = LinearLoRA(self.out_form)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This abstract method is implemented as needed by
        `nn.Module`.
        The inputs pass through query, key and value weights
        to compute attention weights and final outputs
        where the model learns about attending other inputs
        in the sequence.

        :param torch.Tensor x: inputs
        :return: outputs
        :rtype: torch.Tensor
        """
        batch_size, num_tokens, _ = x.shape

        # Multiplying the inputs with all the query, key and value weights.
        queries = self.wq(x)
        keys = self.wk(x)
        values = self.wv(x)

        # Reshaping the inputs to (batch_size, num_tokens, heads, head_dim)
        # where head_dim = emb_dim / n_heads
        # This is done to compute the attention scores.
        # The inputs are reshaped to (batch_size, num_tokens, heads, head_dim)
        queries = queries.view(batch_size, num_tokens, self.heads, self.head_dim)
        keys = keys.view(batch_size, num_tokens, self.heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.heads, self.head_dim)

        # The inputs are reshaped to (batch_size, heads, num_tokens, head_dim)
        # This is done to compute the attention scores.
        # The inputs are reshaped to (batch_size, heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # The attention scores are computed by multiplying the queries and keys.
        # The attention scores are reshaped to (batch_size, heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # The attention scores are scaled by the square root of the head dimension.
        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        # The attention weights are multiplied with the values to get the context vector.

        context_vec = (attn_weights @ values).transpose(1, 2)

        # The context vector is reshaped to (batch_size, num_tokens, heads, head_dim)
        context_vec = context_vec.contiguous().view(
            batch_size, num_tokens, self.emb_dim
        )

        return self.out_form(context_vec)

    def load_weights(self, weights: OrderedDict | None):
        """
        Abstract method implemented as needed by `EModule`.
        This method is used to load the weights from the
        pretrained MLPGPT2 layers to the custom MLPGPT2 module.

        :param OrderedDict | None weights: weights to be loaded
        :raises ValueError: if weights is None
        :raises NotImplementedError: if the method is not implemented
        :return: None
        :rtype: None
        """
        if weights is None:
            raise ValueError("Weights passed should be an instance of OrderedDict")
        conv1d_weights = weights["c_attn_weights"]
        conv1d_bias = weights["c_attn_bias"]
        c_proj_weight = weights["c_proj_weight"]
        c_proj_bias = weights["c_proj_bias"]

        # This runs if LoRA is used in the model. This is becuase,
        # we need to load the weights according to the config of HuggingFace
        # model.
        if (
            self.use_lora
            and isinstance(self.wq, LinearLoRA)
            and isinstance(self.wv, LinearLoRA)
            and isinstance(self.wk, LinearLoRA)
            and isinstance(self.out_form, LinearLoRA)
        ):
            self.wq.load_weights(
                OrderedDict({"weights": conv1d_weights[0].T, "bias": conv1d_bias[0]})
            )
            self.wk.load_weights(
                OrderedDict({"weights": conv1d_weights[1].T, "bias": conv1d_bias[1]})
            )
            self.wv.load_weights(
                OrderedDict({"weights": conv1d_weights[2].T, "bias": conv1d_bias[2]})
            )
            self.out_form.load_weights(
                OrderedDict({"weights": c_proj_weight.T, "bias": c_proj_bias})
            )
        # This runs for the saved model to load the weights.
        else:
            self.wq.load_state_dict(
                {
                    "weight": self.assign_nn_parameter(conv1d_weights[0].T),
                    "bias": self.assign_nn_parameter(conv1d_bias[0]),
                }
            )

            self.wk.load_state_dict(
                {
                    "weight": self.assign_nn_parameter(conv1d_weights[1].T),
                    "bias": self.assign_nn_parameter(conv1d_bias[1]),
                }
            )

            self.wv.load_state_dict(
                {
                    "weight": self.assign_nn_parameter(conv1d_weights[2].T),
                    "bias": self.assign_nn_parameter(conv1d_bias[2]),
                }
            )

            self.out_form.load_state_dict(
                {
                    "weight": self.assign_nn_parameter(c_proj_weight.T),
                    "bias": self.assign_nn_parameter(c_proj_bias),
                }
            )


class TransformerBlock(EModule):
    """Experimental"""

    def __init__(self, _gpt_flavor: dict, layer_num: int) -> None:
        super().__init__()
        self.layer_num = layer_num
        self.norm1 = LayerNorm(_gpt_flavor["emb_dim"])
        self.norm2 = LayerNorm(_gpt_flavor["emb_dim"])
        self.mlp = MLPGPT2(_gpt_flavor["emb_dim"])
        self.multi_head = MultiHeadAttention(_gpt_flavor)
        self.dropout = nn.Dropout(_gpt_flavor["drop_rate"])

    def forward(self, x):
        """
        This abstract method is implemented as needed by
        `nn.Module`.
        The inputs pass through the transformer layers and returns the outputs
        of same dimensions as that of inputs.

        :param torch.Tensor x: inputs
        :return: outputs
        :rtype: torch.Tensor
        """

        # First Phase
        x_copy = x
        x = self.norm1(x)
        x = self.multi_head(x)
        x = self.dropout(x)
        x += x_copy  # shortcut

        # Second Phase
        x_copy = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x += x_copy  # shortcut

        return x

    def load_weights(self, weights: OrderedDict | None):
        """
        Abstract method implemented as needed by `EModule`.
        This method is used to load the weights from the
        pretrained TransformerBlock layers to the custom TransformerBlock module.

        :param OrderedDict | None weights: weights to be loaded
        :raises ValueError: if weights is None
        :raises NotImplementedError: if the method is not implemented
        :return: None
        :rtype: None

        """
        if weights is None:
            raise ValueError("Weights passed should be an instance of OrderedDict")
        self._log.info("Layer: %s", self.layer_num + 1)

        # Layer Norm 1 weights and bias loading.
        self._log.info("loading norm1 weights and bias")
        self.norm1.load_weights(
            weights=OrderedDict(
                {
                    "scale": weights[f"h.{self.layer_num}.ln_1.weight"],
                    "shift": weights[f"h.{self.layer_num}.ln_1.bias"],
                }
            )
        )

        # Layer Norm 2 weights and bias loading.
        self._log.info("loading norm2 weights and bias")
        self.norm2.load_weights(
            weights=OrderedDict(
                {
                    "scale": weights[f"h.{self.layer_num}.ln_2.weight"],
                    "shift": weights[f"h.{self.layer_num}.ln_2.bias"],
                }
            )
        )

        # MLPGPT2 weights and bias loading.
        self._log.info("loading ff weights and bias")
        self.mlp.load_weights(
            weights=OrderedDict(
                {
                    "c_fc_weights": weights[f"h.{self.layer_num}.mlp.c_fc.weight"].T,
                    "c_proj_weights": weights[
                        f"h.{self.layer_num}.mlp.c_proj.weight"
                    ].T,
                    "c_fc_bias": weights[f"h.{self.layer_num}.mlp.c_fc.bias"],
                    "c_proj_bias": weights[f"h.{self.layer_num}.mlp.c_proj.bias"],
                }
            )
        )

        # Multi-head Attention weights and bias loading.
        self._log.info("loading attention weights and bias")
        self.multi_head.load_weights(
            weights=OrderedDict(
                {
                    "c_attn_weights": [
                        *np.split(
                            weights[f"h.{self.layer_num}.attn.c_attn.weight"],
                            3,
                            axis=-1,
                        )
                    ],
                    "c_attn_bias": [
                        *np.split(
                            weights[f"h.{self.layer_num}.attn.c_attn.bias"], 3, axis=-1
                        )
                    ],
                    "c_proj_weight": weights[f"h.{self.layer_num}.attn.c_proj.weight"],
                    "c_proj_bias": weights[f"h.{self.layer_num}.attn.c_proj.bias"],
                }
            )
        )


class GPT2(EModule):
    """
    GPT2 class that wraps Token Embedding, Positional Embedding, Transformer
    layers.
    """

    def __init__(self) -> None:
        super().__init__()

        # Token Embedding and Positional Embedding are initialized here.
        self.tok_emb = nn.Embedding(
            self._gpt_flavor["vocab_size"], self._gpt_flavor["emb_dim"]
        )
        self.pos_emb = nn.Embedding(
            self._gpt_flavor["context_length"], self._gpt_flavor["emb_dim"]
        )

        # The dropout layer is initialized here. Dropout is same as the one used in
        # the HuggingFace model of GPT2.
        self.dropout = nn.Dropout(self._gpt_flavor["drop_rate"])

        # The transformer layers are initialized here.
        self.transformers = nn.Sequential(
            OrderedDict(
                (f"layer_{i}", TransformerBlock(self._gpt_flavor, i))
                for i in range(self._gpt_flavor["n_layers"])
            )
        )

        # The final layer normalization and linear layer are initialized here.
        self.final_norm = LayerNorm(self._gpt_flavor["emb_dim"])
        self.final_layer = nn.Linear(
            self._gpt_flavor["emb_dim"], self._gpt_flavor["vocab_size"], bias=False
        )

        # This runs if LoRA is used in the model. This loads the final layer
        # with LoRA weights.
        if self._use_lora:
            if isinstance(self.final_layer, nn.Linear):
                self._log.info("Using Lora for final output layer!!")
                self.final_layer = LinearLoRA(self.final_layer)

    def forward(self, batch_x):
        """
        This abstract method is implemented as needed by
        `nn.Module`.
        The inputs pass through the transformer layers and returns the outputs
        of same dimensions as that of inputs.

        :param torch.Tensor batch_x: inputs
        :return: outputs
        :rtype: torch.Tensor
        :raises NotImplementedError: if the method is not implemented
        :raises ValueError: if batch_x is None
        :raises TypeError: if batch_x is not a torch.Tensor
        :raises RuntimeError: if batch_x is not a 2D tensor
        """
        _, seq_length = batch_x.shape
        tok_embs = self.tok_emb(batch_x)  # token embedding

        pos_embs = self.pos_emb(torch.arange(seq_length, device=self._device))
        x = tok_embs + pos_embs  # positional embedding
        x = self.dropout(x)
        x = self.transformers(x)
        x = self.final_norm(x)
        x = self.final_layer(x)
        return x

    def get_model_params(self, verbose: bool = True) -> dict:
        """
        This method is used to get the model parameters.
        The method returns the total number of parameters and
        the total number of trainable parameters in the model.
        The method also returns the model parameters in a dataframe
        if verbose is True.

        :param bool verbose: if True, returns the model parameters
        """
        total_params = 0
        total_trainable = 0
        records = []
        for name, layer in self.named_parameters():
            if verbose:
                records.append(
                    {
                        "layer_name": name,
                        "requires_grad": layer.requires_grad,
                        "params": layer.numel(),
                    }
                )
            layer_params = layer.numel()
            if layer.requires_grad:
                total_trainable += layer_params
            total_params += layer_params
        param_info: dict = {
            "total_params": total_params,
            "total_trainable": total_trainable,
        }
        if verbose:
            param_info["df"] = pd.DataFrame(records)

        return param_info

    def download_model(self) -> GPT2Model:
        """
        This method is used to download the model from HuggingFace.
        The method checks if the model is already downloaded
        in the specified directory. If not, it downloads the model
        from HuggingFace and saves it in the specified directory.
        The method returns the model object.

        :return: model object
        :rtype: GPT2Model
        """
        self._log.info("Repo: %s.", self._model)
        path_exists = False
        if os.path.exists(self._outdir):
            path_exists = True

        # Check if the model is already downloaded
        if not path_exists:
            self._log.info("Downloading %s from Hugginface...", self._model)
        else:
            self._log.info(
                "%s exists in %s. Skipping Download...", self._model, self._outdir
            )

        # Download the model from HuggingFace
        gpt_hf = GPT2Model.from_pretrained(
            self._model, cache_dir=self._outdir, use_safetensors=True, from_tf=False
        )

        if not path_exists:
            self._log.info("Dowload complete.")

        return gpt_hf

    def load_weights(self, _=None):
        """
        This method is used to load the weights from the pretrained
        model to the custom model. The method also loads the weights
        from the pretrained model to the custom model if LoRA is used.
        """
        # Check and download the model from HuggingFace
        gpt_hf = self.download_model()

        # Set the model to evaluation mode as we need to load the weights only.
        gpt_hf.eval()
        self._log.info("Loading weights into model.")

        # Get the pretrained weights from the model as a OrderedDict.
        gpt_pretrained_weights = gpt_hf.state_dict()

        # If LoRA is used, we need to add additional LoRA weights to the model.
        if self._use_lora:
            self._log.warning(
                "Using loRA will freeze the whole model except for loRA weights."
            )

        # Load embedding weights for token and positional embedding.
        self.pos_emb.weight = self.assign_nn_parameter(
            gpt_pretrained_weights["wpe.weight"]
        )

        self.tok_emb.weight = self.assign_nn_parameter(
            gpt_pretrained_weights["wte.weight"]
        )

        # Load final normalization layer weights.
        self.final_norm.load_weights(
            weights=OrderedDict(
                {
                    "scale": gpt_pretrained_weights["ln_f.weight"],
                    "shift": gpt_pretrained_weights["ln_f.bias"],
                }
            )
        )

        # Load final linear layer weights with LoRA or without LoRA
        # depending on the config.
        if self._use_lora and isinstance(self.final_layer, LinearLoRA):
            self.final_layer.linear_layer.weight = self.assign_nn_parameter(
                gpt_pretrained_weights["wte.weight"]
            )
        else:
            self.final_layer.load_state_dict(
                {
                    "weight": self.assign_nn_parameter(
                        gpt_pretrained_weights["wte.weight"]
                    )
                }
            )

        # Load transformer layers weights.
        for layer in range(self._layers):
            transfomer = self.transformers[layer]
            if isinstance(transfomer, TransformerBlock) and isinstance(
                gpt_pretrained_weights, OrderedDict
            ):
                transfomer.load_weights(gpt_pretrained_weights)
