from abc import ABC, abstractmethod
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from transformers.models.gpt2 import GPT2Model
import chathist


class EModule(nn.Module, ABC):
    """
    Experimental
    """

    def __init__(self) -> None:
        super().__init__()

    def assign_nn_parameter(self, weights: torch.Tensor) -> torch.nn.Parameter:
        """
        Experimental
        """
        return nn.Parameter(weights.clone().detach())

    @abstractmethod
    def load_weights(self, weights: OrderedDict | None):
        """
        Experimental
        """
        pass


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
        """Experimental"""

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

    def __init__(self, in_dim, rank, out_dim, alpha):
        super().__init__()
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
        return self.alpha * (self.a_matrix @ self.b_matrix @ x)


class LinearLoRA(nn.Module):
    """
    This layer replaces the regular linear layers in the model,
    that adds additional loRa weights.
    """

    def __init__(self, linear_layer: nn.Linear, rank, alpha):
        super().__init__()
        self.linear_layer = linear_layer
        in_features, out_features = linear_layer.in_features, linear_layer.out_features
        self.lora = LoRA(in_features, rank, out_features, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This abstract method is implemented as needed by
        `nn.Module`.

        Final output = x * weight_linear + x * A @ B
        """
        return self.linear_layer(x) + self.lora(x)


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
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, out_dim), GeLU(), nn.Linear(out_dim, emb_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This abstract method is implemented as needed by
        `nn.Module`.

        The inputs pass through this feedforward layers and
        returns the outputs of same dimensions as that of inputs.
        """
        return self.layers(x)

    def load_weights(self, weights: OrderedDict | None):
        """
        Experimental
        """
        if weights is None:
            raise ValueError("Weights passed should be an instance of OrderedDict")
        self.layers[0].weight = self.assign_nn_parameter((weights["c_fc_weights"]))
        self.layers[0].bias = self.assign_nn_parameter(weights["c_fc_bias"])
        self.layers[2].weight = self.assign_nn_parameter(weights["c_proj_weights"])
        self.layers[2].bias = self.assign_nn_parameter(weights["c_proj_bias"])


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
        self.out_form = nn.Linear(_gpt_flavor["emb_dim"], _gpt_flavor["emb_dim"])
        self.heads = _gpt_flavor["n_heads"]
        self.head_dim = _gpt_flavor["emb_dim"] // self.heads
        self.dropout = nn.Dropout(_gpt_flavor["drop_rate"])
        self.mask = torch.triu(
            torch.ones(
                _gpt_flavor["context_length"],
                _gpt_flavor["context_length"],
                dtype=torch.bool,
                requires_grad=False,
            ),
            diagonal=1,
        )
        # self.register_buffer(
        #     "mask",
        #     torch.triu(
        #         torch.ones(_gpt_flavor["context_length"], _gpt_flavor["context_length"]),
        #         diagonal=1,
        #     ),
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Experimental"""
        batch_size, num_tokens, _ = x.shape
        queries = self.wq(x)
        keys = self.wk(x)
        values = self.wv(x)

        queries = queries.view(batch_size, num_tokens, self.heads, self.head_dim)
        keys = keys.view(batch_size, num_tokens, self.heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.heads, self.head_dim)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(
            batch_size, num_tokens, self.emb_dim
        )

        return self.out_form(context_vec)

    def load_weights(self, weights: OrderedDict | None):
        """
        Experimental
        """
        if weights is None:
            raise ValueError("Weights passed should be an instance of OrderedDict")
        conv1d_weights = weights["c_attn_weights"]
        conv1d_bias = weights["c_attn_bias"]
        c_proj_weight = weights["c_proj_weight"]
        c_proj_bias = weights["c_proj_bias"]
        self.wq.weight = self.assign_nn_parameter(conv1d_weights[0].T)
        self.wq.bias = self.assign_nn_parameter(conv1d_bias[0])

        self.wk.weight = self.assign_nn_parameter(conv1d_weights[1].T)
        self.wk.bias = self.assign_nn_parameter(conv1d_bias[1])

        self.wv.weight = self.assign_nn_parameter(conv1d_weights[2].T)
        self.wv.bias = self.assign_nn_parameter(conv1d_bias[2])

        self.out_form.weight = self.assign_nn_parameter(c_proj_weight.T)
        self.out_form.bias = self.assign_nn_parameter(c_proj_bias)


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
        """Experimental"""

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
        if weights is None:
            raise ValueError("Weights passed should be an instance of OrderedDict")
        print(f"Layer: {self.layer_num + 1}\n")
        print("loading norm1 weights and bias")
        self.norm1.load_weights(
            weights=OrderedDict(
                {
                    "scale": weights[f"h.{self.layer_num}.ln_1.weight"],
                    "shift": weights[f"h.{self.layer_num}.ln_1.bias"],
                }
            )
        )
        print("loading norm2 weights and bias")
        self.norm2.load_weights(
            weights=OrderedDict(
                {
                    "scale": weights[f"h.{self.layer_num}.ln_2.weight"],
                    "shift": weights[f"h.{self.layer_num}.ln_2.bias"],
                }
            )
        )
        print("loading ff weights and bias")
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
        print("loading attention weights and bias")
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
        print("\n")


class GPT2(EModule):
    """
    GPT2 class that wraps Token Embedding, Positional Embedding, Transformer
    layers.
    """

    def __init__(self) -> None:
        super().__init__()
        _gpt_flavor = chathist.config.gpt_flavor
        self.layers = _gpt_flavor["n_layers"]
        self.tok_emb = nn.Embedding(_gpt_flavor["vocab_size"], _gpt_flavor["emb_dim"])
        self.pos_emb = nn.Embedding(
            _gpt_flavor["context_length"], _gpt_flavor["emb_dim"]
        )
        self.dropout = nn.Dropout(_gpt_flavor["drop_rate"])
        self.transformers = nn.Sequential(
            OrderedDict(
                (f"layer_{i}", TransformerBlock(_gpt_flavor, i))
                for i in range(_gpt_flavor["n_layers"])
            )
        )
        self.final_norm = LayerNorm(_gpt_flavor["emb_dim"])
        self.final_layer = nn.Linear(
            _gpt_flavor["emb_dim"], _gpt_flavor["vocab_size"], bias=False
        )

    def forward(self, batch_x):
        """Experimental"""
        _, seq_length = batch_x.shape
        tok_embs = self.tok_emb(batch_x)
        pos_embs = self.pos_emb(
            torch.arange(
                seq_length,
                # Need to change this
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        )

        x = tok_embs + pos_embs
        x = self.dropout(x)
        x = self.transformers(x)
        x = self.final_norm(x)
        x = self.final_layer(x)
        return x

    def get_model_params(self):
        """Experimental"""
        total = 0
        for layer in self.parameters():
            total += layer.numel()

        return total

    def assign_check(self, left, right):
        """
        Experimental
        """
        if left.shape != right.shape:
            raise ValueError(
                f"Shape mismatch. Left: {left.shape}, Right: {right.shape}"
            )
        return torch.nn.Parameter(right.clone().detach())

    def load_weights(self, weights=None):
        """
        Experimental
        """
        weights = None
        model = chathist.config.huggingface_repo
        print(f"Repo: {model}.")
        print("Downloading model from Hugginface...")
        gpt_hf = GPT2Model.from_pretrained(
            model,
            cache_dir="./../../checkpoints",
            # cache_dir=f"checkpoints_{model.split('/')[1]}",
        )
        print("Dowload complete.")
        gpt_hf.eval()
        print("Loading weights into model.")
        gpt_pretrained_weights = gpt_hf.state_dict()
        self.pos_emb.weight = self.assign_nn_parameter(
            gpt_pretrained_weights["wpe.weight"]
        )

        self.tok_emb.weight = self.assign_nn_parameter(
            gpt_pretrained_weights["wte.weight"]
        )
        self.final_norm.load_weights(
            weights=OrderedDict(
                {
                    "scale": gpt_pretrained_weights["ln_f.weight"],
                    "shift": gpt_pretrained_weights["ln_f.bias"],
                }
            )
        )
        self.final_layer.weight = self.assign_nn_parameter(
            gpt_pretrained_weights["wte.weight"]
        )

        for layer in range(self.layers):
            transfomer = self.transformers[layer]
            if isinstance(transfomer, TransformerBlock) and isinstance(
                gpt_pretrained_weights, OrderedDict
            ):
                transfomer.load_weights(gpt_pretrained_weights)

        # for b in range(self.layers):
        #     conv1d_weights = np.split(
        #         gpt_pretrained_weights[f"h.{b}.attn.c_attn.weight"], 3, axis=-1
        #     )

        #     self.transformers[b].multi_head.wq.weight = self.assign_check(
        #         self.transformers[b].multi_head.wq.weight, conv1d_weights[0].T
        #     )
        #     self.transformers[b].multi_head.wk.weight = self.assign_check(
        #         self.transformers[b].multi_head.wk.weight, conv1d_weights[1].T
        #     )
        #     self.transformers[b].multi_head.wv.weight = self.assign_check(
        #         self.transformers[b].multi_head.wv.weight, conv1d_weights[2].T
        #     )

        #     conv1d_bias = np.split(
        #         gpt_pretrained_weights[f"h.{b}.attn.c_attn.bias"], 3, axis=-1
        #     )
        #     self.transformers[b].multi_head.wq.bias = self.assign_check(
        #         self.transformers[b].multi_head.wq.bias, conv1d_bias[0]
        #     )
        #     self.transformers[b].multi_head.wk.bias = self.assign_check(
        #         self.transformers[b].multi_head.wk.bias, conv1d_bias[1]
        #     )
        #     self.transformers[b].multi_head.wv.bias = self.assign_check(
        #         self.transformers[b].multi_head.wv.bias, conv1d_bias[2]
        #     )

        #     self.transformers[b].multi_head.out_form.weight = self.assign_check(
        #         self.transformers[b].multi_head.out_form.weight,
        #         gpt_pretrained_weights[f"h.{b}.attn.c_proj.weight"].T,
        #     )
        #     self.transformers[b].multi_head.out_form.bias = self.assign_check(
        #         self.transformers[b].multi_head.out_form.bias,
        #         gpt_pretrained_weights[f"h.{b}.attn.c_proj.bias"],
        #     )

        #     self.transformers[b].mlp.layers[0].weight = self.assign_check(
        #         self.transformers[b].mlp.layers[0].weight,
        #         gpt_pretrained_weights[f"h.{b}.mlp.c_fc.weight"].T,
        #     )
        #     self.transformers[b].mlp.layers[0].bias = self.assign_check(
        #         self.transformers[b].mlp.layers[0].bias,
        #         gpt_pretrained_weights[f"h.{b}.mlp.c_fc.bias"],
        #     )
        #     self.transformers[b].mlp.layers[2].weight = self.assign_check(
        #         self.transformers[b].mlp.layers[2].weight,
        #         gpt_pretrained_weights[f"h.{b}.mlp.c_proj.weight"].T,
        #     )
        #     self.transformers[b].mlp.layers[2].bias = self.assign_check(
        #         self.transformers[b].mlp.layers[2].bias,
        #         gpt_pretrained_weights[f"h.{b}.mlp.c_proj.bias"],
        #     )

        #     self.transformers[b].norm1.scale = self.assign_check(
        #         self.transformers[b].norm1.scale,
        #         gpt_pretrained_weights[f"h.{b}.ln_1.weight"],
        #     )
        #     self.transformers[b].norm1.shift = self.assign_check(
        #         self.transformers[b].norm1.shift,
        #         gpt_pretrained_weights[f"h.{b}.ln_1.bias"],
        #     )
        #     self.transformers[b].norm2.scale = self.assign_check(
        #         self.transformers[b].norm2.scale,
        #         gpt_pretrained_weights[f"h.{b}.ln_2.weight"],
        #     )
        #     self.transformers[b].norm2.shift = self.assign_check(
        #         self.transformers[b].norm2.shift,
        #         gpt_pretrained_weights[f"h.{b}.ln_2.bias"],
        #     )
