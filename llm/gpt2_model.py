import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2Model
import math
from huggingface_hub import login

GPT_CONFIG_1558M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 1600,  # Embedding dimension
    "n_heads": 25,  # Number of attention heads
    "n_layers": 48,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": True,  # Query-Key-Value bias
}


class LayerNorm(nn.Module):
    """Experimental"""

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        """Experimental"""
        x_mean = torch.mean(x, dim=-1, keepdim=True)
        x_std = torch.std(x, dim=-1, keepdim=True)
        x_norm = (x - x_mean) / (x_std + self.eps)
        return self.scale * x_norm + self.shift


class GeLU(nn.Module):
    """Experimental"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Experimental"""
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
    """Experimental"""

    def __init__(self, in_dim, rank, out_dim, alpha):
        super().__init__()
        self.a_matrix = nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.a_matrix, a=math.sqrt(5))
        self.b_matrix = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        """Experimental"""
        return self.alpha * (self.a_matrix @ self.b_matrix @ x)


class LinearLoRA(nn.Module):
    """Experimental"""

    def __init__(self, linear_layer: nn.Linear, rank, alpha):
        super().__init__()
        self.linear_layer = linear_layer
        in_features, out_features = linear_layer.in_features, linear_layer.out_features
        self.lora = LoRA(in_features, rank, out_features, alpha)

    def forward(self, x):
        """Experimental"""
        return self.linear_layer(x) + self.lora(x)


class MLPGPT2(nn.Module):
    """Experimental"""

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        out_dim = emb_dim * 4
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, out_dim), GeLU(), nn.Linear(out_dim, emb_dim)
        )

    def forward(self, x):
        """Experimental"""
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    """Experimental"""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.wq = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.wk = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.wv = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.out_form = nn.Linear(cfg["emb_dim"], cfg["emb_dim"])
        self.heads = cfg["n_heads"]
        self.head_dim = cfg["emb_dim"] // self.heads
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(cfg["context_length"], cfg["context_length"]), diagonal=1
            ),
        )

    def forward(self, x):
        """need to fix"""
        batch_size, num_tokens, _in_shape = x.shape
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
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(
            batch_size, num_tokens, self.cfg["emb_dim"]
        )

        return self.out_form(context_vec)


class TransformerBlock(nn.Module):
    """Experimental"""

    def __init__(self) -> None:
        super().__init__()
        self.norm1 = LayerNorm(GPT_CONFIG_1558M["emb_dim"])
        self.norm2 = LayerNorm(GPT_CONFIG_1558M["emb_dim"])
        self.mlp = MLPGPT2(GPT_CONFIG_1558M["emb_dim"])
        self.multi_head = MultiHeadAttention(GPT_CONFIG_1558M)
        self.dropout = nn.Dropout(GPT_CONFIG_1558M["drop_rate"])

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


class GPT2(nn.Module):
    """Experimental"""

    def __init__(self) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(
            GPT_CONFIG_1558M["vocab_size"], GPT_CONFIG_1558M["emb_dim"]
        )
        self.pos_emb = nn.Embedding(
            GPT_CONFIG_1558M["context_length"], GPT_CONFIG_1558M["emb_dim"]
        )
        self.dropout = nn.Dropout(GPT_CONFIG_1558M["drop_rate"])
        self.transformers = nn.Sequential(
            *[TransformerBlock() for _ in range(GPT_CONFIG_1558M["n_layers"])]
        )
        self.final_norm = LayerNorm(GPT_CONFIG_1558M["emb_dim"])
        self.final_layer = nn.Linear(
            GPT_CONFIG_1558M["emb_dim"], GPT_CONFIG_1558M["vocab_size"], bias=False
        )

    def forward(self, batch_x):
        """Experimental"""
        _, seq_length = batch_x.shape
        tok_embs = self.tok_emb(batch_x)
        pos_embs = self.pos_emb(
            torch.arange(
                seq_length, device="cuda" if torch.cuda.is_available() else "cpu"
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
        if left.shape != right.shape:
            raise ValueError(
                f"Shape mismatch. Left: {left.shape}, Right: {right.shape}"
            )
        return torch.nn.Parameter(right.clone().detach())

    def load_weights(self):
        model = "openai-community/gpt2-xl"
        gpt_hf = GPT2Model.from_pretrained(
            model,
            cache_dir="checkpoints",
        )
        # gpt_hf.eval()
        print("Loading weights")
        d = gpt_hf.state_dict()

        self.pos_emb.weight = self.assign_check(self.pos_emb.weight, d["wpe.weight"])
        self.tok_emb.weight = self.assign_check(self.tok_emb.weight, d["wte.weight"])

        for b in range(GPT_CONFIG_1558M["n_layers"]):
            q_w, k_w, v_w = np.split(d[f"h.{b}.attn.c_attn.weight"], 3, axis=-1)
            self.transformers[b].multi_head.wq.weight = self.assign_check(
                self.transformers[b].multi_head.wq.weight, q_w.T
            )
            self.transformers[b].multi_head.wk.weight = self.assign_check(
                self.transformers[b].multi_head.wk.weight, k_w.T
            )
            self.transformers[b].multi_head.wv.weight = self.assign_check(
                self.transformers[b].multi_head.wv.weight, v_w.T
            )

            q_b, k_b, v_b = np.split(d[f"h.{b}.attn.c_attn.bias"], 3, axis=-1)
            self.transformers[b].multi_head.wq.bias = self.assign_check(
                self.transformers[b].multi_head.wq.bias, q_b
            )
            self.transformers[b].multi_head.wk.bias = self.assign_check(
                self.transformers[b].multi_head.wk.bias, k_b
            )
            self.transformers[b].multi_head.wv.bias = self.assign_check(
                self.transformers[b].multi_head.wv.bias, v_b
            )

            self.transformers[b].multi_head.out_form.weight = self.assign_check(
                self.transformers[b].multi_head.out_form.weight,
                d[f"h.{b}.attn.c_proj.weight"].T,
            )
            self.transformers[b].multi_head.out_form.bias = self.assign_check(
                self.transformers[b].multi_head.out_form.bias,
                d[f"h.{b}.attn.c_proj.bias"],
            )

            self.transformers[b].mlp.layers[0].weight = self.assign_check(
                self.transformers[b].mlp.layers[0].weight, d[f"h.{b}.mlp.c_fc.weight"].T
            )
            self.transformers[b].mlp.layers[0].bias = self.assign_check(
                self.transformers[b].mlp.layers[0].bias, d[f"h.{b}.mlp.c_fc.bias"]
            )
            self.transformers[b].mlp.layers[2].weight = self.assign_check(
                self.transformers[b].mlp.layers[2].weight,
                d[f"h.{b}.mlp.c_proj.weight"].T,
            )
            self.transformers[b].mlp.layers[2].bias = self.assign_check(
                self.transformers[b].mlp.layers[2].bias, d[f"h.{b}.mlp.c_proj.bias"]
            )

            self.transformers[b].norm1.scale = self.assign_check(
                self.transformers[b].norm1.scale, d[f"h.{b}.ln_1.weight"]
            )
            self.transformers[b].norm1.shift = self.assign_check(
                self.transformers[b].norm1.shift, d[f"h.{b}.ln_1.bias"]
            )
            self.transformers[b].norm2.scale = self.assign_check(
                self.transformers[b].norm2.scale, d[f"h.{b}.ln_2.weight"]
            )
            self.transformers[b].norm2.shift = self.assign_check(
                self.transformers[b].norm2.shift, d[f"h.{b}.ln_2.bias"]
            )

            self.final_norm.scale = self.assign_check(
                self.final_norm.scale, d["ln_f.weight"]
            )
            self.final_norm.shift = self.assign_check(
                self.final_norm.shift, d["ln_f.bias"]
            )
            self.final_layer.weight = self.assign_check(
                self.final_layer.weight, d["wte.weight"]
            )
