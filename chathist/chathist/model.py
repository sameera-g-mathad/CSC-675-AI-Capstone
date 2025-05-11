import torch
import chathist
from chathist import GPT2


class Model:
    """
    Experimental
    """

    def __init__(self):
        self._tokenizer = chathist.config.tokenizer
        self._model = GPT2()
        self._model.load_weights()

    # def

    def generate(self, prompt: str) -> str:
        """Experimental"""
        token_ids = self._tokenizer.encode_text(prompt)
        token_ids = torch.unsqueeze(token_ids, dim=0)
        self._model.eval()
        for _ in range(30):
            with torch.inference_mode():
                logits = self._model(token_ids)  # (1, token_len, emb_dim)
                last_token = logits[:, -1, :]  # Get the last embedding
                last_token = torch.softmax(
                    last_token, dim=-1
                )  # Making sure that all the logits are between 0 and 1.
                token = torch.argmax(last_token, dim=-1, keepdim=True)
                # print(token.shape, token_ids.shape)
                token_ids = torch.cat((token_ids, token), dim=-1)

        return self._tokenizer.decode_ids(token_ids.squeeze(dim=0))
