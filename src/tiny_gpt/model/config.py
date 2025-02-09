from dataclasses import dataclass
from transformers import PretrainedConfig
import tiktoken

n_vocab = tiktoken.get_encoding("gpt2").n_vocab


@dataclass
class TinyGPTConfig(PretrainedConfig):
    model_type: str = "tiny_gpt"
    num_workers: int = 1
    epochs: int = 1
    learning_rate: float = 6e-4
    accumulation_steps: int = 10
    # clip gradients at this value, or disable if == 0.0
    clip_grad_max_norm: float = 1
    log_interval: int = 100
    checkpoint_interval: int = 100
    dtype: str = "float16"
    vocab_size: int = n_vocab
    batch_size: int = 16
    seq_len: int = 256
    n_embd: int = 768
    n_layers: int = 4
    n_heads: int = 4

    attention_is_fast: bool = True
    attention_dropout_p: float = 0.1

    ffn_dropout_p: float = 0.1
    ffn_activation_fn: str = "gelu"

    normalization: str = "layer"
    normalization_bias: bool = True

    embd_dropout_p: float = 0.1


print(TinyGPTConfig())
