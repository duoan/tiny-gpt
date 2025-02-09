from dataclasses import dataclass
from transformers import PretrainedConfig
import tiktoken

n_vocab = tiktoken.get_encoding("gpt2").n_vocab


@dataclass
class TinyGPTConfig(PretrainedConfig):
    model_type: str = "tiny_gpt"
    num_workers: int = 1
    epochs: int = 1
    sample_rate: float = 0.001
    # adamw optimizer
    # max learning rate
    learning_rate: float = 6e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 2000
    ############################
    accumulation_steps: int = 10
    # clip gradients at this value, or disable if == 0.0
    clip_grad_max_norm: float = 1
    log_interval: int = 100
    eval_interval: int = 2000
    checkpoint_interval: int = 100
    dtype: str = "float16"
    vocab_size: int = n_vocab
    batch_size: int = 16
    seq_len: int = 128
    n_embd: int = 768
    n_layers: int = 2
    n_heads: int = 2

    attention_is_fast: bool = True
    attention_dropout_p: float = 0.1

    ffn_dropout_p: float = 0.1
    ffn_activation_fn: str = "gelu"

    normalization: str = "layer"
    normalization_bias: bool = True

    embd_dropout_p: float = 0.1

    out_dir: str = "out"


print(TinyGPTConfig())
