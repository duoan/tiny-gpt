from dataclasses import dataclass
from transformers import PretrainedConfig
import tiktoken

n_vocab = tiktoken.get_encoding("gpt2").n_vocab


@dataclass
class TinyGPTConfig(PretrainedConfig):
    model_type: str = "tiny_gpt"

    num_workers: int = 1
    epochs: int = 3
    # Input config
    train_sample_rate: float = 0.001
    val_sample_rate: float = 0.001
    # output config
    out_dir: str = "out"

    # adamw optimizer
    learning_rate: float = 5e-5
    adamw_weight_decay: float = 0.1
    adamw_beta1: float = 0.95
    adamw_beta2: float = 0.999  # 更保守的二阶动量
    adamw_eps: float = 1e-8

    # learning rate scheduler
    scheduler_warmup_pct: float = 0.15
    scheduler_anneal_strategy: str = "cos"
    scheduler_div_factor: int = 15
    scheduler_final_div_factor: int = 1e2
    scheduler_min_lr_factor: float = 0.01
    scheduler_base_momentum: float = 0.85
    scheduler_max_momentum: float = 0.95
    scheduler_cycle_momentum: bool = True

    accumulation_steps: int = 16
    # clip gradients at this value, or disable if == 0.0
    max_grad_norm: float = 2
    log_interval: int = 200
    eval_interval: int = 2000
    dtype: str = "float32"

    # Model config
    vocab_size: int = n_vocab
    batch_size: int = 16
    seq_len: int = 128
    n_embd: int = 768
    n_layers: int = 2
    n_heads: int = 2

    # position encoding approach
    # support
    # Absolute positional encoding
    # 1. [sinusoidal] Sinusoidal positional encoding   [TODO]
    # 2. [learnable] Learnable positional embedding    [DONE]
    # Relative Positional Encoding
    # 3. [rope] Rotary position embedding (RoPE)       [TODO]
    # 4. [alibi] ALiBi (Attention with Linear Biases)  [TODO]
    position_enc: str = "learnable"

    attention_is_fast: bool = True
    attention_dropout_p: float = 0.05

    ffn_dropout_p: float = 0.05
    ffn_activation_fn: str = "gelu"
    # support layer norm and RMS norm
    normalization: str = "layer"
    normalization_bias: bool = True

    embd_dropout_p: float = 0.05


print(TinyGPTConfig())
