import math
import torch
import inspect
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PreTrainedModel
from tiny_gpt.model.config import TinyGPTConfig
import logging

logger = logging.getLogger(__name__)


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_heads == 0

        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_heads
        self.is_fast = config.attention_is_fast
        self.dropout_p = config.attention_dropout_p

        # Q, K, V projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * self.n_embd, bias=False)
        # output projection layer
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        if not self.is_fast:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            seq_len = config.seq_len
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len),
            )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): shape (batch_size, seq_len, n_embd)

        Returns:
            torch.Tensor: shape (batch_size, seq_len, embd) of tensor with multi-head attention output.
        """
        batch_size, seq_len, n_embd = x.size()
        q, k, v = self.c_attn(x).split(n_embd, dim=2)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if self.is_fast:
            attn_weights = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout_p
            )
        else:
            scale_factor = 1 / math.sqrt(q.size(-1))
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
            # add mask for causal
            # prediction to prevent current token know the next tokens information [data leak]
            attention_mask = self.tril[:seq_len, :seq_len]
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -9e15)
            # Normalizing the attention score at token level
            attn_probs = F.softmax(attn_scores, dim=-1)
            # Dropout to prevent overfitting
            attn_probs = torch.dropout(attn_probs, self.dropout_p, train=True)
            # Apply the attention probs on the values
            attn_weights = torch.matmul(attn_probs, v)

        attn_weights = (
            attn_weights.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        )

        attn_outputs = self.c_proj(attn_weights)
        return attn_outputs


class FeedForwardNetwork(nn.Module):
    _activation_functions = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "swish": nn.SiLU(),
    }

    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.activation = self._activation_functions[config.ffn_activation_fn]
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


def create_norm_layer(config):
    return (
        nn.LayerNorm(config.n_embd, bias=config.normalization_bias)
        if config.normalization == "layer"
        else nn.RMSNorm(config.n_embd)
    )


class TransformerLayer(nn.Module):

    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        self.norm_1 = create_norm_layer(config)
        self.attn = MultiHeadCausalSelfAttention(config)
        self.norm_2 = create_norm_layer(config)
        self.ffn = FeedForwardNetwork(config)

    def forward(self, x):
        # We use the pre-normalization for stable training
        x = x + self.attn(self.norm_1(x))
        x = x + self.ffn(self.norm_2(x))
        return x


class TinyGPTModel(PreTrainedModel):

    def __init__(self, config: TinyGPTConfig = None):
        super().__init__(config)
        if config is None:
            config = TinyGPTConfig()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.seq_len, config.n_embd),
                drop=nn.Dropout(config.embd_dropout_p),
                layers=nn.ModuleList(
                    [TransformerLayer(config) for _ in range(config.n_layers)]
                ),
                norm=create_norm_layer(config),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: Tensor, targets: Tensor = None):
        """Foward calculate the next token probobilities in the vocabs for the given tokens of the input sequence.

        Args:
            idx (Tensor): a batch of sequences' tokens. shape (batch_size, sequence_length)
            targets (Tensor, optional): the ground truth next tokens for the batch sequences. shape (batch_size, 1)

        Returns:
            Tuple(Tensor, optional):
                - the first element is the next token probobilities Tensor. Shape (batch_size, vocab_size)
                - the second element is the loss, which meansure the gap with targets. Shape (batch_size, sequence_length)
        """
        device = idx.device
        batch_size, seq_len = idx.size()
        assert (
            seq_len <= self.config.seq_len
        ), f"Cannot forward sequence of length {seq_len}, block size is only {self.config.seq_len}"
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        # token embeddings of shape (batch_size, seq_len, n_embd)
        tok_emb = self.transformer.wte(idx)
        # position embeddings of shape (batch_size, n_embd)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for layer in self.transformer.layers:
            x = layer(x)

        x = self.transformer.norm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @torch.inference_mode
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=8,
        stream=True,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.seq_len
                else idx[:, -self.config.seq_len :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

            if stream:
                yield idx[:, index:]

        if not stream:
            yield idx[:, index:]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configure_optimizer(self, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.adamw_weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.info(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        logger.info(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(self.config.adamw_beta1, self.config.adamw_beta2),
            eps=self.config.adamw_eps,
            **extra_args,
        )
        logger.info(f"using fused AdamW: {use_fused}")

        return optimizer
