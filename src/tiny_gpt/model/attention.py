import torch
from torch import nn
from typing import Optional
import math
import torch.nn.functional as F


def _split_head(
    x: torch.Tensor, num_heads: int, head_dim: int, group_num: int | None = None
) -> torch.Tensor:
    """Split q, k, into mult-heads
    Args:
        x (torch.Tensor):
            [
                # batch#0
                [
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], # token#0 embeddings
                    [1.1, 1.2, 1.3, 1.4, 1.5, 1.6], # token#1 embeddings
                    [2.1, 2.2, 2.3, 2.4, 2.5, 2.6], # token#2 embeddings
                    [3.1, 3.2, 3.3, 3.4, 3.5, 3.6]  # token#3 embeddings
                ],
                # batch#1
                [
                    [4.1, 4.2, 4.3, 4.4, 4.5, 4.6],
                    [5.1, 5.2, 5.3, 5.4, 5.5, 5.6],
                    [6.1, 6.2, 6.3, 6.4, 6.5, 6.6],
                    [7.1, 7.2, 7.3, 7.4, 7.5, 7.6]
                ]
            ]

    Returns:
        torch.Tensor: example
            [
                # batch#0
                [
                    # head#0
                    [[0.1, 0.2], # token#0
                    [1.1, 1.2],  # token#1
                    [2.1, 2.2],  # token#2
                    [3.1, 3.2]], # token#3
                    # head#1
                    [[0.3, 0.4],
                    [1.3, 1.4],
                    [2.3, 2.4],
                    [3.3, 3.4]],
                    # head#2
                    [[0.5, 0.6],
                    [1.5, 1.6],
                    [2.5, 2.6],
                    [3.5, 3.6]]
                ],
                # batch#1
                [...]
            ]
    """
    batch_size, seq_len = x.size()[:2]
    if group_num is None:
        return x.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
    else:
        x = x.view(batch_size, -1, group_num, head_dim).transpose(1, 2)
        x = (
            x[:, :, None, :, :]
            .expand(batch_size, group_num, num_heads // group_num, seq_len, head_dim)
            .reshape(batch_size, num_heads // group_num * group_num, seq_len, head_dim)
        )
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()

        assert (
            hidden_size % num_heads == 0
        ), "Embedding dimension must be 0 modulo number of heads."
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # q, k, v projection
        self.qw = nn.Linear(hidden_size, hidden_size)
        self.kw = nn.Linear(hidden_size, hidden_size)
        self.vw = nn.Linear(hidden_size, hidden_size)

        # attention output projection
        self.ow = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        batch_size = x.size()[0]

        # project q, k, v
        qx = self.qw(x)
        kx = self.kw(x)
        vx = self.vw(x)

        qx = _split_head(qx, self.num_heads, self.head_dim)
        kx = _split_head(kx, self.num_heads, self.head_dim)
        vx = _split_head(vx, self.num_heads, self.head_dim)

        # calculate attention score using Q dot K
        scale_factor = 1 / math.sqrt(self.head_dim)
        attention_scores = torch.matmul(qx, kx.transpose(-1, -2)) * scale_factor
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -9e15)

        # normalize attention scores for each token
        attention_probs = F.softmax(attention_scores, dim=-1)
        # optional dropout
        attention_weights = torch.matmul(attention_probs, vx)
        # convert back to
        attention_weights = (
            attention_weights.transpose(-1, -2)
            .contiguous()
            .view(batch_size, -1, self.head_dim * self.num_heads)
        )

        return self.ow(attention_weights)


class MultiQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiQueryAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qw = nn.Linear(hidden_size, hidden_size)

        # all heads share k, v projection
        self.kw = nn.Linear(hidden_size, self.head_dim)
        self.vw = nn.Linear(hidden_size, self.head_dim)

        # output projection
        self.ow = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        batch_size = x.size()[0]

        qx = self.qw(x)
        kx = self.kw(x)
        vx = self.vw(x)

        qx = _split_head(qx, self.num_heads, self.head_dim)

        kx = _split_head(kx, 1, self.head_dim)
        vx = _split_head(vx, 1, self.head_dim)

        attention_scores = torch.matmul(qx, kx.transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -9e15)

        attention_probs = F.softmax(attention_scores, dim=-1)

        attention_weights = torch.matmul(attention_probs, vx)
        attention_weights = (
            attention_weights.transpose(-1, -2)
            .contiguous()
            .view(batch_size, -1, self.head_dim * self.num_heads)
        )

        return self.ow(attention_weights)


class GroupQueryAttention(nn.Module):

    def __init__(self, hidden_size, num_heads, group_num):
        super(GroupQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.group_num = group_num

        self.qw = nn.Linear(hidden_size, hidden_size)
        self.kw = nn.Linear(hidden_size, self.head_dim * self.group_num)
        self.vw = nn.Linear(hidden_size, self.head_dim * self.group_num)

        self.ow = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        batch_size = x.size()[0]

        qx = self.qw(x)
        kx = self.kw(x)
        vx = self.vw(x)

        qx = _split_head(qx, self.num_heads, self.head_dim)
        kx = _split_head(kx, self.num_heads, self.head_dim, self.group_num)
        vx = _split_head(vx, self.num_heads, self.head_dim, self.group_num)

        attention_scores = torch.matmul(qx, kx.transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -9e15)

        attention_probs = F.softmax(attention_scores, dim=-1)

        attention_weights = torch.matmul(attention_probs, vx)
        attention_weights = (
            attention_weights.transpose(-1, -2)
            .contiguous()
            .view(batch_size, -1, self.head_dim * self.num_heads)
        )

        return self.ow(attention_weights)


if __name__ == "__main__":
    x = torch.randn(2, 4, 16)
    mha = MultiHeadAttention(16, 4)
    print(mha(x, None))

    mqa = MultiQueryAttention(16, 4)
    print(mqa(x, None))

    gqa = GroupQueryAttention(16, 4, 2)
    print(gqa(x, None))
