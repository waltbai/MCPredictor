"""Attention functions."""
import logging
import math

import torch
from torch import nn


logger = logging.getLogger(__name__)


class AdditiveAttention(nn.Module):
    """Additive attention function."""

    def __init__(self, event_repr_size, directions=1):
        super(AdditiveAttention, self).__init__()
        self.ffn = nn.Linear(event_repr_size * directions * 2, 1)

    def forward(self, context, choice, mask=None):
        """Forward.

        :param context: size(batch_size, *, seq_len, event_repr_size)
        :param choice: size(batch_size, *, 1 or seq_len, event_repr_size)
        :param mask: size(batch_size, *, seq_len)
        :return: size(batch_size, *, seq_len)
        """
        if choice.size(-2) == 1:
            choice = choice.expand(context.size())
        __input = torch.cat([context, choice], dim=-1)
        weight = self.ffn(__input).sequeeze()
        if mask is not None:
            weight = weight.masked_fill(mask, -1e9)
        attn = torch.softmax(weight, dim=-1)
        return attn


class DotAttention(nn.Module):
    """Dot attention function."""

    def __init__(self):
        super(DotAttention, self).__init__()

    def forward(self, context, choice, mask=None):
        """Forward.

        :param context: size(batch_size, *, seq_len, event_repr_size)
        :param choice: size(batch_size, *, 1 or seq_len, event_repr_size)
        :param mask: size(batch_size, *, seq_len)
        :return: size(batch_size, *, seq_len)
        """
        weight = (context * choice).sum(-1)
        if mask is not None:
            weight = weight.masked_fill(mask, -1e9)
        attn = torch.softmax(weight, dim=-1)
        return attn


class ScaledDotAttention(nn.Module):
    """Scaled dot attention function."""

    def __init__(self):
        super(ScaledDotAttention, self).__init__()

    def forward(self, context, choice, mask=None):
        """Forward.

        :param context: size(batch_size, *, seq_len, event_repr_size)
        :param choice: size(batch_size, *, 1 or seq_len, event_repr_size)
        :param mask: size(batch_size, *, seq_len)
        :return: size(batch_size, *, seq_len)
        """
        event_repr_size = context.size(-1)
        weight = (context * choice).sum(-1) / math.sqrt(event_repr_size)
        if mask is not None:
            weight = weight.masked_fill(mask, -1e9)
        attn = torch.softmax(weight, dim=-1)
        return attn


class AverageAttention(nn.Module):
    """Average attention function."""

    def __init__(self):
        super(AverageAttention, self).__init__()

    def forward(self, context, choice, mask=None):
        """Forward.

        :param context: size(batch_size, *, seq_len, event_repr_size)
        :param choice: size(batch_size, *, 1 or seq_len, event_repr_size)
        :param mask: size(batch_size, *, seq_len)
        :return: size(batch_size, *, seq_len)
        """
        weight = context.new_ones(context.size()[:-1], dtype=torch.float)
        if mask is not None:
            weight = weight.masked_fill(mask, -1e9)
        attn = torch.softmax(weight, dim=-1)
        return attn


def build_attention(config):
    """Build attention function."""
    layer_name = config["attention"]
    if layer_name not in ["average", "additive", "dot", "scaled-dot"]:
        logger.info("Unknown attention function '{}', "
                    "default to use scaled-dot.".format(layer_name))
        layer_name = "scaled-dot"
    if layer_name == "average":
        layer = AverageAttention()
    elif layer_name == "additive":
        event_repr_size = config["event_repr_size"]
        directions = config["directions"]
        layer = AdditiveAttention(event_repr_size, directions)
    elif layer_name == "dot":
        layer = DotAttention()
    else:   # layer_name == "scaled-dot"
        layer = ScaledDotAttention()
    return layer


__all__ = ["build_attention"]
