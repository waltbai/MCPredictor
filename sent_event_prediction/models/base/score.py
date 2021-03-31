"""Score functions."""
import logging

import torch
from torch import nn


logger = logging.getLogger(__name__)


class FusionScore(nn.Module):
    """Fusion score."""

    def __init__(self, event_repr_size, directions=1):
        super(FusionScore, self).__init__()
        self.context_ffn = nn.Linear(event_repr_size * directions, 1)
        self.choice_ffn = nn.Linear(event_repr_size * directions, 1)

    def forward(self, context, choice):
        """Forward.

        :param context: size(batch_size, *, seq_len, event_repr_size)
        :param choice: size(batch_size, *, 1, event_repr_size)
        :return: size(batch_size, *, seq_len)
        """
        context_score = self.context_ffn(context).squeeze(-1)
        choice_score = self.choice_ffn(choice).squeeze(-1)
        return context_score + choice_score


class EuclideanScore(nn.Module):
    """Euclidean score."""

    def __init__(self):
        super(EuclideanScore, self).__init__()

    def forward(self, context, choice):
        """Forward.

        :param context: size(batch_size, *, seq_len, event_repr_size)
        :param choice: size(batch_size, *, 1, event_repr_size)
        :return: size(batch_size, *, seq_len)
        """
        return -torch.sqrt(torch.pow(context-choice, 2.).sum(-1))


class ManhattanScore(nn.Module):
    """Manhattan score."""

    def __init__(self):
        super(ManhattanScore, self).__init__()

    def forward(self, context, choice):
        """Forward.

        :param context: size(batch_size, *, seq_len, event_repr_size)
        :param choice: size(batch_size, *, 1, event_repr_size)
        :return: size(batch_size, *, seq_len)
        """
        return -torch.abs(context - choice).sum(-1)


class CosineScore(nn.Module):
    """Cosine score."""

    def __init__(self):
        super(CosineScore, self).__init__()

    def forward(self, context, choice):
        """Forward.

        :param context: size(batch_size, *, seq_len, event_repr_size)
        :param choice: size(batch_size, *, 1, event_repr_size)
        :return: size(batch_size, *, seq_len)
        """
        inner_prod = (context * choice).sum(-1)
        context_length = torch.sqrt(torch.pow(context, 2.).sum(-1))
        choice_length = torch.sqrt(torch.pow(choice, 2.).sum(-1))
        score = inner_prod / context_length / choice_length
        return score


def build_score(config):
    """Build score function."""
    layer_name = config["score"]
    if layer_name not in ["fusion", "manhattan", "euclidean", "cosine"]:
        logger.info("Unknown score function '{}', "
                    "default to use euclidean.".format(layer_name))
        layer_name = "euclidean"
    if layer_name == "fusion":
        # Get layer specific hyper-parameters
        event_repr_size = config["event_repr_size"]
        directions = config["directions"]
        # Initialize layer
        layer = FusionScore(event_repr_size, directions)
    elif layer_name == "manhattan":
        layer = ManhattanScore()
    elif layer_name == "euclidean":
        layer = EuclideanScore()
    else:   # layer_name == "cosine"
        layer = CosineScore()
    return layer


__all__ = ["build_score"]
