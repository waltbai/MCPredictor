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


class ConveScore(nn.Module):
    """ConvE score."""

    def __init__(self, event_repr_size,
                 num_out_channels=32,
                 kernel_size=3,
                 emb_2d_d1=8,
                 emb_2d_d2=None,
                 directions=1, dropout=0.1):
        super(ConveScore, self).__init__()
        # args
        event_repr_size = event_repr_size * directions
        if emb_2d_d2 is None:
            emb_2d_d2 = event_repr_size * directions // emb_2d_d1
        else:
            emb_2d_d2 = emb_2d_d2 * directions
        assert emb_2d_d1 * emb_2d_d2 == event_repr_size
        self.event_repr_size = event_repr_size
        self.emb_2d_d1 = emb_2d_d1
        self.emb_2d_d2 = emb_2d_d2
        self.num_out_channels = num_out_channels
        self.w_d = kernel_size
        h_out = 2 * self.emb_2d_d1 - self.w_d + 1
        w_out = self.emb_2d_d2 - self.w_d + 1
        self.feat_dim = self.num_out_channels * h_out * w_out
        # layers
        self.hidden_dropout = nn.Dropout(dropout)
        self.feature_dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv2d(1, self.num_out_channels, (self.w_d, self.w_d), 1, 0)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(self.num_out_channels)
        self.bn2 = nn.BatchNorm1d(self.event_repr_size)
        self.fc1 = nn.Linear(self.feat_dim, self.event_repr_size)
        self.fc2 = nn.Linear(self.event_repr_size, 1)
        self.relu = nn.ReLU()

    def forward(self, context, choice):
        """Forward.

        :param context: size(batch_size, *, seq_len, event_repr_size)
        :param choice: size(batch_size, *, 1, event_repr_size)
        :return: size(batch_size, *, seq_len)
        """
        ori_context_size = context.size()
        context = context.contiguous().view(-1, self.event_repr_size)
        choice = choice.expand(ori_context_size).contiguous().view(-1, self.event_repr_size)
        # context: size(batch_size, event_repr_size)
        # choice: size(batch_size, event_repr_size)
        context = context.view(-1, 1, self.emb_2d_d1, self.emb_2d_d2)
        choice = choice.view(-1, 1, self.emb_2d_d1, self.emb_2d_d2)
        stacked_inputs = self.bn0(torch.cat([context, choice], 2))
        x = self.conv1(stacked_inputs)
        x = self.relu(x)
        x = self.feature_dropout(x)
        x = x.view(-1, self.feat_dim)
        x = self.fc1(x)
        x = self.hidden_dropout(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x: size(batch_size, 1)
        x = x.view(ori_context_size[:-1])
        return x


def build_score(config):
    """Build score function."""
    layer_name = config["score"]
    if layer_name not in ["fusion", "manhattan", "euclidean", "cosine", "conve"]:
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
    elif layer_name == "conve":
        event_repr_size = config["event_repr_size"]
        directions = config["directions"]
        # num_out_channels = config["num_out_channels"]
        # kernel_size = config["kernel_size"]
        # emb_2d_d1 = config["emb_2d_d1"]
        # emb_2d_d2 = config["emb_2d_d2"]
        dropout = config["dropout"]
        layer = ConveScore(event_repr_size=event_repr_size)
    else:   # layer_name == "cosine"
        layer = CosineScore()
    return layer


__all__ = ["build_score"]
