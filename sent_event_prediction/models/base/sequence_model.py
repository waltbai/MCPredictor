import logging
import math

import torch
from torch import nn
from torch.autograd import Variable


logger = logging.getLogger(__name__)


class PositionEncoder(nn.Module):
    """Position encoder used by transformer."""

    def __init__(self, d_model, seq_len, dropout=0.1):
        super(PositionEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0., seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Add position embedding to input tensor.

        :param x: size(batch_size, seq_len, d_model)
        """
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class Transformer(nn.Module):
    """Transformer layer with position encoding."""

    def __init__(self, d_model, seq_len=9, nhead=16,
                 dim_feedforward=256, num_layers=1, dropout=0.1):
        super(Transformer, self).__init__()
        self.position_encoder = PositionEncoder(d_model, seq_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers)

    def forward(self, x, mask=None):
        """Forward transformer layer.

        :param x: size(batch_size, seq_len, d_model)
        :param mask: size(batch_size, seq_len)
        """
        x = self.position_encoder(x)
        __input = x.transpose(0, 1)
        __output = self.transformer(__input, src_key_padding_mask=mask)
        return __output.transpose(0, 1)


def build_sequence_model(config):
    """Build sequence modeling layer."""
    # Get hyper-parameters
    event_repr_size = config["event_repr_size"]
    seq_len = config["seq_len"] + 1
    num_layers = config["num_layers"]
    dropout = config["dropout"]
    layer_name = config["sequence_model"]
    # Select and initialize layer
    if layer_name not in ["transformer", "lstm", "bilstm"]:
        logger.info("Unknown sequence model '{}', "
                    "default to use transformer.".format(layer_name))
        layer_name = "transformer"
    layer = None
    if layer_name == "transformer":
        # Get layer specific hyper-parameters
        num_heads = config["num_heads"]
        dim_feedforward = config["dim_feedforward"]
        config["directions"] = 1
        # Initialize transformer
        layer = Transformer(d_model=event_repr_size,
                            seq_len=seq_len,
                            nhead=num_heads,
                            num_layers=num_layers,
                            dim_feedforward=dim_feedforward,
                            dropout=dropout)
    else:  # layer_name in ["lstm", "bilstm"]
        if layer_name == "lstm":
            config["directions"] = 1
        else:
            config["directions"] = 2
    return layer


__all__ = ["build_sequence_model"]
