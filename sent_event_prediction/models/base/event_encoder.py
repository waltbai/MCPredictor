"""Event encoder that encodes verb and arguments."""
from torch import nn


class EventFusionEncoder(nn.Module):
    """Event encoder layer."""

    def __init__(self, embedding_size, event_repr_size, dropout=0.):
        """Event encoder layer."""
        super(EventFusionEncoder, self).__init__()
        self.linear = nn.Linear(embedding_size * 4, event_repr_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, events):
        """Forward.

        input_dim: (*, 4, embedding_size)

        output_dim: (*, event_repr_size)
        """
        shape = events.size()
        input_shape = shape[:-2] + (shape[-1] * 4, )
        projections = self.activation(self.linear(events.view(input_shape)))
        projections = self.dropout(projections)
        return projections


def build_event_encoder(config):
    """Build event encoder layer."""
    embedding_size = config["embedding_size"]
    event_repr_size = config["event_repr_size"]
    dropout = config["dropout"]
    return EventFusionEncoder(embedding_size=embedding_size,
                              event_repr_size=event_repr_size,
                              dropout=dropout)


__all__ = ["build_event_encoder"]
