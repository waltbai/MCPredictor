import torch
from torch import nn

from sent_event_prediction.models.base.embedding import build_embedding
from sent_event_prediction.models.base.event_encoder import build_event_encoder
from sent_event_prediction.models.base.sentence_encoder import build_sent_encoder
from sent_event_prediction.models.base.sequence_model import build_sequence_model


class SCPredictor(nn.Module):
    """This model contains two different parts: event part and sentence part.

    Event part encodes events and predict next event,
    while sentence part encodes sentences and predict next sentence.
    A constraint is applied between events and sentences to force their representation to be similar.
    """

    def __init__(self, config, pretrain_embedding=None):
        super(SCPredictor, self).__init__()
        self.config = config
        self.embedding = build_embedding(config, pretrain_embedding)
        self.event_encoder = build_event_encoder(config)
        self.sent_encoder = build_sent_encoder(config)
        self.event_sequence_model = build_sequence_model(config)
        self.sent_sequence_model = build_sequence_model(config)

    def forward(self, events, sents=None, sent_mask=None, target=None):
        """Forward function.

        If "sents" and "target" is not None, return the sum of three losses,
        otherwise, only return the scores of 5 choices.

        :param events: size(batch_size, choice_num, seq_len, 5)
        :param sents: size(batch_size, choice_num, seq_len, sent_len)
        :param sent_mask: size(batch_size, choice_num, seq_len, sent_len)
        :param target:
        """
        # Event encoding
        event_repr = self.event_encoding(events)
        # Sentence encoding
        if sents is not None:
            sent_repr = self.sent_encoding(sents, sent_mask)
        else:
            sent_repr = None
        # Event scoring and attention
        # Event loss
        # Sentence scoring and attention
        # Sentence loss
        # Event-Sentence constraint
        if sent_repr is not None:
            # event_repr: size(batch_size, choice_num, seq_len, event_repr_size)
            # sent_repr: size(batch_size, choice_num, seq_len, event_repr_size)
            dist = torch.sqrt(torch.pow(event_repr - sent_repr, 2.).sum(-1)).mean()
        # Return

    def event_encoding(self, events):
        """Encode events."""
        # Embedding
        event_repr = self.embedding(events)
        # Encoding
        event_repr = self.event_encoder(event_repr)
        return event_repr

    def sent_encoding(self, sents, sent_mask):
        """Encode sentences."""
        return self.sent_encoder(sents, sent_mask)


