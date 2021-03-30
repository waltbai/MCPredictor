from torch import nn

from sent_event_prediction.models.base.sentence_encoder import build_sent_encoder


class SCPredictor(nn.Module):
    """This model contains two different parts: event part and sentence part.

    Event part encodes events and predict next event,
    while sentence part encodes sentences and predict next sentence.
    A constraint is applied between events and sentences to force their representation to be similar.
    """

    def __init__(self, config):
        super(SCPredictor, self).__init__()
        self.config = config
        self.sent_encoder = build_sent_encoder(config)

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
        # Sentence encoding
        if sents is not None:
            sent_repr = self.sent_encoding(sents, sent_mask)
        # Event loss
        # Sentence loss
        # Event-Sentence constraint
        # Return

    def event_encoding(self, events):
        """Encode events."""

    def sent_encoding(self, sents, sent_mask):
        """Encode sentences."""
        return self.sent_encoder(sents, sent_mask)


