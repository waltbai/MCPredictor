import torch
from torch import nn

from mcpredictor.models.base.attention import build_attention
from mcpredictor.models.base.embedding import build_embedding
from mcpredictor.models.base.event_encoder import build_event_encoder
from mcpredictor.models.base.score import build_score
from mcpredictor.models.base.sentence_encoder import build_sent_encoder
from mcpredictor.models.base.sequence_model import build_sequence_model


class MCPredictorSent(nn.Module):
    """This model contains two different parts: event part and sentence part.

    Event part encodes events and predict next event,
    while sentence part encodes sentences and predict next sentence.
    A constraint is applied between events and sentences to force their representation to be similar.
    """

    def __init__(self, config, pretrain_embedding=None, tokenizer=None):
        super(MCPredictorSent, self).__init__()
        self.config = config
        # Event part
        self.embedding = build_embedding(config, pretrain_embedding)
        self.event_encoder = build_event_encoder(config)
        self.event_sequence_model = build_sequence_model(config)
        self.event_score = build_score(config)
        self.event_attention = build_attention(config)
        # Sentence part
        vocab_size = len(tokenizer) if tokenizer is not None else 30525
        self.sent_encoder = build_sent_encoder(config, vocab_size=vocab_size)
        # Criterion
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, events, sents=None, sent_mask=None, target=None):
        """Forward function.

        If "sents" and "target" is not None, return the loss,
        otherwise, only return the scores of 5 choices.

        :param events: size(batch_size, choice_num, chain_num, seq_len + 1, 5)
        :param sents: size(batch_size, choice_num, chain_num, seq_len, sent_len)
        :param sent_mask: size(batch_size, choice_num, chain_num, seq_len, sent_len)
        :param target:
        """
        batch_size = events.size(0)
        choice_num = events.size(1)
        chain_num = events.size(2)
        seq_len = events.size(3) - 1
        # Event mask
        event_mask = events.sum(-1)[:, :, :, :-1].to(torch.bool)
        # Event encoding
        # event_repr: size(batch_size, choice_num, chain_num, seq_len + 1, event_repr_size)
        event_repr = self.event_encoding(events)
        # Sentence encoding
        if sents is not None:
            sent_repr = self.sent_encoding(sents, sent_mask)
            event_repr_size = sent_repr.size(-1)
            zeros = sent_repr.new_zeros(batch_size, choice_num, chain_num, 1, event_repr_size)
            sent_repr = torch.cat([sent_repr, zeros], dim=-2)
        else:
            sent_repr = None
        # Event sequence modeling
        # updated_event_repr: size(batch_size, choice_num, chain_num, seq_len+1, event_repr_size)
        if sents is not None:
            event_repr = event_repr + sent_repr
        event_repr = event_repr.view(batch_size * choice_num * chain_num, seq_len + 1, -1)
        updated_event_repr = self.event_sequence_model(event_repr)
        updated_event_repr = updated_event_repr.view(batch_size, choice_num, chain_num, seq_len + 1, -1)
        # Event scoring and attention
        # event_context: size(batch_size, choice_num, chain_num, seq_len, event_repr_size)
        # event_choice: size(batch_size, choice_num, chain_num, 1, event_repr_size)
        event_context = updated_event_repr[:, :, :, :-1, :]
        event_choice = updated_event_repr[:, :, :, -1:, :]
        # Event loss
        # event_score: size(batch_size, choice_num)
        event_score = self.event_score(event_context, event_choice)
        event_attention = self.event_attention(event_context, event_choice, event_mask)
        event_score = (event_score * event_attention).sum(-1).sum(-1)
        if target is not None:
            return self.criterion(event_score, target)
        else:
            return event_score

    def event_encoding(self, events):
        """Encode events."""
        # Embedding
        # event_repr: size(batch_size, choice_num, chain_num, seq_len + 1, 4, embedding_size)
        event_repr = self.embedding(events)
        # Encoding
        # event_repr: size(batch_size, choice_num, chain_num, seq_len + 1, event_repr_size)
        event_repr = self.event_encoder(event_repr)
        return event_repr

    def sent_encoding(self, sents, sent_mask):
        """Encode sentences."""
        # size(batch_size, choice_num, chain_num, seq_len, event_repr_size)
        return self.sent_encoder(sents, sent_mask)
