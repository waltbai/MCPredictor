import torch
from torch import nn

from sent_event_prediction.models.base.attention import build_attention
from sent_event_prediction.models.base.embedding import build_embedding
from sent_event_prediction.models.base.event_encoder import build_event_encoder
from sent_event_prediction.models.base.score import build_score
from sent_event_prediction.models.base.sentence_encoder import build_sent_encoder
from sent_event_prediction.models.base.sequence_model import build_sequence_model


class SCPredictorSent(nn.Module):
    """This model contains two different parts: event part and sentence part.

    Event part encodes events and predict next event,
    while sentence part encodes sentences and predict next sentence.
    A constraint is applied between events and sentences to force their representation to be similar.
    """

    def __init__(self, config, pretrain_embedding=None, tokenizer=None):
        super(SCPredictorSent, self).__init__()
        self.config = config
        # Event part
        self.embedding = build_embedding(config, pretrain_embedding)
        self.event_encoder = build_event_encoder(config)
        self.event_sequence_model = build_sequence_model(config)
        self.event_score = build_score(config)
        self.event_attention = build_attention(config)
        # Sentence part
        self.sent_encoder = build_sent_encoder(config, vocab_size=30525)
        self.sent_sequence_model = build_sequence_model(config)
        # self.sent_score = nn.Linear(128, 1)
        self.sent_score = build_score(config)
        self.sent_attention = build_attention(config)
        # Criterion
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, events, sents=None, sent_mask=None, target=None):
        """Forward function.

        If "sents" and "target" is not None, return the sum of three losses,
        otherwise, only return the scores of 5 choices.

        :param events: size(batch_size, choice_num, seq_len + 1, 5)
        :param sents: size(batch_size, choice_num, seq_len, sent_len)
        :param sent_mask: size(batch_size, choice_num, seq_len, sent_len)
        :param target:
        """
        batch_size = events.size(0)
        choice_num = events.size(1)
        seq_len = events.size(2) - 1
        # Event encoding
        # event_repr: size(batch_size, choice_num, seq_len + 1, event_repr_size)
        event_repr = self.event_encoding(events)
        # Event sequence modeling
        # updated_event_repr: size(batch_size, choice_num, seq_len+1, event_repr_size)
        event_repr = event_repr.view(batch_size * choice_num, seq_len + 1, -1)
        # Sentence encoding
        sent_repr = self.sent_encoding(sents, sent_mask)
        sent_repr = sent_repr.view(batch_size * choice_num, seq_len + 1, -1)
        # Sentence sequence modeling
        event_repr = torch.cat([event_repr, sent_repr], dim=-1)
        updated_event_repr = self.event_sequence_model(event_repr)
        updated_event_repr = updated_event_repr.view(batch_size, choice_num, seq_len + 1, -1)
        # Event scoring and attention
        # event_context: size(batch_size, choice_num, seq_len, event_repr_size)
        # event_choice: size(batch_size, choice_num, seq_len, event_repr_size)
        event_context = updated_event_repr[:, :, :-1, :]
        event_choice = updated_event_repr[:, :, -1:, :]
        # Event loss
        # event_score: size(batch_size, choice_num)
        event_score = self.event_score(event_context, event_choice)
        event_attention = self.event_attention(event_context, event_choice)
        event_score = (event_score * event_attention).sum(-1)
        if target is not None:
            event_loss = self.criterion(event_score, target)
        else:
            event_loss = event_score
        return event_loss

    def forward_event(self, events, target=None, return_hidden=False):
        """Only forward event part."""
        batch_size = events.size(0)
        choice_num = events.size(1)
        seq_len = events.size(2) - 1
        event_repr = self.event_encoding(events)
        event_repr = event_repr.view(batch_size * choice_num, seq_len + 1, -1)
        updated_event_repr = self.event_sequence_model(event_repr)
        updated_event_repr = updated_event_repr.view(batch_size, choice_num, seq_len + 1, -1)
        event_context = updated_event_repr[:, :, :-1, :]
        event_choice = updated_event_repr[:, :, -1:, :]
        event_score = self.event_score(event_context, event_choice)
        event_attention = self.event_attention(event_context, event_choice)
        event_score = (event_score * event_attention).sum(-1)
        if target is None:
            event_loss = event_score
        else:
            event_loss = self.criterion(event_score, target)
        if return_hidden:
            return event_loss, updated_event_repr
        else:
            return event_loss

    def forward_sent(self, sents, sent_mask, target=None, return_hidden=False):
        """Only forward sentence part."""
        batch_size = sents.size(0)
        choice_num = sents.size(1)
        seq_len = sents.size(2) - 1
        sent_repr = self.sent_encoding(sents, sent_mask)
        sent_repr = sent_repr.view(batch_size * choice_num, seq_len + 1, -1)
        updated_sent_repr = self.sent_sequence_model(sent_repr)
        updated_sent_repr = updated_sent_repr.view(batch_size, choice_num, seq_len + 1, -1)
        sent_context = updated_sent_repr[:, :, :-1, :]
        sent_choice = updated_sent_repr[:, :, -1:, :]
        sent_score = self.sent_score(sent_context, sent_choice)
        sent_attention = self.sent_attention(sent_context, sent_choice)
        sent_score = (sent_score * sent_attention).sum(-1)
        if target is None:
            sent_loss = sent_score
        else:
            sent_loss = self.criterion(sent_score, target)
        if return_hidden:
            return sent_loss, updated_sent_repr
        else:
            return sent_loss

    def forward_all(self, events, sents=None, sent_mask=None, target=None):
        """Forward function.

        If "sents" and "target" is not None, return the sum of three losses,
        otherwise, only return the scores of 5 choices.

        :param events: size(batch_size, choice_num, seq_len + 1, 5)
        :param sents: size(batch_size, choice_num, seq_len, sent_len)
        :param sent_mask: size(batch_size, choice_num, seq_len, sent_len)
        :param target:
        """
        batch_size = events.size(0)
        choice_num = events.size(1)
        seq_len = events.size(2) - 1
        # Event encoding
        # event_repr: size(batch_size, choice_num, seq_len + 1, event_repr_size)
        event_repr = self.event_encoding(events)
        # Event sequence modeling
        # updated_event_repr: size(batch_size, choice_num, seq_len+1, event_repr_size)
        event_repr = event_repr.view(batch_size * choice_num, seq_len + 1, -1)
        updated_event_repr = self.event_sequence_model(event_repr)
        updated_event_repr = updated_event_repr.view(batch_size, choice_num, seq_len + 1, -1)
        # Sentence encoding
        if sents is not None:
            sent_repr = self.sent_encoding(sents, sent_mask)
        else:
            sent_repr = None
        # Sentence sequence modeling
        if sent_repr is not None:
            sent_repr = sent_repr.view(batch_size * choice_num, seq_len + 1, -1)
            updated_sent_repr = self.sent_sequence_model(sent_repr)
            updated_sent_repr = updated_sent_repr.view(batch_size, choice_num, seq_len + 1, -1)
        else:
            updated_sent_repr = None
        # Event scoring and attention
        # event_context: size(batch_size, choice_num, seq_len, event_repr_size)
        # event_choice: size(batch_size, choice_num, seq_len, event_repr_size)
        event_context = updated_event_repr[:, :, :-1, :]
        event_choice = updated_event_repr[:, :, -1:, :]
        # Event loss
        # event_score: size(batch_size, choice_num)
        event_score = self.event_score(event_context, event_choice)
        event_attention = self.event_attention(event_context, event_choice)
        event_score = (event_score * event_attention).sum(-1)
        if target is not None:
            event_loss = self.criterion(event_score, target)
        else:
            event_loss = event_score
        # Sentence scoring, only use last hidden state
        if updated_sent_repr is not None:
            sent_context = updated_sent_repr[:, :, :-1, :]
            sent_choice = updated_sent_repr[:, :, -1:, :]
            sent_score = self.sent_score(sent_context, sent_choice)
            sent_attention = self.sent_attention(sent_context, sent_choice)
            sent_score = (sent_score * sent_attention).sum(-1)
        else:
            sent_score = None
        # Sentence loss
        if target is not None and sent_score is not None:
            sent_loss = self.criterion(sent_score, target)
        else:
            sent_loss = None
        # Event-Sentence constraint
        if updated_sent_repr is not None:
            # event_repr: size(batch_size, choice_num, seq_len, event_repr_size)
            # sent_repr: size(batch_size, choice_num, seq_len, event_repr_size)
            dist = torch.sqrt(torch.pow(updated_event_repr - updated_sent_repr, 2.).sum(-1)).mean()
        else:
            dist = None
        # Return
        if sent_loss is None:
            if event_loss is None:
                return event_score
            else:
                return event_loss
        else:
            return event_loss, sent_loss, dist

    def event_encoding(self, events):
        """Encode events."""
        # Embedding
        # event_repr: size(batch_size, choice_num, seq_len + 1, 4, embedding_size)
        event_repr = self.embedding(events)
        # Encoding
        # event_repr: size(batch_size, choice_num, seq_len + 1, event_repr_size)
        event_repr = self.event_encoder(event_repr)
        return event_repr

    def sent_encoding(self, sents, sent_mask):
        """Encode sentences."""
        return self.sent_encoder(sents, sent_mask)
