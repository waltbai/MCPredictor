"""Sentence encoder using bert."""
from torch import nn
from transformers import AutoModel


class BertEncoder(nn.Module):
    """Bert sentence encoder."""

    def __init__(self, sent_repr_size=None):
        super(BertEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        bert_repr_size = 128
        if sent_repr_size != bert_repr_size:
            self.linear = nn.Linear(self.bert_repr_size, sent_repr_size)
        else:
            self.linear = None

    def forward(self, sents, mask=None):
        """Forward.

        :param sents: size(*, sent_len)
        :param mask: size(*, sent_len)
        :return: size(*, sent_repr_size)
        """
        original_size = sents.size()
        sent_len = original_size[-1]
        sents = sents.view(-1, sent_len)
        mask = mask.view(-1, sent_len)
        sent_embeddings = self.bert(input_ids=sents, attention_mask=mask)
        sent_embeddings = sent_embeddings[:, 0, :]
        if self.linear is not None:
            sent_embeddings = self.linear(sent_embeddings)
        sent_embeddings = sent_embeddings.view(original_size[:-1] + (-1, ))
        return sent_embeddings


def build_sent_encoder(config):
    """Build sentence encoder."""
    event_repr_size = config["event_repr_size"]
    return BertEncoder(sent_repr_size=event_repr_size)


__all__ = ["build_sent_encoder"]
