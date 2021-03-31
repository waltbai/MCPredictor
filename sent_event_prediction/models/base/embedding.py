import torch
from torch import nn


class EventEmbedding(nn.Module):
    """Word embedding"""
    def __init__(self, vocab_size, embedding_size,
                 dropout=0., pretrain_embedding=None):
        super(EventEmbedding, self).__init__()
        # Define word embedding
        if pretrain_embedding is not None:
            # Fix embedding works, otherwise corrupts.
            # I don't know why, but it works.
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(pretrain_embedding),
                padding_idx=0)
        else:
            self.embedding = nn.Embedding(
                vocab_size, embedding_size, padding_idx=0)
        # Define dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, events):
        """Forward"""
        output = self.dropout(self.embedding(events))
        return output


def build_embedding(config, pretrain_embedding=None):
    """Build embedding layer."""
    vocab_size = config["vocab_size"]
    embedding_size = config["embedding_size"]
    dropout = config["dropout"]
    return EventEmbedding(vocab_size=vocab_size,
                          embedding_size=embedding_size,
                          dropout=dropout,
                          pretrain_embedding=pretrain_embedding)


__all__ = ["build_embedding"]
