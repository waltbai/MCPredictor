import json
import logging
import os

import numpy
from gensim.models import Word2Vec

from mcpredictor.utils.document import document_iterator

logger = logging.getLogger(__name__)


class ChainIterator:
    """Narrative event chain iterator for word2vec."""

    def __init__(self, corp_dir):
        self.corp_dir = corp_dir

    def __iter__(self):
        for doc in document_iterator(self.corp_dir,
                                     tokenized_dir=None,
                                     file_type="tar",
                                     doc_type="train",
                                     shuffle=False,
                                     pos_dir=None):
            for entity, chain in doc.get_chains():
                sequence = []
                for event in chain:
                    _, a0, a1, a2 = event.tuple()
                    sequence.extend([event.predicate_gr(entity), a0, a1, a2])
                sequence = [t for t in sequence if t != "None"]
                yield sequence


def generate_word_embedding(train_corp_dir, work_dir, embedding_size=300, force=False):
    """Generate word embeddings and dictionary."""
    # Train word2vec
    word2vec_path = os.path.join(work_dir, "word2vec_{}.bin".format(embedding_size))
    if os.path.exists(word2vec_path) and not force:
        logger.info("Word embeddings generated.")
    else:
        logger.info("Generating word embeddings ...")
        word2vec = Word2Vec(ChainIterator(train_corp_dir),
                            size=embedding_size,
                            window=15,
                            workers=8,
                            min_count=20)
        word2vec.save(word2vec_path)
        logger.info("Save word2vec to {}".format(word2vec_path))
    # Generate pretrain embedding matrix
    pretrain_embedding_path = os.path.join(work_dir, "pretrain_embedding.npy")
    if os.path.exists(pretrain_embedding_path) and not force:
        logger.info("Pretrain embedding matrix generated.")
    else:
        logger.info("Generating pretrain embedding matrix ...")
        word2vec = Word2Vec.load(word2vec_path)
        word2vec.init_sims()
        total_words = len(word2vec.wv.vocab) + 1
        pretrain_embedding = numpy.zeros(
            (total_words, embedding_size), dtype=numpy.float32)
        for word in word2vec.wv.vocab:
            idx = word2vec.wv.vocab[word].index
            pretrain_embedding[idx + 1] = word2vec.wv.syn0norm[idx]
        numpy.save(pretrain_embedding_path, pretrain_embedding)
        logger.info("Save pretrain embedding matrix to {}".format(pretrain_embedding_path))
    # Generate word dictionary
    word_dict_path = os.path.join(work_dir, "word_dict.json")
    if os.path.exists(word_dict_path) and not force:
        logger.info("Word dictionary generated.")
    else:
        logger.info("Generating word dictionary ...")
        word2vec = Word2Vec.load(word2vec_path)
        word2vec.init_sims()
        word_dict = {"None": 0}
        for word in word2vec.wv.vocab:
            idx = word2vec.wv.vocab[word].index
            word_dict[word] = idx + 1
        with open(word_dict_path, "w") as f:
            json.dump(word_dict, f)
        logger.info("Word dictionary save to {}, "
                    "totally {} words.".format(word_dict_path, len(word_dict)))
