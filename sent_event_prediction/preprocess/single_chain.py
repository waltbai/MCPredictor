"""Generate single chain data."""
import logging
import os

from tqdm import tqdm

from sent_event_prediction.preprocess.negative_pool import load_negative_pool
from sent_event_prediction.preprocess.stop_event import load_stop_event
from sent_event_prediction.preprocess.word_dict import load_word_dict
from sent_event_prediction.utils.document import document_iterator

logger = logging.getLogger(__name__)


def negative_sampling():
    """Replace mention in sentence and replace entity in argument."""


def generate_single_train(corp_dir,
                          work_dir,
                          tokenized_dir,
                          part_size=200000,
                          file_type="tar",
                          context_size=8):
    """Generate single chain train data.

    :param corp_dir: train corpus directory
    :param work_dir: workspace directory
    :param tokenized_dir: tokenized raw text directory
    :param part_size: size of each partition
    :param file_type: "tar" or "txt"
    :param context_size: length of the context chain
    """
    # All parts of the dataset will be store in a sub directory.
    data_dir = os.path.join(work_dir, "single_train")
    if os.path.exists(data_dir):
        logger.info("{} already exists.".format(data_dir))
    else:
        # Load stop list
        stoplist = load_stop_event(work_dir)
        # Load negative pool
        neg_pool = load_negative_pool(work_dir)
        # Load word dictionary
        word_dict = load_word_dict(work_dir)
        # Make sub directory
        os.makedirs(data_dir)
        partition = []
        partition_id = 0
        with tqdm() as pbar:
            for doc in document_iterator(corp_dir=corp_dir,
                                         tokenized_dir=tokenized_dir,
                                         file_type=file_type,
                                         doc_type="train"):
                for protagonist, chain in doc.get_chains(stoplist):
                    # Context + Answer
                    if len(chain) < context_size + 1:
                        continue
                    # Get non protagonist entities
                    non_protagonist_entities = doc.non_protagonist_entities(protagonist)
                    # Make sample



def generate_single_eval():
    """Generate single chain evaluate data."""
