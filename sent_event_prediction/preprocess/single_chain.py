"""Generate single chain data."""
import logging
import os
import random
from copy import copy

from tqdm import tqdm

from sent_event_prediction.preprocess.negative_pool import load_negative_pool
from sent_event_prediction.preprocess.stop_event import load_stop_event
from sent_event_prediction.preprocess.word_dict import load_word_dict
from sent_event_prediction.utils.document import document_iterator

logger = logging.getLogger(__name__)


def replace_mention(sentence, old_mention, new_mention):
    """Replace old mention in sentence with new mention."""
    return sentence.replace(old_mention["text"], new_mention["text"])


def negative_sampling(positive_event,
                      negative_pool,
                      protagonist,
                      non_protagonist_entities,
                      num_events=4):
    """Sampling negative events from negative pool.

    Entities in negative events are replaced with protagonist
    and random non-protagonist entities.
    Entity mentions in sentences are replaced, too.

    :param positive_event: positive event
    :param negative_pool: negative event pool
    :param protagonist: protagonist entity
    :param non_protagonist_entities: non-protagonist entities
    :param num_events: number of negative events
    """
    negative_events = []
    for _ in range(num_events):
        # Sample a negative event
        negative_event = random.choice(negative_pool)
        while negative_event["verb_lemma"] == positive_event["verb_lemma"]:
            negative_event = random.choice(negative_pool)
        negative_event = copy(negative_event)
        # Assign entity mapping
        negative_entities = negative_event.get_entities()
        negative_protagonist = random.choice(negative_entities)
        negative_non_protagonist = [e for e in negative_entities if e is not negative_protagonist]
        replace_non_protagonist = []
        for e in negative_non_protagonist:
            if len(non_protagonist_entities) > 0:
                replace_non_protagonist.append(random.choice(non_protagonist_entities))
            else:
                replace_non_protagonist.append(e)
        # Replace mention in sentence
        # Replace entity
        negative_event.replace_entity(negative_protagonist, protagonist)
        for neg, rep in zip(negative_non_protagonist, replace_non_protagonist):
            negative_event.replace_entity(neg, rep)



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
