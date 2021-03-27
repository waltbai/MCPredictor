"""Generate single chain dataset."""
import logging
import os
import pickle
import random
from copy import copy
from pprint import pprint

from tqdm import tqdm

from sent_event_prediction.preprocess.negative_pool import load_negative_pool
from sent_event_prediction.preprocess.stop_event import load_stop_event
from sent_event_prediction.utils.document import document_iterator
from sent_event_prediction.utils.event import transform_entity

logger = logging.getLogger(__name__)


def replace_argument(event, role, new_arg):
    """Replace argument in the event."""
    # Replace mention in sentence
    old_arg = event[role]
    # Use lower case in case model judge by the first letter.
    event["sent"] = event["sent"].replace(old_arg["mention"], new_arg["mention"]).lower()
    # Replace argument
    event[role] = new_arg


def negative_sampling(neg_pool,
                      positive_event,
                      protagonist,
                      non_protagonist_entities,
                      num=4):
    """Sample negative events for a chain."""
    neg_events = []
    for _ in range(num):
        neg_event = random.choice(neg_pool)
        # If an event with the same verb is sampled, re-sample one.
        while neg_event["verb_lemma"] == positive_event["verb_lemma"]:
            neg_event = random.choice(neg_pool)
        # Copy the negative event
        neg_event = copy(neg_event)
        # Choose an argument as the protagonist
        arguments = [neg_event[role]
                     for role in ["subject", "object", "iobject"]
                     if neg_event[role]["entity"] >= 0]
        neg_protagonist = random.choice(arguments)
        # Replace arguments
        for role in ["subject", "object", "iobject"]:
            if neg_event[role] is neg_protagonist:
                replace_argument(neg_event, role, protagonist)
            elif neg_event[role]["entity"] >= 0 and len(non_protagonist_entities) > 0:
                new_arg = random.choice(non_protagonist_entities)
                replace_argument(neg_event, role, new_arg)
        neg_events.append(neg_event)
    return neg_events


def single_train(corp_dir,
                 work_dir,
                 tokenized_dir,
                 part_size=100000,
                 file_type="tar",
                 context_size=8):
    """Generate single chain train set.

    Dataset will split into several parts.

    :param corp_dir: corpus directory
    :param work_dir: workspace directory
    :param tokenized_dir: raw text directory
    :param part_size: size of each part
    :param file_type: whether corpus is in tar/txt mode
    :param context_size: context size of each question
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
        # Make sub directory
        os.makedirs(data_dir)
        part = []
        part_id = 0
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
                    non_protagonist_entities = [transform_entity(e)
                                                for e in non_protagonist_entities]
                    # Make samples
                    n = len(chain)
                    chain = [e.filter for e in chain]
                    for begin, end in zip(range(n), range(8, n)):
                        # Split context and answer
                        context = chain[begin:end]
                        answer = chain[end]
                        # Find mention by answer verb position
                        p = transform_entity(protagonist, answer["verb_position"])
                        # Negative sampling
                        neg_choices = negative_sampling(neg_pool=neg_pool,
                                                        positive_event=answer,
                                                        protagonist=p,
                                                        non_protagonist_entities=non_protagonist_entities)
                        # Generate choices
                        choices = neg_choices + [answer]
                        random.shuffle(choices)
                        target = choices.index(answer)
                        # Generate sample
                        sample = [p, context, choices, target]
                        part.append(sample)
                        if len(part) == part_size:
                            part_path = os.path.join(data_dir, "train.{}".format(part_id))
                            with open(part_path, "wb") as f:
                                pickle.dump(part, f)
                            part_id += 1
                            part = []
                pbar.update(1)
        logger.info("train set saved. Totally {} parts".format(part_id))


def single_eval(corp_dir,
                work_dir,
                tokenized_dir,
                file_type="txt"):
    """Generate single chain evaluation set."""
    for eval_mode in ["dev", "test"]:
        data_path = os.path.join(work_dir, "single_{}.pkl".format(eval_mode))
        if os.path.exists(data_path):
            logger.info("{} already exists".format(data_path))
        else:
            eval_set = []
            with tqdm() as pbar:
                for doc in document_iterator(corp_dir=corp_dir,
                                             tokenized_dir=tokenized_dir,
                                             file_type=file_type,
                                             doc_type="eval"):
                    protagonist = doc.entity
                    context = doc.context
                    choices = doc.choices
                    target = doc.target
                    # Transform protagonist
                    verb_position = choices[target]["verb_position"]
                    protagonist = transform_entity(protagonist, verb_position)
                    context = [e.filter for e in context]
                    # Assume each choice appear in the same position,
                    # thus they use the same mention of protagonist.
                    for e in choices:
                        e["verb_position"] = verb_position
                    # TODO: there is a problem:
                    #  there is no corresponding sentence for negative samples in evaluation set!
                    #  Unless we re-sample negative events.
                    choices = [e.filter for e in choices]
                    eval_set.append([protagonist, context, choices, target])
                    pbar.update(1)
            with open(data_path, "wb") as f:
                pickle.dump(eval_set, f)
            logger.info("{} set saved".format(eval_mode))



