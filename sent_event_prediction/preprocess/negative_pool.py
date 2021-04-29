"""Generate negative event pool."""
import json
import logging
import os
import pickle
import random

from tqdm import tqdm

from sent_event_prediction.utils.document import document_iterator
from sent_event_prediction.utils.entity import Entity
from sent_event_prediction.utils.event import Event

logger = logging.getLogger(__name__)


def entity_check(event):
    """Check if the given event contains an entity."""
    return isinstance(event["subject"], Entity) or \
        isinstance(event["object"], Entity) or \
        isinstance(event["iobject"], Entity)


def generate_negative_pool(corp_dir, tokenized_dir, work_dir, num_events=None, suffix="train", file_type="tar"):
    """Sample a number of negative events."""
    neg_pool_path = os.path.join(work_dir, "negative_pool_{}.json".format(suffix))
    if os.path.exists(neg_pool_path):
        logger.info("{} already exists".format(neg_pool_path))
    else:
        neg_pool = []
        with tqdm() as pbar:
            for doc in document_iterator(corp_dir, tokenized_dir, shuffle=True, file_type=file_type):
                if num_events is not None and len(neg_pool) >= num_events:
                    break
                else:
                    for ent in doc.entities:
                        ent.clear_mentions()
                    # events = [e for e in doc.events]
                    events = doc.events
                    # If event less than 10, pick all events,
                    # else randomly pick 10 events from event list.
                    # Notice: all events should have
                    # at least one argument that is an entity!
                    events = [e for e in events if entity_check(e)]
                    if len(events) < 10:
                        neg_pool.extend(events)
                    else:
                        neg_pool.extend(random.sample(events, 10))
                    if num_events is not None and len(neg_pool) > num_events:
                        neg_pool = neg_pool[:num_events]
                pbar.update(1)
        with open(neg_pool_path, "w") as f:
            json.dump(neg_pool, f)
        logger.info("Save negative pool to {}".format(neg_pool_path))


def load_negative_pool(work_dir, suffix="train"):
    """Load negative event pool."""
    neg_pool_path = os.path.join(work_dir, "negative_pool_{}.json".format(suffix))
    with open(neg_pool_path, "r") as f:
        neg_pool = json.load(f)
    neg_pool = [Event(**e) for e in neg_pool]
    return neg_pool


__all__ = ["generate_negative_pool", "load_negative_pool"]
