"""Count frequent event(verb) type."""
import logging
import os
from collections import Counter

from tqdm import tqdm

from sent_event_prediction.utils.document import document_iterator


logger = logging.getLogger(__name__)


def count_stop_event(corp_dir,
                     work_dir,
                     file_type="tar",
                     num_events=10,
                     overwrite=False,
                     ):
    """Count frequent event(verb) type."""
    stop_event_path = os.path.join(work_dir, "stoplist.txt")
    if os.path.exists(stop_event_path) and not overwrite:
        logger.info("{} already exists".format(stop_event_path))
    else:
        # Count verb:role occurrence
        counter = Counter()
        logger.info("Scanning training documents ...")
        with tqdm() as pbar:
            for doc in document_iterator(corp_dir, file_type):
                pbar.set_description("Processing {}".format(doc.doc_id))
                for entity, chain in doc.get_chains():
                    preds = [e.predicate_gr(entity) for e in chain]
                    counter.update(preds)
                pbar.update(1)
        # Select top N verb:role
        stop_events = [t[0] for t in counter.most_common(num_events)]
        logger.info("Top {} frequent predicates are: {}".format(num_events, ", ".join(stop_events)))
        # Save to file
        with open(stop_event_path, "w") as f:
            f.write("\n".join(stop_events))
        logger.info("Save stop_events to {}".format(stop_event_path))
