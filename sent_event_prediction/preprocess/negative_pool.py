"""Generate negative event pool."""
from sent_event_prediction.utils.document import document_iterator


def negative_sampling(corp_dir, tokenize_dir, num_events=None):
    """Sample a number of negative events."""
    neg_pool = []
    for doc in document_iterator(corp_dir, tokenize_dir, shuffle=True):
        events = doc.events
        # TODO
