import logging
import os
import pickle
import random

from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizerFast

from sent_event_prediction.preprocess.negative_pool import load_negative_pool
from sent_event_prediction.preprocess.single_chain import negative_sampling
from sent_event_prediction.preprocess.stop_event import load_stop_event
from sent_event_prediction.preprocess.word_dict import load_word_dict
from sent_event_prediction.utils.document import document_iterator


logger = logging.getLogger(__name__)


def make_sample(doc,
                choices,
                target,
                context_size,
                verb_position,
                word_dict,
                tokenizer):
    """Make sample."""
    sample = []
    for choice in choices:
        for p_role in ["subject", "object", "iobject"]:
            protagonist = choice[p_role]
            # Get chain by protagonist
            chain = doc.get_chain_for_entity(protagonist, verb_position)
            # Truncate
            if len(chain) > context_size:
                chain = chain[-context_size:]
            if len(chain) < context_size:
                chain = [None] * (context_size - len(chain)) + chain
            # Make sample
            chain = chain + [choice]
            event_tmp = []
            sent_tmp = []
            for event in chain:
                if event is not None:
                    verb, subj, obj, iobj, role = event.tuple(protagonist)
                    predicate_gr = "{}:{}".format(verb, role)
                else:
                    predicate_gr = subj = obj = iobj = "None"
                tmp = [predicate_gr, subj, obj, iobj]







def generate_multi_train(corp_dir,
                         work_dir,
                         tokenized_dir,
                         part_size=100000,
                         file_type="tar",
                         context_size=8,
                         overwrite=False):
    """Generate multichain train data.

    :param corp_dir: train corpus directory
    :param work_dir: workspace directory
    :param tokenized_dir: tokenized raw text directory
    :param part_size: size of each partition
    :param file_type: "tar" or "txt"
    :param context_size: length of the context chain
    :param overwrite: whether to overwrite old data
    """
    # All parts of the dataset will be store in a sub directory.
    data_dir = os.path.join(work_dir, "multi_train")
    # Load stop list
    stoplist = load_stop_event(work_dir)
    # Load negative pool
    neg_pool = load_negative_pool(work_dir, "train")
    # Load word dictionary
    word_dict = load_word_dict(work_dir)
    # Load tokenizer
    special_tokens = ["[subj]", "[obj]", "[iobj]"]
    tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny",
                                                  additional_special_tokens=special_tokens)
    # Make sub directory
    os.makedirs(data_dir, exist_ok=True)
    partition = []
    partition_id = 0
    total_num = 0
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
                n = len(chain)
                for begin, end in zip(range(n), range(8, n)):
                    context = chain[begin:end]
                    answer = chain[end]
                    # Negative sampling
                    neg_choices = negative_sampling(positive_event=answer,
                                                    negative_pool=neg_pool,
                                                    protagonist=protagonist,
                                                    non_protagonist_entities=non_protagonist_entities)
                    # Make choices
                    choices = [answer] + neg_choices
                    random.shuffle(choices)
                    target = choices.index(answer)
                    # Make sample
                    sample = make_sample(doc=doc,
                                         choices=choices,
                                         target=target,
                                         context_size=context_size,
                                         verb_position=context[-1]["verb_position"],
                                         word_dict=word_dict,
                                         tokenizer=tokenizer)
                    partition.append(sample)
                    if len(partition) == part_size:
                        partition_path = os.path.join(data_dir, "train.{}".format(partition_id))
                        with open(partition_path, "wb") as f:
                            pickle.dump(partition, f)
                        total_num += len(partition)
                        partition_id += 1
                        partition = []
                pbar.update(1)
                if len(partition) > 0:
                    partition_path = os.path.join(data_dir, "train.{}".format(partition_id))
                    with open(partition_path, "wb") as f:
                        pickle.dump(partition, f)
                    total_num += len(partition)
                logger.info("Totally {} samples generated.".format(total_num))
