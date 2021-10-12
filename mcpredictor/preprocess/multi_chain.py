import logging
import os
import pickle
import random

from tqdm import tqdm
from transformers import BertTokenizerFast

from mcpredictor.preprocess.negative_pool import load_negative_pool
from mcpredictor.preprocess.single_chain import negative_sampling, align_pos_to_token
from mcpredictor.preprocess.stop_event import load_stop_event
from mcpredictor.preprocess.word_dict import load_word_dict
from mcpredictor.utils.document import document_iterator
from mcpredictor.utils.entity import Entity

logger = logging.getLogger(__name__)


def generate_mask_list(chain):
    """Generate masked words in chain."""
    masked_list = set()
    for event in chain:
        masked_list.update(event.get_words())
    return masked_list


def make_sample(doc,
                choices,
                target,
                context_size,
                verb_position,
                word_dict,
                stoplist,
                tokenizer):
    """Make sample."""
    sample_event = []
    sample_sent = []
    sample_mask = []
    sample_pos = []
    for choice in choices:
        choice_event = []
        choice_sent = []
        choice_mask = []
        # choice_pos = []
        for choice_role in ["subject", "object", "iobject"]:
            protagonist = choice[choice_role]
            # Get chain by protagonist
            chain = doc.get_chain_for_entity(protagonist, end_pos=verb_position, stoplist=stoplist)
            mask_list = generate_mask_list(chain)
            # Truncate
            if len(chain) > context_size:
                chain = chain[-context_size:]
            if len(chain) < context_size:
                chain = [None] * (context_size - len(chain)) + chain
            # Make sample
            chain_event = []
            chain_sent = []
            chain_mask = []
            # chain_pos = []
            if isinstance(protagonist, Entity):
                p_head = protagonist["head"]
            else:
                p_head = protagonist
            for event in chain:
                if event is not None:
                    verb, subj, obj, iobj, role = event.tuple(protagonist)
                    predicate_gr = "{}:{}".format(verb, role) if protagonist != "None" else "None"
                    tmp_mask_list = mask_list.difference(event.get_words())
                    # sent, pos = event.tagged_sent(role, mask_list=tmp_mask_list)
                    sent = event.tagged_sent(role, mask_list=tmp_mask_list)
                else:
                    predicate_gr = subj = obj = iobj = "None"
                    # sent, pos = [], []
                    sent = []
                tmp = [predicate_gr, subj, obj, iobj]
                event_input = [word_dict[w] if w in word_dict else word_dict["None"] for w in tmp]
                # input_ids, attention_mask, aligned_pos = align_pos_to_token(sent, pos, tokenizer)
                input_ids, attention_mask = align_pos_to_token(sent, tokenizer)
                chain_event.append(event_input)
                chain_sent.append(input_ids)
                chain_mask.append(attention_mask)
                # chain_pos.append(aligned_pos)
            # Choice event
            verb, subj, obj, iobj, role = choice.tuple(protagonist)
            predicate_gr = "{}:{}".format(verb, role) if protagonist != "None" else "None"
            tmp = [predicate_gr, subj, obj, iobj]
            event_input = [word_dict[w] if w in word_dict else word_dict["None"] for w in tmp]
            chain_event.append(event_input)
            # Add to list
            choice_event.append(chain_event)
            choice_sent.append(chain_sent)
            choice_mask.append(chain_mask)
            # choice_pos.append(chain_pos)
        sample_event.append(choice_event)
        sample_sent.append(choice_sent)
        sample_mask.append(choice_mask)
        # sample_pos.append(choice_pos)
    # Adding pos makes a lot of changes, ignore it.
    # return sample_event, sample_sent, sample_mask, sample_pos, target
    return sample_event, sample_sent, sample_mask, target


def generate_multi_train(corp_dir,
                         work_dir,
                         tokenized_dir,
                         # pos_dir,
                         part_size=100000,
                         file_type="tar",
                         context_size=8,
                         overwrite=False):
    """Generate multichain train data.

    :param corp_dir: train corpus directory
    :param work_dir: workspace directory
    :param tokenized_dir: tokenized raw text directory
    :param pos_dir: pos tagging directory
    :param part_size: size of each partition
    :param file_type: "tar" or "txt"
    :param context_size: length of the context chain
    :param overwrite: whether to overwrite old data
    """
    # All parts of the dataset will be store in a sub directory.
    data_dir = os.path.join(work_dir, "multi_train")
    if os.path.exists(data_dir) and not overwrite:
        logger.info("{} already exists.".format(data_dir))
    else:
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
                                         # pos_dir=pos_dir,
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
                                             stoplist=stoplist,
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


def generate_multi_eval(corp_dir,
                        work_dir,
                        tokenized_dir,
                        # pos_dir,
                        mode="dev",
                        file_type="txt",
                        context_size=8,
                        overwrite=False):
    """Generate multi chain evaluate data."""
    data_path = os.path.join(work_dir, "multi_{}".format(mode))
    if os.path.exists(data_path) and not overwrite:
        logger.info("{} already exists.".format(data_path))
    else:
        # Load stop list
        stoplist = load_stop_event(work_dir)
        # Load word dictionary
        word_dict = load_word_dict(work_dir)
        # Load tokenizer
        special_tokens = ["[subj]", "[obj]", "[iobj]"]
        tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny",
                                                      additional_special_tokens=special_tokens)
        # Make sample
        eval_data = []
        with tqdm() as pbar:
            for doc in document_iterator(corp_dir=corp_dir,
                                         tokenized_dir=tokenized_dir,
                                         # pos_dir=pos_dir,
                                         file_type=file_type,
                                         doc_type="eval"):
                # protagonist = doc.entity
                context = doc.context
                choices = doc.choices
                target = doc.target
                # Make sample
                sample = make_sample(doc=doc,
                                     choices=choices,
                                     target=target,
                                     context_size=context_size,
                                     verb_position=context[-1]["verb_position"],
                                     word_dict=word_dict,
                                     stoplist=stoplist,
                                     tokenizer=tokenizer)
                eval_data.append(sample)
                pbar.update(1)
        with open(data_path, "wb") as f:
            pickle.dump(eval_data, f)
        logger.info("Totally {} samples generated.".format(len(eval_data)))
