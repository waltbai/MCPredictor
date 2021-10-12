"""Generate single chain data."""
import logging
import os
import pickle
import random
from copy import copy

from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizerFast

from mcpredictor.preprocess.negative_pool import load_negative_pool
from mcpredictor.preprocess.stop_event import load_stop_event
from mcpredictor.preprocess.word_dict import load_word_dict
from mcpredictor.utils.document import document_iterator

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
        # Replace mention and argument
        for old_ent in negative_entities:
            if old_ent is not negative_protagonist:
                # Select new entity
                if len(non_protagonist_entities) > 0:
                    new_ent = random.choice(non_protagonist_entities)
                else:
                    new_ent = old_ent
            else:
                new_ent = protagonist
            # Replace entity
            negative_event.replace_argument(old_ent, new_ent)
        negative_events.append(negative_event)
    return negative_events


def generate_mask_list(chain):
    """Generate masked words in chain."""
    masked_list = set()
    for event in chain:
        if event is not None:
            masked_list.update(event.get_words())
    return masked_list


def align_pos_to_token(words, pos, tokenizer):
    """Align pos tagging to bert tokenized result."""
    inputs = tokenizer(words,
                       is_split_into_words=True,
                       return_offsets_mapping=True,
                       padding="max_length",
                       truncation=True,
                       max_length=50)
    input_ids = inputs.pop("input_ids")
    attention_mask = inputs.pop("attention_mask")
    offset_mapping = inputs.pop("offset_mapping")
    tag_index = 0
    cur_tag = "O"
    aligned_pos = []
    for offset in offset_mapping:
        if offset[0] == 0 and offset[1] != 0 and tag_index < len(pos):
            # Begin of a new word
            cur_tag = pos[tag_index]
            tag_index += 1
            aligned_pos.append(cur_tag)
        elif offset[0] == 0 and offset[1] == 0 or tag_index >= len(pos):
            # Control tokens
            aligned_pos.append("O")
        else:
            # Subword
            aligned_pos.append(cur_tag)
    return input_ids, attention_mask, aligned_pos


def make_sample(protagonist,
                context,
                choices,
                target,
                word_dict,
                tokenizer):
    """Make sample."""
    sample_event = []
    sample_sent = []
    sample_mask = []
    sample_pos = []
    for choice_id, choice in enumerate(choices):
        # chain = context + [choice]
        chain_event = []
        chain_sent = []
        chain_mask = []
        chain_pos = []
        mask_list = generate_mask_list(context)
        # Context
        for event in context:
            if event is not None:
                verb, subj, obj, iobj, role = event.tuple(protagonist)
                predicate_gr = "{}:{}".format(verb, role)
                # Convert sentence
                tmp_mask_list = mask_list.difference(event.get_words())
                sent, pos = event.tagged_sent(role, mask_list=tmp_mask_list)
            else:
                predicate_gr = subj = obj = iobj = "None"
                sent, pos = [], []
            # Convert event
            tmp = [predicate_gr, subj, obj, iobj]
            tmp = [word_dict[w] if w in word_dict else word_dict["None"] for w in tmp]
            chain_event.append(tmp)
            input_ids, attention_mask, aligned_pos = align_pos_to_token(sent, pos, tokenizer)
            chain_sent.append(input_ids)
            chain_mask.append(attention_mask)
            chain_pos.append(aligned_pos)
        # Choice
        verb, subj, obj, iobj, role = choice.tuple(protagonist)
        predicate_gr = "{}:{}".format(verb, role)
        tmp = [predicate_gr, subj, obj, iobj]
        tmp = [word_dict[w] if w in word_dict else word_dict["None"] for w in tmp]
        chain_event.append(tmp)
        # Add to sample
        sample_event.append(chain_event)
        sample_sent.append(chain_sent)
        sample_mask.append(chain_mask)
        sample_pos.append(chain_pos)
    return sample_event, sample_sent, sample_mask, sample_pos, target


def generate_single_train(corp_dir,
                          work_dir,
                          tokenized_dir,
                          pos_dir,
                          part_size=200000,
                          file_type="tar",
                          context_size=8,
                          overwrite=False):
    """Generate single chain train data.

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
    data_dir = os.path.join(work_dir, "single_train")
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
                                         pos_dir=pos_dir,
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
                        sample = make_sample(protagonist=protagonist,
                                             context=context,
                                             choices=choices,
                                             target=target,
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


def generate_single_eval(corp_dir,
                         work_dir,
                         tokenized_dir,
                         pos_dir,
                         mode="dev",
                         file_type="txt",
                         context_size=8,
                         overwrite=False):
    """Generate single chain evaluate data."""
    data_path = os.path.join(work_dir, "single_{}".format(mode))
    if os.path.exists(data_path) and not overwrite:
        logger.info("{} already exists.".format(data_path))
    else:
        # Load stop event list
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
                                         pos_dir=pos_dir,
                                         file_type=file_type,
                                         doc_type="eval"):
                protagonist = doc.entity
                # context = doc.context
                # Context cannot be directly used, since there are slight differences
                context = doc.get_chain_for_entity(protagonist,
                                                   end_pos=doc.context[-1]["verb_position"],
                                                   stoplist=stoplist)
                if len(context) > context_size:
                    context = context[-context_size:]
                if len(context) < context_size:
                    context = [None] * (context_size - len(context)) + context
                target = doc.target
                choices = doc.choices
                # Make sample
                sample = make_sample(protagonist=protagonist,
                                     context=context,
                                     choices=choices,
                                     target=target,
                                     word_dict=word_dict,
                                     tokenizer=tokenizer)
                eval_data.append(sample)
                pbar.update(1)
        with open(data_path, "wb") as f:
            pickle.dump(eval_data, f)
        logger.info("Totally {} samples generated.".format(len(eval_data)))
