import logging
import os
import pickle

import nltk
import torch
from torch.utils import data
from tqdm import tqdm
from transformers import BertTokenizerFast

from mcpredictor.models.multi_chain_sent.model import MultiChainSentModel
from mcpredictor.models.single_chain.model import SingleChainSentModel
from mcpredictor.preprocess.multi_chain import generate_mask_list
from mcpredictor.preprocess.stop_event import load_stop_event
from mcpredictor.utils.config import CONFIG
from mcpredictor.utils.document import document_iterator


logger = logging.getLogger(__name__)


def align_tags_to_ids(words, tags, tokenizer):
    """This function aligns tags to bert tokenized results."""
    inputs = tokenizer(words,
                       is_split_into_words=True,
                       return_offsets_mapping=True,
                       padding="max_length",
                       truncation=True,
                       max_length=50)
    input_ids = inputs.pop("input_ids")
    offset_mapping = inputs.pop("offset_mapping")
    label_index = 0
    cur_label = "O"
    labels = []
    for offset in offset_mapping:
        if offset[0] == 0 and offset[1] != 0 and label_index < len(tags):
            # Begin of a new word
            cur_label = tags[label_index]
            label_index += 1
            labels.append(cur_label)
        elif offset[0] == 0 and offset[1] == 0 or label_index >= len(tags):
            # Control tokens
            labels.append("O")
        else:
            # Subword
            labels.append(cur_label)
    assert len(labels) == 50
    return labels


class MaskedDataSet(data.Dataset):
    """This dataset masks some constituents in sentence."""

    def __init__(self, data, tags):
        self.data = data
        self.tags = tags

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # Sample
        events, sents, masks, target = self.data[item]
        events = torch.tensor(events)
        sents = torch.tensor(sents)
        masks = torch.tensor(masks)
        target = torch.tensor(target)
        # Tag mask
        tags = self.tags[item]
        tags = torch.tensor(tags).to(torch.int)
        masks = torch.logical_and(masks, tags).to(torch.int)
        return events, sents, masks, target


def tag_dev(data_dir, work_dir):
    """POS tagging results."""
    dev_corp_dir = os.path.join(data_dir, "gigaword-nyt", "eval", "multiple_choice", "dev_10k")
    tokenized_dir = os.path.join(data_dir, "gigaword-nyt", "tokenized")
    pos_dir = os.path.join(data_dir, "gigaword-nyt", "candc", "tags")
    # Load stop event list
    stoplist = load_stop_event(work_dir)
    # Build tokenizer
    special_tokens = ["[subj]", "[obj]", "[iobj]"]
    tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny",
                                                  additional_special_tokens=special_tokens)
    control_tokens = special_tokens + ["[UNK]"]
    context_size = 8
    tags = []
    with tqdm() as pbar:
        for doc in document_iterator(corp_dir=dev_corp_dir,
                                     tokenized_dir=tokenized_dir,
                                     pos_dir=pos_dir,
                                     file_type="txt",
                                     doc_type="eval"):
            target = doc.target
            choices = doc.choices
            context = doc.context
            verb_position = context[-1]["verb_position"]
            # Make tags
            sample_tags = []
            for choice in choices:
                choice_tags = []
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
                    chain_tags = []
                    for event in chain:
                        if event is not None:
                            verb, subj, obj, iobj, role = event.tuple(protagonist)
                            tmp_mask_list = mask_list.difference(event.get_words())
                            sent_words, sent_tags = event.tagged_sent(role, mask_list=tmp_mask_list)
                            vi = event["verb_position"][1] + 1
                            sent_tags[vi] = "VBSelf"
                            # Align
                            # Align
                            sent_tags = align_tags_to_ids(sent_words, sent_tags, tokenizer)
                        else:
                            sent_tags = ["O"] * 50
                        chain_tags.append(sent_tags)
                    choice_tags.append(chain_tags)
                sample_tags.append(choice_tags)
            tags.append(sample_tags)
            pbar.update()
    tag_path = os.path.join(work_dir, "dev_tags")
    with open(tag_path, "wb") as f:
        pickle.dump(tags, f)
    logger.info("Dev tags save to {}".format(tag_path))


def replace_verb_self(tags):
    result = []
    verb_flag = False
    for t in tags:
        if t in ["[subj]", "[obj]", "[iobj]"]:
            result.append(t)
            verb_flag = not verb_flag
        elif verb_flag:
            result.append("VBSelf")
    return result


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        level=logging.INFO)
    data_dir = CONFIG.data_dir
    work_dir = CONFIG.work_dir
    # Load tags
    dev_tag_path = os.path.join(work_dir, "dev_tags")
    if not os.path.exists(dev_tag_path):
        tag_dev(data_dir, work_dir)
    with open(dev_tag_path, "rb") as f:
        raw_tags = pickle.load(f)
    masked_tags = [
        # "CC",   # Conjunctions
        # "JJ", "JJR", "JJS", "PDT",      # Adjectives
        # "NN", "NNS", "NNP", "NNPS",     # Nouns
        # "RB", "RBR", "RBS", "RP",       # Adverbs
        # "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",    # Verbs(Other)
        # "VBSelf",     # Verbs(Self)
    ]
    tags = []
    verb_flag = False
    for sample in raw_tags:
        sample_tags = []
        for choice in sample:
            choice_tags = []
            for chain in choice:
                chain_tags = []
                for event in chain:
                    event_tags = [t not in masked_tags for t in event]
                    chain_tags.append(event_tags)
                choice_tags.append(chain_tags)
            sample_tags.append(choice_tags)
        tags.append(sample_tags)
    # Load original dataset`
    dev_data_path = os.path.join(work_dir, "multi_dev")
    with open(dev_data_path, "rb") as f:
        dev_data = pickle.load(f)
    dev_set = MaskedDataSet(dev_data, tags)
    # Build model
    model = MultiChainSentModel(CONFIG.model_config)
    model.build_model()
    model.print_model_info()
    model.load_model()
    model.evaluate(dev_set)
