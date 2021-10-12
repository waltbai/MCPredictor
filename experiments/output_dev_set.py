import logging
import os
import pickle

from transformers import BertTokenizerFast

from mcpredictor.models.multi_chain_sent.model import MCSDataset
from mcpredictor.preprocess.word_dict import load_word_dict
from mcpredictor.utils.config import CONFIG


def idx2word(event, word_dict):
    return [word_dict[i.item()] for i in event]


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        level=logging.INFO)
    data_dir = CONFIG.data_dir
    work_dir = CONFIG.work_dir
    # Load original dataset`
    dev_data_path = os.path.join(work_dir, "multi_dev")
    with open(dev_data_path, "rb") as f:
        dev_data = pickle.load(f)
    dev_set = MCSDataset(dev_data)
    special_tokens = ["[subj]", "[obj]", "[iobj]"]
    tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny",
                                                  additional_special_tokens=special_tokens)
    # return map
    word_dict = load_word_dict(work_dir)
    rev_word_dict = dict([t[::-1] for t in word_dict.items()])
    os.makedirs("dev_docs", exist_ok=True)
    for idx, (events, sents, _, _) in enumerate(dev_set):
        if idx == 100:
            break
        new_events = [
            [
                [
                    idx2word(event, rev_word_dict)
                    for event in chain
                ]
                for chain in choice
            ]
            for choice in events
        ]
        new_sents = [
            [
                [
                    tokenizer.decode(event).replace("[PAD]", "").strip()
                    for event in chain
                ]
                for chain in choice
            ]
            for choice in sents
        ]
        with open("dev_docs/{}.txt".format(idx), "w") as f:
            for choice_id in range(5):
                for chain_id in range(3):
                    for event_id in range(9):
                        f.write(" ".join(new_events[choice_id][chain_id][event_id]))
                        f.write("\t")
                    f.write("\n")
                    for event_id in range(8):
                        f.write(new_sents[choice_id][chain_id][event_id])
                        f.write("\t")
                    f.write("\n")
                f.write("\n\n")
