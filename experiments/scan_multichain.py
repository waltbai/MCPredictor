import os

from tqdm import tqdm

from mcpredictor.preprocess.stop_event import load_stop_event
from mcpredictor.utils.document import document_iterator

if __name__ == "__main__":
    tokenize_dir = "/home/jinxiaolong/bl/data/gandc16/gigaword-nyt/tokenized"
    data_dir = "/home/jinxiaolong/bl/data/gandc16"
    dev_corp_dir = os.path.join(data_dir, "gigaword-nyt", "eval", "multiple_choice", "dev_10k")
    stoplist = load_stop_event("/home/jinxiaolong/bl/data/sent_event_data")
    more = 0
    less = 0
    with tqdm() as pbar:
        for doc in document_iterator(corp_dir=dev_corp_dir,
                                     tokenized_dir=tokenize_dir,
                                     file_type="txt", doc_type="eval"):
            # Count single chain sents
            entity = doc.entity
            verb_position = doc.context[-1]["verb_position"]
            context = doc.get_chain_for_entity(entity, end_pos=verb_position, stoplist=stoplist)
            if len(context) > 8:
                context = context[-8:]
            sent_set = set()
            for e in context:
                sent_id = e["verb_position"][0]
                sent_set.add(sent_id)
            # Count multi chain sents
            choices = doc.choices
            # verb_position = context[-1]["verb_position"]
            target = doc.target
            tmp_sent_set = set()
            for role in ["subject", "object", "iobject"]:
                protagonist = choices[target][role]
                chain = doc.get_chain_for_entity(protagonist, end_pos=verb_position, stoplist=stoplist)
                if len(chain) > 8:
                    chain = chain[-8:]
                for e in chain:
                    sent_id = e["verb_position"][0]
                    tmp_sent_set.add(sent_id)
            if len(tmp_sent_set) > len(sent_set):
                more += 1
            elif len(tmp_sent_set) < len(sent_set):
                less += 1
            pbar.update()
    print(more, less)
