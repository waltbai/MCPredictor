import os
from pprint import pprint

from tqdm import tqdm

from sent_event_prediction.utils.document import document_iterator

if __name__ == "__main__":
    # corp_dir = "/home/jinxiaolong/bl/data/gandc16/gigaword-nyt/rich_docs/training"
    tokenize_dir = "/home/jinxiaolong/bl/data/gandc16/gigaword-nyt/tokenized"
    # for doc in document_iterator(corp_dir, tokenize_dir):
    #     for entity, chain in doc.get_chains():
    #         print(entity)
    #         for event in chain:
    #             pprint(event.filter)
    #         input()
    data_dir = "/home/jinxiaolong/bl/data/gandc16"
    dev_corp_dir = os.path.join(data_dir, "gigaword-nyt", "eval", "multiple_choice", "dev_10k")
    tot, hit = 0, 0
    with tqdm() as pbar:
        for doc in document_iterator(corp_dir=dev_corp_dir,
                                     tokenized_dir=tokenize_dir,
                                     file_type="txt", doc_type="eval"):
            context = doc.context
            answer = doc.choices[doc.target]
            final_event = context[-1]
            if final_event["verb_position"][0] == answer["verb_position"][0]:
                hit += 1
            tot += 1
            pbar.update()
    print("{} / {}".format(hit, tot))
