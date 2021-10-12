import os

from tqdm import tqdm

from mcpredictor.preprocess.stop_event import load_stop_event
from mcpredictor.utils.document import document_iterator


if __name__ == "__main__":
    corp_dir = "/home/jinxiaolong/bl/data/gandc16/gigaword-nyt/rich_docs/training"
    tokenize_dir = "/home/jinxiaolong/bl/data/gandc16/gigaword-nyt/tokenized"
    pos_dir = "/home/jinxiaolong/bl/data/gandc16/gigaword-nyt/candc/tags"
    data_dir = "/home/jinxiaolong/bl/data/gandc16"
    stoplist = load_stop_event("/home/jinxiaolong/bl/data/sent_event_data")
    for doc in document_iterator(corp_dir=corp_dir,
                                 tokenized_dir=tokenize_dir,
                                 pos_dir=pos_dir):
        for event in doc.events:
            print(event["sent"])
            print(event["pos"])
            input()
