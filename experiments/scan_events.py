from pprint import pprint

from sent_event_prediction.utils.document import document_iterator

if __name__ == "__main__":
    corp_dir = "/home/jinxiaolong/bl/data/gandc16/gigaword-nyt/rich_docs/training"
    tokenize_dir = "/home/jinxiaolong/bl/data/gandc16/gigaword-nyt/tokenized"
    for doc in document_iterator(corp_dir, tokenize_dir):
        for entity, chain in doc.get_chains():
            print(entity)
            for event in chain:
                pprint(event.filter)
            input()
