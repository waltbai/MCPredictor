"""Preprocess."""
import logging
import os

from sent_event_prediction.preprocess.multi_chain import generate_multi_train, generate_multi_eval
from sent_event_prediction.preprocess.negative_pool import generate_negative_pool
from sent_event_prediction.preprocess.single_chain import generate_single_train, generate_single_eval
from sent_event_prediction.preprocess.stop_event import count_stop_event
from sent_event_prediction.utils.config import CONFIG


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        level=logging.INFO)
    data_dir = CONFIG.data_dir
    work_dir = CONFIG.work_dir
    train_doc_dir = os.path.join(data_dir, "gigaword-nyt", "rich_docs", "training")
    dev_doc_dir = os.path.join(data_dir, "gigaword-nyt", "rich_docs", "dev")
    test_doc_dir = os.path.join(data_dir, "gigaword-nyt", "rich_docs", "test")
    tokenize_dir = os.path.join(data_dir, "gigaword-nyt", "tokenized")
    count_stop_event(train_doc_dir, work_dir)
    generate_negative_pool(corp_dir=train_doc_dir,
                           tokenized_dir=None,
                           work_dir=work_dir,
                           num_events=None,
                           suffix="train",
                           file_type="tar")
    # Single chain
    generate_single_train(corp_dir=train_doc_dir,
                          work_dir=work_dir,
                          tokenized_dir=tokenize_dir,
                          overwrite=False)
    dev_corp_dir = os.path.join(data_dir, "gigaword-nyt", "eval", "multiple_choice", "dev_10k")
    generate_single_eval(corp_dir=dev_corp_dir,
                         work_dir=work_dir,
                         tokenized_dir=tokenize_dir,
                         mode="dev",
                         overwrite=False)
    test_corp_dir = os.path.join(data_dir, "gigaword-nyt", "eval", "multiple_choice", "test_10k")
    generate_single_eval(corp_dir=test_corp_dir,
                         work_dir=work_dir,
                         tokenized_dir=tokenize_dir,
                         mode="test",
                         overwrite=False)
    # Multi chain
    generate_multi_train(corp_dir=train_doc_dir,
                         work_dir=work_dir,
                         tokenized_dir=tokenize_dir,
                         overwrite=False)
    generate_multi_eval(corp_dir=dev_corp_dir,
                        work_dir=work_dir,
                        tokenized_dir=tokenize_dir,
                        mode="dev",
                        overwrite=False)
    generate_multi_eval(corp_dir=test_corp_dir,
                        work_dir=work_dir,
                        tokenized_dir=tokenize_dir,
                        mode="test",
                        overwrite=False)

