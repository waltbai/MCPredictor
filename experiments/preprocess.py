"""Preprocess."""
import logging
import os

from sent_event_prediction.preprocess.negative_pool import generate_negative_pool
from sent_event_prediction.preprocess.single_chain import single_train
from sent_event_prediction.preprocess.stop_event import count_stop_event
from sent_event_prediction.utils.config import CONFIG


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        level=logging.INFO)
    data_dir = CONFIG.data_dir
    work_dir = CONFIG.work_dir
    corp_dir = os.path.join(data_dir, "gigaword-nyt", "rich_docs", "training")
    tokenize_dir = os.path.join(data_dir, "gigaword-nyt", "tokenized")
    count_stop_event(corp_dir, work_dir)
    generate_negative_pool(corp_dir=corp_dir,
                           tokenize_dir=tokenize_dir,
                           work_dir=work_dir,
                           # num_events=100
                           )
    single_train(corp_dir=corp_dir,
                 work_dir=work_dir,
                 tokenized_dir=tokenize_dir)