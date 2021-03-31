import os
import pickle

from tqdm import tqdm
from transformers import AutoTokenizer

from sent_event_prediction.utils.config import CONFIG

if __name__ == "__main__":
    work_dir = CONFIG.work_dir
    train_dir = os.path.join(work_dir, "single_train")
    sent_len = 0
    dist = [0] * 11
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    for fn in os.listdir(train_dir):
        with open(os.path.join(train_dir, fn), "rb") as f:
            s = pickle.load(f)
        for sample in tqdm(s):
            protagonist, context, choices, target = sample
            for e in context + choices:
                sent = e["sent"]
                sent = sent.split()
                sent = tokenizer(sent, is_split_into_words=True)["input_ids"]
                tmp_len = len(sent)
                sent_len = max(sent_len, tmp_len)
                if tmp_len // 10 < 10:
                    dist[tmp_len // 10] += 1
                else:
                    dist[10] += 1
    print(sent_len)
    print(dist)
