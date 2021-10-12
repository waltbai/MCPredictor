"""The longest sentence is more than 500 words, thus we need to extract a span.
Following the distribution of train set, there are:
    4.4% sentences within 10 words,
    23.9% sentences between 10~20 words,
    33.3% sentences between 20~30 words,
    23.6% sentences between 30~40 words,
    10.9% sentences between 40~50 words,
    0.4% sentences more than 50 words.
Thus, we extract a span that contains 50 words (25 words before verb, 25 words after verb).

Notice: after bert tokenizer, the length will be longer than 50.
"""

import os
import pickle

from tqdm import tqdm
from transformers import AutoTokenizer

from mcpredictor.utils.config import CONFIG

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
                old_len = len(sent)
                if old_len > 50:
                    verb_position = e["verb_position"]
                    token_idx = verb_position[1]
                    sent = sent[max(0, token_idx-25):token_idx+25]
                sent = tokenizer(sent, is_split_into_words=True)["input_ids"]
                new_len = len(sent)
                sent_len = max(sent_len, new_len)
                if new_len // 10 < 10:
                    dist[new_len // 10] += 1
                else:
                    dist[10] += 1
    print(sent_len)
    print([i / sum(dist) for i in dist])
