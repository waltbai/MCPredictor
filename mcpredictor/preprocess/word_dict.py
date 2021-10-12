import json
import os


def generate_word_dict(corp_dir, work_dir):
    """Generate word dictionary."""


def load_word_dict(work_dir):
    """Load word dictionary."""
    word_dict_path = os.path.join(work_dir, "word_dict.json")
    with open(word_dict_path, "r") as f:
        word_dict = json.load(f)
    return word_dict
