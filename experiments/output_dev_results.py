import logging
import os
import pickle

import numpy

from mcpredictor.models.multi_chain_sent.model import MCSDataset, MultiChainSentModel
from mcpredictor.utils.config import CONFIG

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
    # Build model
    model = MultiChainSentModel(CONFIG.model_config)
    model.build_model()
    model.print_model_info()
    model.load_model()
    prec, result = model.evaluate(dev_set)
    numpy.save("dev_result.npy", result)
