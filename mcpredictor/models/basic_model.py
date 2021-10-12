"""Basic model."""
import json
import os
from abc import ABC, abstractmethod

import torch

from mcpredictor.utils.config import CONFIG


class BasicModel(ABC):
    """Basic model."""

    def __init__(self, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        self._config = config
        self._model = None
        self._model_name = config["model_name"]
        self._data_dir = CONFIG.data_dir
        self._work_dir = CONFIG.work_dir
        self._device = CONFIG.device
        self._logger = None

    @abstractmethod
    def train(self, train_data=None, dev_data=None):
        """Train."""

    @abstractmethod
    def evaluate(self, eval_data=None, verbose=True):
        """Evaluate."""

    @abstractmethod
    def build_model(self):
        """Build model."""

    def print_model_info(self):
        """Print model information in logger."""
        model_info = "\n".join(["{0:<15} = {1:}".format(*t) for t in self._config.items()])
        self._logger.info("\n===== Model hyper-parameters =====\n{}".format(model_info))
        self._logger.info("\n===== Model architecture =====\n{}".format(repr(self._model)))

    def load_model(self, suffix="best"):
        """Load model."""
        model_path = os.path.join(
            self._work_dir, "model",
            "{}.{}.pt".format(self._model_name, suffix))
        if os.path.exists(model_path):
            self._logger.info("Load model from {}".format(model_path))
            self._model.load_state_dict(
                torch.load(model_path, map_location=self._device), strict=False)
        else:
            self._logger.info("Fail to load model from {}".format(model_path))

    def save_model(self, suffix="best",  verbose=False):
        """Save model."""
        model_dir = os.path.join(self._work_dir, "model")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(
            model_dir, "{}.{}.pt".format(self._model_name, suffix))
        if verbose:
            self._logger.info("Save model to {}".format(model_path))
        torch.save(self._model.state_dict(), model_path)
