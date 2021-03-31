from transformers import AutoTokenizer

from sent_event_prediction.models.basic_model import BasicModel
from sent_event_prediction.models.single_chain_sent.networks import SCPredictorSent


class SingleChainSentModel(BasicModel):
    """Single chain model combines next sentence prediction."""

    def __init__(self, config_path):
        super(SingleChainSentModel, self).__init__(config_path)
        self._tokenizer = None

    def build_model(self):
        """Build model."""
        self._tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self._model = SCPredictorSent(self._config)

    def preprocess(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass
