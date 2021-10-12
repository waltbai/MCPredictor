import logging
import os
import pickle
import random

import numpy
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils import data
from tqdm import tqdm
from transformers import AdamW

from mcpredictor.models.basic_model import BasicModel
from mcpredictor.models.single_chain_sent.network import SCPredictorSent


class SCSDataset(data.Dataset):
    """Single Chain Dataset for mention."""

    def __init__(self, __data):
        super(SCSDataset, self).__init__()
        self.data = __data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        events, sents, masks, target = self.data[item]
        events = torch.tensor(events)
        sents = torch.tensor(sents)
        masks = torch.tensor(masks)
        target = torch.tensor(target)
        return events, sents, masks, target


class SingleChainSentModel(BasicModel):
    """Single chain model combines next sentence prediction."""

    def __init__(self, config_path):
        super(SingleChainSentModel, self).__init__(config_path)
        self._logger = logging.getLogger(__name__)

    def build_model(self):
        """Build model."""
        work_dir = self._work_dir
        device = self._device
        pretrain_embedding = numpy.load(os.path.join(work_dir, "pretrain_embedding.npy"))
        self._model = SCPredictorSent(self._config, pretrain_embedding).to(device)

    def train(self, train_data=None, dev_data=None):
        """Train."""
        # Get hyper-parameters
        work_dir = self._work_dir
        device = self._device
        npoch = self._config["npoch"]
        batch_size = self._config["batch_size"]
        lr = self._config["lr"]
        interval = self._config["interval"]
        use_sent = self._config["use_sent"]
        # Use default datasets
        dev_path = os.path.join(work_dir, "single_dev")
        with open(dev_path, "rb") as f:
            dev_set = SCSDataset(pickle.load(f))
        # Model
        model = self._model.to(device)
        # model.sent_encoder.requires_grad_(False)
        # model.sent_sequence_model.requires_grad_(False)
        # Optimizer and loss function
        param_group = [
            {
                "params": [p for n, p in model.named_parameters() if "bert" in n],
                "lr": 1e-5,
            },
            {
                "params": [p for n, p in model.named_parameters() if "bert" not in n]
            }
        ]
        # optimizer = AdamW(param_group, lr=lr, weight_decay=1e-6)
        optimizer = Adam(param_group, lr=lr, weight_decay=1e-6)
        # Train
        tmp_dir = os.path.join(work_dir, "single_train")
        # with open(os.path.join(tmp_dir, "train.0"), "rb") as f:
        #     train_set = SCSDataset(pickle.load(f)[:100000])
        best_performance = 0.
        for epoch in range(1, npoch + 1):
            self._logger.info("===== Epoch {} =====".format(epoch))
            batch_loss = []
            batch_event_loss = []
            fn_list = os.listdir(tmp_dir)
            random.shuffle(fn_list)
            for fn in fn_list:
            # if True:
                self._logger.info("Processing slice {} ...".format(fn))
                train_fp = os.path.join(tmp_dir, fn)
                with open(train_fp, "rb") as f:
                    train_set = SCSDataset(pickle.load(f))
                train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
                with tqdm(total=len(train_set)) as pbar:
                    for iteration, (events, sents, masks, target) in enumerate(train_loader):
                        events = events.to(device)
                        sents = sents.to(device)
                        masks = masks.to(device)
                        target = target.to(device)
                        model.train()
                        if use_sent:
                            event_loss = model(events=events,
                                               sents=sents,
                                               sent_mask=masks,
                                               target=target)
                        else:
                            event_loss = model(events=events, target=target)
                        loss = event_loss
                        # Get loss
                        batch_loss.append(loss.item())
                        batch_event_loss.append(event_loss.item())
                        # Update gradient
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # Evaluate on dev set
                        if (iteration + 1) % interval == 0:
                            result = self.evaluate(eval_set=dev_set, verbose=False)
                            if result > best_performance:
                                best_performance = result
                                self.save_model("best")
                        # Update progress bar
                        # pbar.set_description("Loss: {:.4f}".format(loss.item()))
                        pbar.set_description("event loss: {:.4f}, best_performance: {:.2%}".format(
                            sum(batch_event_loss) / len(batch_event_loss),
                            best_performance
                        ))
                        pbar.update(len(events))
            result = self.evaluate(eval_set=dev_set, verbose=False)
            if result > best_performance:
                best_performance = result
                self.save_model("best")
            self._logger.info("Average loss: {:.4f}".format(
                sum(batch_loss) / len(batch_loss)))
            self._logger.info("Best evaluation accuracy: {:.2%}".format(best_performance))

    def evaluate(self, eval_set=None, verbose=True):
        """Evaluate."""
        # Get hyper-parameters
        work_dir = self._work_dir
        device = self._device
        batch_size = self._config["batch_size"]
        use_sent = self._config["use_sent"]
        # Use default test data
        if eval_set is None:
            eval_path = os.path.join(work_dir, "single_test")
            with open(eval_path, "rb") as f:
                eval_set = SCSDataset(pickle.load(f))
        eval_loader = data.DataLoader(eval_set, batch_size, num_workers=8)
        # Evaluate
        model = self._model
        model.eval()
        tot, acc = 0, 0
        with torch.no_grad():
            for events, sents, masks, target in eval_loader:
                events = events.to(device)
                sents = sents.to(device)
                masks = masks.to(device)
                target = target.to(device)
                if use_sent:
                    pred = model(events=events,
                                 sents=sents,
                                 sent_mask=masks)
                else:
                    pred = model(events=events)
                acc += pred.argmax(1).eq(target).sum().item()
                tot += len(events)
        accuracy = acc / tot
        if verbose:
            self._logger.info("Evaluation accuracy: {:.2%}".format(accuracy))
        return accuracy
