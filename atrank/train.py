import logging
import os
import time
import json
import joblib
import pickle
import random
import numpy as np
from collections import OrderedDict, defaultdict

import torch
import torch.nn.functional as F
from torchviz import make_dot

from input import DataInput, DataInputTest
from model import Model

# For reproducibility
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


_logger = logging.getLogger(__name__)


weight_dir = "./"


def create_model(config, cate_list, device):
    _logger.info(json.dumps(config, indent=4))
    model = Model(cate_list, device=device, **config)

    return model.to(device)


def create_optimizer(model, config):
    return torch.optim.SGD(model.parameters(), lr=config["learning_rate"])


def train_model(train_set, test_set, model, optimizer, criteria, config, device):

    random.shuffle(train_set)

    batch_per_epoch = (len(train_set) // config["traing_batch_size"]) + 1
    batch_per_epoch_tst = (len(test_set) // config["test_batch_size"]) + 1

    _logger.info("--------start----------")

    history = defaultdict(list)
    for e in range(config["num_epochs"]):
        # Train
        model.train()
        cum_loss = 0.0
        for i, uij in DataInput(train_set, config["traing_batch_size"]):
            y = torch.Tensor(uij[2]).to(device)
            kwargs = dict(
                u=torch.IntTensor(uij[0]).to(device),
                i=torch.LongTensor(uij[1]).to(device),
                hist_i=torch.LongTensor(uij[3]).to(device),
                hist_t=torch.LongTensor(uij[4]).to(device),
                sl=torch.IntTensor(uij[5]).to(device),
            )

            if e == 0 and i == 1:
                _logger.debug(f"Label: {y}")
                _logger.debug(f"kwargs: {kwargs}")

            # Initialize
            optimizer.zero_grad()

            # Forward
            logits, norm = model(**kwargs)
            probs = torch.sigmoid(logits)
            loss = criteria(probs, y) + config["regulation_rate"] * norm

            if e == 0 and i == 1:
                make_dot(probs, params=dict(model.named_parameters())).render("graph")

            # Backward
            loss.backward()
            optimizer.step()
            cum_loss += loss.item()

            if i % 100 == 0:
                _logger.debug(f"{logits}")
                _logger.info(
                    f"Epoch {e + 1:02d} batch {i + 1:04d} / {batch_per_epoch}, "
                    + f"trn_loss: {cum_loss / (i + 1):.04f}"
                )

        history["trn_loss"].append(cum_loss / batch_per_epoch)

        # Test
        model.eval()
        cum_loss = 0.0
        cum_auc = 0.0
        with torch.no_grad():
            for i, uij in DataInputTest(test_set, config["test_batch_size"]):
                kwargs = dict(
                    u=torch.IntTensor(uij[0]).to(device),
                    i=torch.LongTensor(uij[1]).to(device),
                    hist_i=torch.LongTensor(uij[3]).to(device),
                    hist_t=torch.LongTensor(uij[4]).to(device),
                    sl=torch.IntTensor(uij[5]).to(device),
                )
                logits, norm = model(**kwargs)
                probs_pos = torch.sigmoid(logits)
                cum_loss += criteria(probs_pos, torch.ones_like(probs_pos)).item() * 0.5

                kwargs = dict(
                    u=torch.IntTensor(uij[0]).to(device),
                    i=torch.LongTensor(uij[2]).to(device),
                    hist_i=torch.LongTensor(uij[3]).to(device),
                    hist_t=torch.LongTensor(uij[4]).to(device),
                    sl=torch.IntTensor(uij[5]).to(device),
                )
                logits, norm = model(**kwargs)
                probs_neg = torch.sigmoid(logits)
                cum_loss += (
                    criteria(probs_neg, torch.zeros_like(probs_neg)).item() * 0.5
                )

                cum_auc += np.mean((probs_pos > probs_neg).detach().cpu().numpy())

        history["tst_auc"].append(cum_auc / batch_per_epoch_tst)
        history["tst_loss"].append(cum_loss / batch_per_epoch_tst)

        for key in ["tst_loss", "tst_auc"]:
            _logger.info(f"{key:5}: {history[key][-1]:.5f}")

        # Saving
        torch.save(model.state_dict(), f"./{e:03d}.pytorch")
        joblib.dump(history, f"{weight_dir}/history")


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _logger.info("Device: {}".format(device))

    start_time = time.time()
    with open("dataset.pkl", "rb") as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count = pickle.load(f)

    # Build Config
    config = OrderedDict()
    config["item_count"] = item_count
    config["cate_count"] = cate_count
    config["itemid_embedding_size"] = 64
    config["cateid_embedding_size"] = 64
    config["num_blocks"] = 1
    config["num_heads"] = 8
    config["learning_rate"] = 0.05
    config["num_epochs"] = 50
    config["traing_batch_size"] = 32
    config["test_batch_size"] = 128
    config["regulation_rate"] = 0.00005

    model = create_model(config, cate_list, device)
    optimizer = create_optimizer(model, config)
    criteria = F.binary_cross_entropy
    train_model(train_set, test_set, model, optimizer, criteria, config, device)


def main():
    train()


if __name__ == "__main__":
    level = logging.INFO
    handler = logging.StreamHandler()
    handler.setLevel(level)
    _logger.setLevel(level)
    _logger.addHandler(handler)
    _logger.propagate = False
    main()
