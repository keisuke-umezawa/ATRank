import atrank.model as target

import pytest

import torch
from torch import nn
import torch.nn.functional as F


@pytest.fixture
def batch_size():
    return 32


def test__sequence_mask():
    length = torch.Tensor([0, 1, 3])
    maxlen = 4
    device = "cpu"
    outputs = target._sequence_mask(
        length, maxlen, dtype=torch.FloatTensor, device=device
    )
    expected = torch.BoolTensor([[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1]])
    assert torch.equal(outputs, expected)


def test_model(batch_size):
    item_count = 32
    itemid_embedding_size = 32
    cate_count = item_count
    cateid_embedding_size = 32
    num_blocks = 1
    num_heads = 8
    cate_list = list(range(item_count))
    max_length_hist = 32
    device = "cpu"

    n = target.Model(
        cate_list,
        item_count,
        itemid_embedding_size,
        cate_count,
        cateid_embedding_size,
        num_blocks,
        num_heads,
        device,
    )

    i = torch.ones(batch_size, dtype=torch.int64)
    hist_i = torch.ones(batch_size, max_length_hist, dtype=torch.int64)
    hist_t = torch.ones(batch_size, max_length_hist, dtype=torch.int64)
    sl = torch.Tensor([i for i in range(batch_size)])

    logits, norm = n(i, hist_i, hist_t, sl)
    assert list(logits.shape) == [batch_size]

    loss = F.mse_loss(logits, logits)
    loss.backward()
