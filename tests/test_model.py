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
    outputs = target._sequence_mask(length, maxlen, dtype=torch.FloatTensor)
    expected = torch.BoolTensor([[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 0]])
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

    n = target.Model(
        cate_list,
        item_count,
        itemid_embedding_size,
        cate_count,
        cateid_embedding_size,
        num_blocks,
        num_heads,
    )

    i = torch.ones(batch_size, dtype=torch.int64)
    hist_i = torch.ones(batch_size, max_length_hist, dtype=torch.int64)
    hist_t = torch.ones(batch_size, max_length_hist, dtype=torch.int64)
    sl = torch.Tensor([i for i in range(batch_size)])

    output = n(i, hist_i, hist_t, sl)
    assert list(output.shape) == [batch_size]

    loss = F.mse_loss(outputs, outputs)
    loss.backward()


def test_attentionnet(batch_size):
    num_units = 16
    num_heads = 8
    num_blocks = 2
    max_sl = batch_size

    n = target.AttentionNet(num_units, num_heads, num_blocks)

    sl = torch.Tensor(list(range(1, 1 + batch_size)))
    enc = torch.ones(batch_size, max_sl, num_units)

    dec = torch.ones(batch_size, 1, num_units)

    outputs, att_vec, stt_vec = n(enc, dec, sl)

    loss = F.mse_loss(outputs, outputs)
    loss.backward()


def test_multiheadattention(batch_size):
    num_units = 16
    num_heads = 8

    n = target.MultiheadAttention(num_units, num_heads)

    queries_length = torch.Tensor(list(range(1, 1 + batch_size)))
    max_query_length = batch_size
    queries = torch.ones(batch_size, max_query_length, num_units)

    keys_length = torch.Tensor(list(range(1, 1 + batch_size))[::-1])
    max_key_length = batch_size
    keys = torch.ones(batch_size, max_key_length, num_units)

    outputs, att_vec = n(queries, queries_length, keys, keys_length)

    assert outputs.shape == queries.shape
    assert list(att_vec.shape) == [
        batch_size * num_heads,
        max_query_length,
        max_key_length,
    ]

    loss = F.mse_loss(outputs, outputs)
    loss.backward()


def test_feedfowardnet(batch_size):
    in_channels = 512
    hidden_channels = 256
    length = 32

    n = target.FeedForwardNet(in_channels, hidden_channels)

    inputs = torch.ones(batch_size, length, in_channels)
    outputs = n(inputs)
    assert list(outputs.shape) == [batch_size, length, in_channels]

    loss = F.mse_loss(outputs, outputs)
    loss.backward()


def test_layer_norm(batch_size):
    epsilon = 1e-8
    query_size = 32

    embedding_size = 16

    inputs = torch.ones(batch_size, query_size, embedding_size)
    outputs = F.layer_norm(inputs, list(inputs.shape)[-1:], eps=epsilon)
    assert list(outputs.shape) == [batch_size, query_size, embedding_size]
