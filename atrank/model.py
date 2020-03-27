import torch

from torch import nn
import torch.nn.functional as F


def _sequence_mask(lengths, maxlen, device, dtype=torch.bool):
    mask = (
        torch.ones((len(lengths), maxlen), device=device).cumsum(dim=1).t() > lengths
    ).t()
    mask.type(dtype)
    return mask


class Model(nn.Module):

    time_embedding_size = 13

    def __init__(
        self,
        cate_list,
        item_count,
        itemid_embedding_size,
        cate_count,
        cateid_embedding_size,
        num_blocks,
        num_heads,
        device,
        **kwargs
    ):
        super(Model, self).__init__()
        hidden_units = itemid_embedding_size + cateid_embedding_size
        concat_units = hidden_units + Model.time_embedding_size
        self.num_units = hidden_units

        self.item_emb = nn.Embedding(item_count, itemid_embedding_size)  # (I, E_I)
        self.cate_emb = nn.Embedding(cate_count, cateid_embedding_size)  # (C, E_C)
        self.item_b = nn.Embedding(item_count, 1)  # (I, 1)

        self.cate_list = torch.tensor(cate_list, dtype=torch.int64, device=device)

        self.dense = nn.Linear(concat_units, hidden_units)

        self.attention_net = nn.modules.transformer.Transformer(
            d_model=hidden_units,
            nhead=num_heads,
            num_encoder_layers=num_blocks,
            num_decoder_layers=num_blocks,
            dropout=0.0,
        )

    def forward(self, i, hist_i, hist_t, sl, **kwargs):
        i_emb = torch.cat(
            (self.item_emb(i), self.cate_emb(self.cate_list[i])), 1
        )  # (N, E_I + E_C)

        h_emb = torch.cat(
            (self.item_emb(hist_i), self.cate_emb(self.cate_list[hist_i])), 2
        )  # (N, T, E_I + E_C)

        # concat time embedding
        t_emb = F.one_hot(hist_t, num_classes=Model.time_embedding_size).type(
            torch.float
        )  # (N, T, E_T)
        h_emb = torch.cat((h_emb, t_emb), -1)  # (N, T, E_I + E_C + E_T)
        h_emb = self.dense(h_emb)  # (N, T, E_h)

        h_emb = torch.transpose(h_emb, 0, 1)  # (T, N, E_h)
        src_key_padding_mask = _sequence_mask(
            sl, len(hist_i[0]), device=h_emb.device, dtype=torch.uint8
        )  # (N, T)
        u_emb = self.attention_net(
            h_emb, i_emb[None, :, :], src_key_padding_mask=src_key_padding_mask,
        )  # (1, N, C)
        u_emb = u_emb.view(-1, self.num_units)  # (N, C)

        logits = self.item_b(i).view(-1) + torch.sum(u_emb * i_emb, 1)  # (N,)
        norm = torch.norm(u_emb) ** 2.0 + torch.norm(i_emb) ** 2.0

        return logits, norm
