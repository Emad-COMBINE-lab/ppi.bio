from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from infer.tasks.weightdrop import WeightDrop


class MeanClassHead(nn.Module):
    def __init__(self, embedding_size, num_layers, weight_drop, variational):
        super(MeanClassHead, self).__init__()

        if num_layers == 1:
            self.fc = WeightDrop(
                nn.Linear(embedding_size, 1),
                ["weight"],
                dropout=weight_drop,
                variational=variational,
            )
        elif num_layers == 2:
            self.fc = nn.Sequential(
                nn.Linear(embedding_size, embedding_size // 2),
                Mish(),
                nn.Linear(embedding_size // 2, 1),
            )
        else:
            raise NotImplementedError

    def forward(self, z_a, z_b):
        z = (z_a + z_b) / 2
        z = self.fc(z)

        return z


class MultClassHead(nn.Module):
    def __init__(self, embedding_size, num_layers, weight_drop, variational):
        super(MultClassHead, self).__init__()

        if num_layers == 1:
            self.fc = WeightDrop(
                nn.Linear(embedding_size, 1),
                ["weight"],
                dropout=weight_drop,
                variational=variational,
            )
        elif num_layers == 2:
            self.fc = nn.Sequential(
                WeightDrop(
                    nn.Linear(embedding_size, embedding_size // 2),
                    ["weight"],
                    dropout=weight_drop,
                    variational=variational,
                ),
                Mish(),
                WeightDrop(
                    nn.Linear(embedding_size // 2, 1),
                    ["weight"],
                    dropout=weight_drop,
                    variational=variational,
                ),
            )
        else:
            raise NotImplementedError

        self.nl = Mish()

    def forward(self, z_a, z_b):
        z_a = (z_a - z_a.mean()) / z_a.std()
        z_b = (z_b - z_b.mean()) / z_b.std()

        z = z_a * z_b

        z = self.nl(z)
        z = self.fc(z)

        return z


class ConcatClassHead(nn.Module):
    def __init__(self, embedding_size, num_layers, weight_drop, variational):
        super(ConcatClassHead, self).__init__()

        if num_layers == 1:
            self.fc = nn.Linear(embedding_size * 2, 1)
        elif num_layers == 2:
            self.fc = nn.Sequential(
                nn.Linear(embedding_size * 2, embedding_size // 2),
                nn.Dropout(weight_drop),
                Mish(),
                nn.Linear(embedding_size // 2, 1),
            )
        else:
            raise NotImplementedError

    def forward(self, z_a, z_b):
        z_ab = torch.cat((z_a, z_b), axis=1)
        z = self.fc(z_ab)

        return z


class ManhattanClassHead(nn.Module):
    def __init__(self):
        super(ManhattanClassHead, self).__init__()

        self.fc = nn.Linear(1, 1)

    def forward(self, z_a, z_b):
        distance = torch.sum(torch.abs(z_a - z_b), dim=1).unsqueeze(1)
        y_logit = self.fc(distance)

        return y_logit


# Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
# https://arxiv.org/abs/1908.08681v1
# implemented for PyTorch / FastAI by lessw2020 (Less Wright)
# github: https://github.com/lessw2020/mish
# https://github.com/lessw2020/mish/blob/master/mish.py
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))


class LSTMAWD(nn.Module):
    def __init__(
        self,
        num_codes,
        embedding_size,
        lstm_dropout_rate,
        classhead_dropout_rate,
        rnn_num_layers,
        classhead_num_layers,
        lr,
        weight_decay,
        bi_reduce,
        class_head_name,
        variational_dropout,
        lr_scaling,
        trunc_len,
        embedding_droprate,
        frozen_epochs,
    ):
        super(LSTMAWD, self).__init__()

        if lr_scaling:
            # IMPORTANT: Manual optimization
            self.automatic_optimization = False

        self.lr_scaling = lr_scaling
        self.trunc_len = trunc_len
        self.num_codes = num_codes
        self.embedding_size = embedding_size
        self.lstm_dropout_rate = lstm_dropout_rate
        self.classhead_dropout_rate = classhead_dropout_rate
        self.rnn_num_layers = rnn_num_layers
        self.classhead_num_layers = classhead_num_layers
        self.lr = lr
        self.embedding_droprate = embedding_droprate
        self.weight_decay = weight_decay
        self.bi_reduce = bi_reduce
        self.class_head_name = class_head_name
        self.lr_base = lr
        self.fc = nn.Linear(embedding_size, embedding_size)
        self.nl = Mish()
        self.frozen_epochs = frozen_epochs

        if self.bi_reduce == "concat":
            self.rnn = nn.LSTM(
                embedding_size,
                embedding_size // 2,
                rnn_num_layers,
                bidirectional=True,
                batch_first=True,
            )
        elif self.bi_reduce in ["max", "mean", "last"]:
            self.rnn = nn.LSTM(
                embedding_size,
                embedding_size,
                rnn_num_layers,
                bidirectional=True,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unexpected value for `bi_reduce` {bi_reduce}")

        self.rnn_dp = WeightDrop(
            self.rnn, ["weight_hh_l0"], lstm_dropout_rate, variational_dropout
        )

        if class_head_name == "concat":
            self.class_head = ConcatClassHead(
                embedding_size,
                classhead_num_layers,
                classhead_dropout_rate,
                variational_dropout,
            )
        elif class_head_name == "mean":
            self.class_head = MeanClassHead(
                embedding_size,
                classhead_num_layers,
                classhead_dropout_rate,
                variational_dropout,
            )
        elif class_head_name == "mult":
            self.class_head = MultClassHead(
                embedding_size,
                classhead_num_layers,
                classhead_dropout_rate,
                variational_dropout,
            )
        elif self.class_head_name == "manhattan":
            self.class_head = ManhattanClassHead()
        else:
            raise ValueError(
                f"Unexpected value for `class_head_name` {class_head_name}"
            )

        self.criterion = nn.BCEWithLogitsLoss()
        self.embedding = nn.Embedding(
            self.num_codes, self.embedding_size, padding_idx=0
        )

    def embedding_dropout(self, embed, words, p=0.2):
        """
        Taken from original authors code.
        TODO: re-write and add test
        """
        if not self.training:
            masked_embed_weight = embed.weight
        elif not p:
            masked_embed_weight = embed.weight
        else:
            mask = embed.weight.data.new().resize_(
                (embed.weight.size(0), 1)
            ).bernoulli_(1 - p).expand_as(embed.weight) / (1 - p)
            masked_embed_weight = mask * embed.weight

        padding_idx = embed.padding_idx
        if padding_idx is None:
            padding_idx = -1

        X = F.embedding(
            words,
            masked_embed_weight,
            padding_idx,
            embed.max_norm,
            embed.norm_type,
            embed.scale_grad_by_freq,
            embed.sparse,
        )
        return X

    def forward(self, x) -> torch.TensorType:
        # Truncate to longest sequence in batch
        max_len = torch.max(torch.sum(x != 0, axis=1))
        x = x[:, :max_len]

        x = self.embedding_dropout(self.embedding, x, p=self.embedding_droprate)
        output, (hn, cn) = self.rnn_dp(x)

        if self.bi_reduce == "concat":
            # Concat both directions
            x = hn[-2:, :, :].permute(1, 0, 2).flatten(start_dim=1)
        elif self.bi_reduce == "max":
            # Max both directions
            x = torch.max(hn[-2:, :, :], dim=0).values
        elif self.bi_reduce == "mean":
            # Mean both directions
            x = torch.mean(hn[-2:, :, :], dim=0)
        elif self.bi_reduce == "last":
            # Just use last direction
            x = hn[-1:, :, :].squeeze(0)

        x = self.fc(x)
        x = self.nl(x)

        return x


def fill_defaults(hparams: Dict[str, Any]):
    default_hparams = {
        "num_codes": 250,
        "embedding_size": 64,
        "lstm_dropout_rate": None,
        "classhead_dropout_rate": None,
        "rnn_num_layers": None,
        "classhead_num_layers": None,
        "lr": None,
        "weight_decay": None,
        "bi_reduce": None,
        "class_head_name": None,
        "variational_dropout": None,
        "lr_scaling": None,
        "trunc_len": None,
        "embedding_droprate": 0.2,
        "frozen_epochs": 0,
    }

    for key in default_hparams:
        default_hparams[key] = hparams[key]

    return default_hparams


def load_chkpt(chkpt_path: Path):
    chkpt = torch.load(chkpt_path, map_location=torch.device("cpu"))
    hparams = fill_defaults(chkpt["hyper_parameters"])
    model = LSTMAWD(**hparams)
    model.load_state_dict(chkpt["state_dict"])

    return model


def encode_seq(spp, seq, trunc_len: Optional[int]):
    toks = spp.encode(seq, enable_sampling=False, alpha=0.1, nbest_size=-1)

    if trunc_len:
        pad_len = trunc_len - len(toks)
        toks = np.pad(toks, (0, pad_len), "constant")

    return torch.tensor(toks).long()


def process_seqs(spp, input_seqs, trunc_len):
    processed_seqs = []
    for input_seq in input_seqs:
        processed_seqs.append(encode_seq(spp, input_seq, trunc_len))

    return torch.vstack(processed_seqs)


def get_embeddings(model, input_seqs):
    return model(input_seqs)


def predict(model, embedding_one, embedding_two):
    logit = model.class_head(embedding_one, embedding_two).float()
    return torch.sigmoid(logit)
