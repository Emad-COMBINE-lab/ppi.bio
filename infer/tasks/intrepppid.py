from typing import Dict, Any, Optional

import numpy as np
import torch

from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import AdamW
from collections import OrderedDict


def embedding_dropout(training, embed, words, p=0.2):
    if not training:
        masked_embed_weight = embed.weight
    elif not p:
        masked_embed_weight = embed.weight
    else:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(
            1 - p
        ).expand_as(embed.weight) / (1 - p)
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


class BarlowTwinsLoss(nn.Module):
    def __init__(self, batch_size, lambda_coeff=5e-3, z_dim=128, eps=1e-5):
        super().__init__()

        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff
        self.eps = eps  # Epsilon will help avoid divide-by-zero errors

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / (torch.std(z1, dim=0) + self.eps)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / (torch.std(z2, dim=0) + self.eps)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return (on_diag + self.lambda_coeff * off_diag) / self.z_dim


# https://github.com/mourga/variational-lstm/blob/master/weight_drop.py
class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=True):
        """
        Dropout class that is paired with a torch module to make sure that the SAME mask
        will be sampled and applied to ALL timesteps.
        :param module: nn. module (e.g. nn.Linear, nn.LSTM)
        :param weights: which weights to apply dropout (names of weights of module)
        :param dropout: dropout to be applied
        :param variational: if True applies Variational Dropout, if False applies DropConnect (different masks!!!)
        """
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        """
        Smerity code I don't understand.
        """
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        """
        This function renames each 'weight name' to 'weight name' + '_raw'
        (e.g. weight_hh_l0 -> weight_hh_l0_raw)
        :return:
        """
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print("Applying weight drop of {} to {}".format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + "_raw", torch.nn.Parameter(w.data))

    def _setweights(self):
        """
        This function samples & applies a dropout mask to the weights of the recurrent layers.
        Specifically, for an LSTM, each gate has
        - a W matrix ('weight_ih') that is multiplied with the input (x_t)
        - a U matrix ('weight_hh') that is multiplied with the previous hidden state (h_t-1)
        We sample a mask (either with Variational Dropout or with DropConnect) and apply it to
        the matrices U and/or W.
        The matrices to be dropped-out are in self.weights.
        A 'weight_hh' matrix is of shape (4*nhidden, nhidden)
        while a 'weight_ih' matrix is of shape (4*nhidden, ninput).
        **** Variational Dropout ****
        With this method, we sample a mask from the tensor (4*nhidden, 1) PER ROW
        and expand it to the full matrix.
        **** DropConnect ****
        With this method, we sample a mask from the tensor (4*nhidden, nhidden) directly
        which means that we apply dropout PER ELEMENT/NEURON.
        :return:
        """
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + "_raw")
            w = None

            if self.variational:
                #######################################################
                # Variational dropout (as proposed by Gal & Ghahramani)
                #######################################################
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                #######################################################
                # DropConnect (as presented in the AWD paper)
                #######################################################
                w = torch.nn.functional.dropout(
                    raw_w, p=self.dropout, training=self.training
                )

            if not self.training:  # (*)
                w = w.data

            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


class TripletE2ENet(pl.LightningModule):
    def __init__(
        self,
        embedding_size,
        encoder,
        head,
        embedding_droprate: float,
        num_epochs: int,
        steps_per_epoch: int,
        beta_classifier: float,
        use_projection: bool,
        optimizer_type: str,
        lr: float
    ):
        super().__init__()
        self.encoder = encoder
        self.embedding_droprate = embedding_droprate
        self.classifier_criterion = nn.BCEWithLogitsLoss()
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch

        self.triplet_criterion = nn.TripletMarginLoss(margin=1.0, p=2)

        if use_projection:
            self.triplet_projection = nn.Sequential(
                nn.Mish(), nn.Linear(embedding_size, embedding_size)
            )

        self.do_rate = 0.3
        self.head = head
        self.beta_classifier = beta_classifier

        self.optimizer_type = optimizer_type
        self.lr = lr

        self.use_projection = use_projection

    def embedding_dropout(self, embed, words, p=0.2):
        return embedding_dropout(self.training, embed, words, p)

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        y_hat = self.head(z1, z2)

        return y_hat

    def step(self, batch, stage):
        p1_seq, p2_seq, omid_anchor_seq, omid_positive_seq, omid_negative_seq, y = batch

        if self.use_projection:
            z_omid_anchor = self.triplet_projection(self.encoder(omid_anchor_seq))
            z_omid_positive = self.triplet_projection(self.encoder(omid_positive_seq))
            z_omid_negative = self.triplet_projection(self.encoder(omid_negative_seq))
        else:
            z_omid_anchor = self.encoder(omid_anchor_seq)
            z_omid_positive = self.encoder(omid_positive_seq)
            z_omid_negative = self.encoder(omid_negative_seq)

        triplet_loss = self.triplet_criterion(z_omid_anchor, z_omid_positive, z_omid_negative)

        y_hat = self(p1_seq, p2_seq).squeeze(1)

        classifier_loss = self.classifier_criterion(y_hat, y.float())

        norm_beta_ssl = 1 / self.beta_classifier
        norm_beta_classifier = 1 - norm_beta_ssl

        loss = norm_beta_classifier * classifier_loss + norm_beta_ssl * triplet_loss
        #loss = classifier_loss + norm_beta_ssl

        self.log(
            f"{stage}_classifier_loss",
            classifier_loss.detach(),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            f"{stage}_triplet_loss",
            triplet_loss.detach(),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

        self.log(
            f"{stage}_classifier_loss_step",
            classifier_loss.detach(),
            on_epoch=False,
            on_step=True,
            prog_bar=False,
        )
        self.log(
            f"{stage}_triplet_loss_step",
            triplet_loss.detach(),
            on_epoch=False,
            on_step=True,
            prog_bar=False,
        )
        self.log(
            f"{stage}_loss_step", loss, on_epoch=False, on_step=True, prog_bar=False
        )

        auroc = self.auroc(y_hat, y)
        self.log(f"{stage}_auroc", auroc.detach(), on_epoch=True, on_step=False)

        ap = self.average_precision(y_hat, y)
        self.log(f"{stage}_ap", ap.detach(), on_epoch=True, on_step=False)

        mcc = self.mcc(y_hat, y)
        self.log(f"{stage}_mcc", mcc.detach(), on_epoch=True, on_step=False)

        pr = self.precision_metric(y_hat, y)
        self.log(f"{stage}_precision", pr.detach(), on_epoch=True, on_step=False)

        rec = self.recall(y_hat, y)
        self.log(f"{stage}_rec", rec.detach(), on_epoch=True, on_step=False)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=self.lr)

        return optimizer


class MLPHead(nn.Module):
    def __init__(self, embedding_size, do_rate):
        super().__init__()

        self.embedding_size = embedding_size
        self.do_rate = do_rate

        self.classify = nn.Sequential(
            OrderedDict(
                [
                    ("nl0", nn.Mish()),
                    (
                        "fc1",
                        WeightDrop(
                            nn.Linear(self.embedding_size, self.embedding_size // 2),
                            ["weight"],
                            dropout=self.do_rate,
                            variational=False,
                        ),
                    ),
                    ("nl1", nn.Mish()),
                    ("do1", nn.Dropout(p=self.do_rate)),
                    ("nl2", nn.Mish()),
                    ("do2", nn.Dropout(p=self.do_rate)),
                    (
                        "fc2",
                        WeightDrop(
                            nn.Linear(self.embedding_size // 2, 1),
                            ["weight"],
                            dropout=self.do_rate,
                            variational=False,
                        ),
                    ),
                ]
            )
        )

    def forward(self, x1, x2):

        x = (x1 + x2) / 2

        return self.classify(x)


class AWDLSTM(nn.Module):
    def __init__(
        self,
        embedding_size,
        rnn_num_layers,
        lstm_dropout_rate,
        variational_dropout,
        bi_reduce,
    ):
        super().__init__()
        self.bi_reduce = bi_reduce

        self.rnn = nn.LSTM(
            embedding_size,
            embedding_size,
            rnn_num_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.rnn_dp = WeightDrop(
            self.rnn, ["weight_hh_l0"], lstm_dropout_rate, variational_dropout
        )

        self.fc = nn.Linear(embedding_size, embedding_size)
        self.nl = nn.Mish()
        self.embedding_size = embedding_size

    def forward(self, x):
        # Truncate to longest sequence in batch
        max_len = torch.max(torch.sum(x != 0, axis=1))
        x = x[:, :max_len]

        x, (hn, cn) = self.rnn_dp(x)

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
        # x = self.nl(x)

        return x


class Projection(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers):
        super().__init__()

        diff_dim = (out_dim - in_dim) // num_layers

        layers = []

        dim = in_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim, dim + diff_dim))
            layers.append(nn.ReLU())

            dim += diff_dim

        layers.append(nn.Linear(dim, out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class BarlowEncoder(pl.LightningModule):
    def __init__(
        self,
        batch_size,
        embedder,
        encoder,
        embedding_droprate,
        num_epochs,
        steps_per_epoch,
    ):
        super().__init__()
        self.embedder = embedder
        self.embedding_droprate = embedding_droprate
        self.encoder = encoder
        self.loss_fn = BarlowTwinsLoss(
            batch_size, z_dim=self.encoder.embedding_size * 2
        )
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.projection = Projection(
            self.encoder.embedding_size, self.encoder.embedding_size * 2, 3
        )

    def embedding_dropout(self, embed, words, p=0.2):
        return embedding_dropout(self.training, embed, words, p)

    def forward(self, x):
        # Truncate to the longest sequence in batch
        max_len = torch.max(torch.sum(x != 0, axis=1))
        x = x[:, :max_len]

        x = self.embedding_dropout(self.embedder, x, p=self.embedding_droprate)
        x = self.encoder(x)

        return x

    def step(self, batch, stage):
        x_anchor, x_positive, x_negative = batch

        z_anchor = self.projection(F.relu(self(x_anchor)))
        z_positive = self.projection(F.relu(self(x_positive)))

        loss = self.loss_fn(z_anchor, z_positive)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_loss_step", loss, on_step=True, on_epoch=False)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, "val")

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, "test")

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def make_rnn_barlow_encoder(
    vocab_size: int,
    embedding_size: int,
    rnn_num_layers: int,
    rnn_dropout_rate: float,
    variational_dropout: bool,
    bi_reduce: str,
    batch_size: int,
    embedding_droprate: float,
    num_epochs: int,
    steps_per_epoch: int,
):
    embedder = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
    encoder = AWDLSTM(
        embedding_size, rnn_num_layers, rnn_dropout_rate, variational_dropout, bi_reduce
    )
    model = BarlowEncoder(
        batch_size, embedder, encoder, embedding_droprate, num_epochs, steps_per_epoch
    )

    return model


def fill_defaults(hparams: Dict[str, Any]):
    default_hparams = {
        "vocab_size": 250,
        "embedding_size": 64,
        "do_rate": 0.3,
        "rnn_num_layers": 2,
        "rnn_dropout_rate": 0.3,
        "variational_dropout": False,
        "bi_reduce": "last",
        "batch_size": 1,
        "embedding_droprate": 0.3,
        "num_epochs": 1,
        "beta_classifier": 1,
        "use_projection": True,
        "lr": 1e-2
    }

    for key in default_hparams:
        default_hparams[key] = hparams[key]

    return default_hparams


def load_chkpt(chkpt_path, hyperparams):
    chkpt = torch.load(chkpt_path, map_location=torch.device("cpu"))
    encoder = make_rnn_barlow_encoder(
        hyperparams['vocab_size'],
        hyperparams['embedding_size'],
        hyperparams['rnn_num_layers'],
        hyperparams['rnn_dropout_rate'],
        hyperparams['variational_dropout'],
        hyperparams['bi_reduce'],
        hyperparams['batch_size'],
        hyperparams['embedding_droprate'],
        hyperparams['num_epochs'],
        0,
    )

    head = MLPHead(hyperparams['embedding_size'], hyperparams['do_rate'])

    model = TripletE2ENet(
        hyperparams['embedding_size'],
        encoder,
        head,
        hyperparams['embedding_droprate'],
        hyperparams['num_epochs'],
        0,
        hyperparams['beta_classifier'],
        hyperparams['use_projection'],
        hyperparams['optimizer_type'],
        hyperparams['lr'] if 'lr' in hyperparams else 1e-2,
    )

    model.load_state_dict(chkpt["state_dict"])

    return model


def encode_seq(
    trunc_len: int, spp, seq: str
):
    seq = seq[:trunc_len]

    toks = np.array(
        spp.encode(seq, enable_sampling=False, alpha=0.1, nbest_size=-1)
    )

    pad_len = trunc_len - len(toks)
    toks = np.pad(toks, (0, pad_len), "constant")

    return torch.tensor(toks).long()


def process_seqs(spp, input_seqs, trunc_len):
    processed_seqs = []
    for input_seq in input_seqs:
        processed_seqs.append(encode_seq(trunc_len, spp, input_seq))

    return torch.vstack(processed_seqs)


def get_embeddings(model, input_seqs):
    return model(input_seqs)


def predict(model, embedding_one, embedding_two):
    logit = model.class_head(embedding_one, embedding_two).float()
    return torch.sigmoid(logit)
