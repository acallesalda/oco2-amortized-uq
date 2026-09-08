"""encoders.py
Encoders used in larger models.
Author(s): Felix Jimenez
"""

import logging
logger = logging.getLogger(__name__)
import torch.nn as nn
import torch

class BandEncoder(nn.Module):
    """
    Convolutional encoder for 1D spectral band data with auxiliary constant features.

    This module encodes a pair of inputs: a multi-channel 1D signal (e.g., spectral band data)
    and a small vector of constant features. The signal is passed through a series of 1D convolutional
    layers with batch normalization and ReLU activations, followed by max pooling and flattening.
    The resulting feature vector is concatenated with the constant features and passed through an MLP
    to produce a fixed-size embedding.

    Args:
        embed_dim (int): Dimension of the output embedding. Default is 256.
        in_channels (int): Number of input channels in the 1D signal. Default is 2.

    Inputs:
        x (Tuple[Tensor, Tensor]):
            - band: Tensor of shape [batch_size, in_channels, length], the 1D input signal.
            - const: Tensor of shape [batch_size, 2], auxiliary constant features.

    Returns:
        Tensor of shape [batch_size, embed_dim]: The encoded feature representation.
    """
    def __init__(self,embed_dim:int=256, in_channels:int = 2):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(),

            nn.Conv1d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),

            nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.mlp = nn.Sequential(
            nn.LazyLinear(embed_dim),   # infer in_features (e.g., 834) → embed_dim
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
    def forward(self, x):
        band, const = x
        if len(band.shape) < 2:
            band = band.unsqueeze(0)
        x = self.conv_layers(band)  # [batch, 256, reduced_length]
        x = self.maxpool(x)
        x = x.flatten(1)  # Flatten spatial dimensions
        x = torch.cat([x,const], axis = -1)
        return self.mlp(x)


class AuxEmbedding(nn.Module):
    """
    Embeds a set of auxiliary features composed of one continuous feature and three categorical features.

    The module maps:
      - One continuous feature to an 8-dimensional vector via a linear layer.
      - Three categorical features to 8-dimensional embeddings using `nn.Embedding` layers.

    The resulting four 8-dimensional vectors are concatenated into a single 32-dimensional embedding.

    Inputs:
        cont_feat (Tensor): Continuous feature tensor of shape [batch_size, 1].
        cat_feat (List[Tensor]): List of three categorical feature tensors, each of shape [batch_size, 1].

    Returns:
        Tensor of shape [batch_size, 32]: The combined embedding of all auxiliary features.
    """
    def __init__(self):
        super().__init__()
        self.cat1 = nn.Embedding(8, 8)
        self.cat2 = nn.Embedding(5, 8)
        self.cat3 = nn.Embedding(5, 8)
        self.fc_cont = nn.Linear(1, 8)
        self.dim = 4 * 8

    def forward(self, cont_feat, cat_feat):
        cont_emb = self.fc_cont(cont_feat)
        cat_embs = [emb(cat).squeeze(1) for emb, cat in zip([self.cat1, self.cat2, self.cat3], cat_feat)]
        all_embs = torch.cat([cont_emb] + cat_embs, dim=-1)

        return all_embs


class AuxEmbeddingV2(nn.Module):
    """
    Embeds an arbitrary set of continuous and categorical auxiliary covariates.

    The generalization of `AuxEmbedding`, which hardcodes one continuous feature and three
    categorical ones with fixed cardinalities (8, 5, 5). Each categorical covariate gets
    its own `feat_dim`-wide embedding table; the continuous block is projected as a whole
    by a single linear map, matching the original `Linear(1, 8)`.

    `AuxEmbedding` is left untouched and is still what a legacy spec builds, so the
    released checkpoints load with their original parameter names.

    Args:
        n_cont (int): Number of continuous covariates.
        cat_cardinalities (sequence of int): Number of levels of each categorical
            covariate, including the reserved "unseen level" slot.
        feat_dim (int): Width of each per-covariate block. Default 8, as in AuxEmbedding.
        cont_dim (int, optional): Width of the continuous block. Defaults to
            `feat_dim * max(1, n_cont)`.

    Returns:
        Tensor of shape [batch_size, dim], `dim = cont_dim + feat_dim * n_categorical`.
    """
    def __init__(self, n_cont: int, cat_cardinalities, feat_dim: int = 8,
                 cont_dim: int = None):
        super().__init__()
        self.n_cont = int(n_cont)
        self.cat_cardinalities = tuple(int(c) for c in cat_cardinalities)
        cont_dim = int(cont_dim) if cont_dim is not None else feat_dim * max(1, self.n_cont)
        self.cont_dim = cont_dim if self.n_cont > 0 else 0
        self.fc_cont = nn.Linear(self.n_cont, self.cont_dim) if self.n_cont > 0 else None
        self.cats = nn.ModuleList([nn.Embedding(c, feat_dim) for c in self.cat_cardinalities])
        self.dim = self.cont_dim + feat_dim * len(self.cat_cardinalities)
        if self.dim == 0:
            raise ValueError("AuxEmbeddingV2 needs at least one covariate")

    def forward(self, cont_feat, cat_feat):
        embs = []
        if self.fc_cont is not None:
            embs.append(self.fc_cont(cont_feat))
        for emb, cat in zip(self.cats, cat_feat):
            embs.append(emb(cat).squeeze(1))
        return torch.cat(embs, dim=-1)


class AuxEncoder(nn.Module):
    """
    Encodes auxiliary input features into a fixed-size embedding vector.

    With no covariate description this uses `AuxEmbedding` -- one continuous and three
    categorical features into 32 dimensions -- and is bit-identical to the published
    model. Passing `n_cont` and `cat_cardinalities` switches to `AuxEmbeddingV2`, whose
    output width is determined by the covariate set rather than by the `aux_dim` argument;
    `aux_dim` is then ignored, because a stale value in a config would otherwise silently
    truncate or mis-shape the projection.

    Args:
        aux_dim (int): Width of the combined auxiliary embedding (legacy path only).
        embed_dim (int): Output embedding dimension (default: 256).
        n_cont (int, optional): Number of continuous covariates.
        cat_cardinalities (sequence of int, optional): Levels per categorical covariate.
        feat_dim (int): Per-covariate block width for the V2 path.

    Inputs:
        aux_cont (Tensor): Continuous features, shape [batch_size, n_cont].
        aux_cat (List[Tensor]): Categorical features, each of shape [batch_size, 1].

    Returns:
        Tensor of shape [batch_size, embed_dim].
    """
    def __init__(self, aux_dim:int=32, embed_dim:int=256,
                 n_cont: int = None, cat_cardinalities=None, feat_dim: int = 8):
        super().__init__()
        if n_cont is None and cat_cardinalities is None:
            self.embedding = AuxEmbedding()
            in_dim = aux_dim
        else:
            self.embedding = AuxEmbeddingV2(
                n_cont=int(n_cont or 0),
                cat_cardinalities=tuple(cat_cardinalities or ()),
                feat_dim=feat_dim,
            )
            in_dim = self.embedding.dim
            if aux_dim is not None and int(aux_dim) != in_dim:
                logger.info(
                    "AuxEncoder: covariate set implies aux_dim=%d; ignoring the "
                    "configured value %s", in_dim, aux_dim
                )
        self.aux_dim = in_dim
        self.encoder = nn.Linear(in_dim, embed_dim)

    def forward(self, aux_cont, aux_cat):
        if len(aux_cont.shape) < 2:
            aux_cont = aux_cont.unsqueeze(0)
        aux_emb = self.embedding(aux_cont, aux_cat)
        aux = self.encoder(aux_emb)
        return aux


class FullEncoder(nn.Module):
    """
    Encodes multi-band sensor data and auxiliary features into a unified embedding tensor.

    This module combines per-band encoders (`BandEncoder`) and an auxiliary encoder (`AuxEncoder`)
    to produce a stacked embedding representation for downstream tasks.

    Args:
        aux_dim (int): Dimensionality of the concatenated auxiliary embeddings before projection (default: 32).
        embed_dim (int): Output embedding dimension for each band and the auxiliary data (default: 256).
        band_names (tuple of str): Tuple of band names used as keys in the input dictionary (default: ('b1', 'b2', 'b3')).

    Inputs:
        x (tuple):
            - bands (dict): Dictionary mapping each band name to a tuple `(band_tensor, band_constants)`,
              where `band_tensor` is a [B, C, L] tensor and `band_constants` is a [B, 2] tensor.
            - aux_cont (Tensor): Continuous auxiliary feature tensor of shape [B, 1].
            - aux_cat (List[Tensor]): List of three categorical auxiliary feature tensors, each of shape [B, 1].

    Returns:
        Tensor of shape [B, N, embed_dim]: Stacked embeddings for each band and the auxiliary input,
        where N = len(band_names) + 1.
    """
    def __init__(self, aux_dim:int=32, embed_dim:int=256,
                 n_cont: int = None, cat_cardinalities=None, feat_dim: int = 8):
        super().__init__()
        band_names = ("b1", "b2", "b3")
        self.band_names = band_names
        self.band_encoders = nn.ModuleDict({
            name: BandEncoder(embed_dim, 2) for name in band_names
        })
        self.aux_encoder = AuxEncoder(
            aux_dim, embed_dim,
            n_cont=n_cont, cat_cardinalities=cat_cardinalities, feat_dim=feat_dim,
        )

    def forward(self,x):
        bands, aux_cont, aux_cat = x
        aux = self.aux_encoder(aux_cont, aux_cat)
        band_embs = [self.band_encoders[name](bands[name]) for name in self.band_names]
        band_embs.append(aux)
        emb = torch.stack(band_embs, axis=1)
        return emb



## encoders for baselines ##

class FullyConnectedEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, embed_dim=256):
        super().__init__()
        in_dim = input_dim
        layers = []
        for h_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim), 
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU()
                )
            )
            in_dim = h_dim
        layers.append(
            nn.Sequential(
                nn.Linear(in_dim, embed_dim), 
                nn.ReLU()
            )
        )

        self.encoder = nn.Sequential(*layers) 

    def forward(self, x):
        return self.encoder(x)

class FullyConnectedSpectralEncoder(nn.Module):
    """
    Encodes spectral band inputs and optionally auxiliary metadata using a fully connected encoder.

    This module processes three spectral bands, each represented by a signal and an associated constant feature.
    The signals are flattened and concatenated with the constants, and optionally with an auxiliary embedding
    derived from categorical and continuous metadata. The resulting feature vector is passed through a
    configurable feedforward encoder to produce a fixed-size embedding.

    Args:
        input_dim (int): Dimensionality of the concatenated input to the fully connected encoder.
        hidden_dims (list of int): Hidden layer dimensions for the encoder network.
        embed_dim (int): Output embedding size.
        use_aux (bool): Whether to include auxiliary metadata via an embedding module.

    Inputs:
        x (tuple): A tuple containing:
            - bands (dict): Dictionary with keys 'b1', 'b2', 'b3', each mapping to a tuple of a spectral signal
              and a constant feature.
            - aux_cont: A tensor of continuous auxiliary metadata.
            - aux_cat: A list of categorical metadata tensors.

    Returns:
        A tensor representing the final embedded representation of the input.
    """
    def __init__(self, input_dim:int, hidden_dims:list, embed_dim:int=256, use_aux:int=False):
        super().__init__()
        self.use_aux = use_aux
        self.encoder = FullyConnectedEncoder(
            input_dim=input_dim, hidden_dims=hidden_dims, embed_dim=embed_dim
        )
        if use_aux:
            self.embedding = AuxEmbedding()

    def forward(self, x):
        bands, aux_cont, aux_cat = x
        b1, c1 = bands['b1']
        b2, c2 = bands['b2']
        b3, c3 = bands['b3']

        if len(b1.shape) < 2:
            b1 = b1.unsqueeze(0)
            b2 = b2.unsqueeze(0)
            b3 = b3.unsqueeze(0)
            aux_cont = aux_cont.unsqueeze(0)

        b = [
                b1.flatten(1),
                b2.flatten(1),
                b3.flatten(1),
                c1, c2, c3
            ]
        if self.use_aux:
            b = b + [self.embedding(aux_cont, aux_cat)]
        b = torch.cat(b, axis = 1)
        return self.encoder(b) 