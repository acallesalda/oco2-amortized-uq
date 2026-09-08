"""fully_connected_aux.py
The fully connected baseline with auxiliary information.

Author(s): Felix Jimenez
"""

import logging
logger = logging.getLogger(__name__)

import torch.nn as nn
from .components.encoders import FullyConnectedSpectralEncoder


class FullyConnectedAux(nn.Module):
    """
    A spectral model that encodes three input bands along with auxiliary metadata 
    using a fully connected encoder, then predicts a scalar output.

    This module flattens and concatenates the three spectral signals, their associated 
    constant features, and learned embeddings from auxiliary continuous and categorical 
    features, encodes them into a fixed-size embedding, and applies a linear predictor 
    to produce the final output.

    Args:
        input_dim (int): Dimensionality of the spectral input to the fully connected encoder.
        hidden_dims (list of int): Hidden layer sizes for the encoder network.
        embed_dim (int, optional): Size of the embedding produced by the encoder. Default is 256.

    Methods:
        embed(x): Returns the intermediate embedding before prediction.
        forward(x): Returns the scalar prediction for the input including auxiliary features.
    """
    def __init__(self, input_dim, hidden_dims, embed_dim=256):
        super().__init__()
        self.encoder = FullyConnectedSpectralEncoder(
            input_dim=input_dim, 
            hidden_dims=hidden_dims, 
            embed_dim=embed_dim,
            use_aux = True
        )
        self.predictor = nn.Linear(embed_dim, 1)
    
    def embed(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.encoder(x)
        return self.predictor(x)