import torch
import torch.nn as nn

from .cnn_encoder import CNNEncoder

from .transformer_encoder import (
    TransformerEncoderBlock
)


class ExoplanetDetector(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.cnn = CNNEncoder()

        self.transformer = (
            TransformerEncoderBlock()
        )

        self.fc = nn.Sequential(

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                1
            ),

            nn.Sigmoid()
        )

    def forward(
        self,
        x
    ):

        x = self.cnn(x)

        x = x.permute(
            0,
            2,
            1
        )

        x = self.transformer(x)

        x = x.mean(
            dim=1
        )

        return self.fc(x)