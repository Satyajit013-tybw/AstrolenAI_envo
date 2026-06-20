import torch.nn as nn


class TransformerEncoderBlock(
    nn.Module
):

    def __init__(
        self,
        d_model=64
    ):

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

    def forward(
        self,
        x
    ):

        return self.transformer(x)