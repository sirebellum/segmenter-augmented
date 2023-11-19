import torch
import torch.nn.functional as F
from torchvision import transforms
from math import log2
from skimage.segmentation import flood_fill as skimage_flood_fill

TILE_SIZE = 512
NUM_SEGMENTS = 20

class AE(torch.nn.Module):
    def __init__(self, **kwargs):
        super(AE, self).__init__()

        # Make sure pisxel size is a power of 2
        assert log2(kwargs["pixel_size"])%1 == 0

        # Get kwargs
        self.input_shape = kwargs["input_shape"]
        self.input_height, self.input_width = self.input_shape
        self.pixel_size = kwargs["pixel_size"]
        self.decode = kwargs["decode"]

        # Compute number of layers
        self.n_layers = int(log2(kwargs["pixel_size"]))

        # Calculate vector spatial size
        self.vector_height = self.input_height // 2**self.n_layers
        self.vector_width = self.input_width // 2**self.n_layers

        self.forward = self.forward_encode

        # Encoder
        self.encoder_layers = []
        for n in range(self.n_layers):
            self.encoder_layers.append(
                torch.nn.Conv2d(
                    in_channels=kwargs["in_channels"]
                    if n == 0
                    else 2**(n+2),
                    kernel_size=(3, 3),
                    out_channels=2**(n+1+2),
                    padding="valid",
                    dtype=torch.float32,
                )
            )
            self.encoder_layers.append(torch.nn.ReLU())
            self.encoder_layers.append(
                torch.nn.MaxPool2d(
                    kernel_size=(2, 2),
                )
            )
        self.encoder = torch.nn.Sequential(*self.encoder_layers)

        # Encoded Vector
        self.encoded_layers = [
            torch.nn.Conv2d(
                in_channels=2**(self.n_layers+2),
                kernel_size=(3, 3),
                out_channels=kwargs["vectors"],
                padding="valid",
                dtype=torch.float32,
            ),
            torch.nn.Sigmoid(),
        ]
        self.encoded = torch.nn.Sequential(*self.encoded_layers)

        if self.decode:
            self.forward = self.forward_decode
            # Decoder
            self.decoder_layers = []
            for n in range(self.n_layers):
                self.decoder_layers.append(torch.nn.Upsample(scale_factor=(2, 2)))
                self.decoder_layers.append(
                    torch.nn.Conv2d(
                        in_channels=kwargs["vectors"] if n == 0
                        else 2**(self.n_layers - n + 1 + 2),
                        kernel_size=(3, 3),
                        out_channels=2**(self.n_layers - n + 2),
                        padding=(4,4),
                        dtype=torch.float32,
                    )
                )
                self.decoder_layers.append(torch.nn.ReLU())
            self.decoder = torch.nn.Sequential(*self.decoder_layers)

            # Decoded image
            self.decoded = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=2**3,
                    kernel_size=(3, 3),
                    out_channels=kwargs["in_channels"],
                    padding=(4,4),
                    dtype=torch.float32,
                ),
                torch.nn.Sigmoid(),
                transforms.CenterCrop(kwargs["input_shape"]),
            )

    def forward_decode(self, x):
        vector = self.forward_encode(x)

        # Decode
        x = self.decoder(vector)

        # Image
        y = self.decoded(x)

        return y, vector

    def forward_encode(self, x):
        # Encode
        x = self.encoder(x)

        # Vector
        vector = self.encoded(x)

        return vector