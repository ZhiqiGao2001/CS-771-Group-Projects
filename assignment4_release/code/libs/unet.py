import torch
from torch import nn

from .utils import default
from .blocks import (ResBlock, SpatialTransformer, SinusoidalPE,
                     LabelEmbedding, Upsample, Downsample)


class UNet(nn.Module):
    """
    UNet as adopted by many diffusion models. This is the conditional version.
    """

    def __init__(
        self,
        dim,
        context_dim,
        num_classes,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4),
        channels=3,
        groups=8,
    ):
        super().__init__()

        # determine dimensions (input, intermediate feature dimensions)
        self.channels = channels
        init_dim = default(init_dim, dim)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # time embeddings
        time_dim = dim * 4

        # mlp for embedding time steps (we will use sinusoidal PE here)
        self.time_embd = nn.Sequential(
            SinusoidalPE(dim),
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # mlp for embedding labels, similar to time steps
        self.label_embd = nn.Sequential(
            LabelEmbedding(num_classes, dim),
            nn.Linear(dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
        )

        # initial conv layer
        self.conv_in = nn.Conv2d(
            channels, init_dim, kernel_size=3, padding=1
        )

        # layers for unet
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        num_resolutions = len(in_out)

        # encoder (ResBlock + Transformer + Downsampling)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.encoder.append(
                nn.ModuleList(
                    [
                        ResBlock(
                            dim_in,
                            out_channel=dim_out,
                            time_emb_dim=time_dim,
                            groups=groups
                        ),
                        SpatialTransformer(
                            dim_out,
                            context_dim,
                            groups=groups
                        ),
                        Downsample(dim_out, dim_out)
                        if not is_last
                        else nn.Identity(),
                    ]
                )
            )

        # middle block (ResBlock + Transformer + ResBlock)
        mid_dim = dims[-1]
        self.mid_block1 = ResBlock(
            mid_dim,
            time_emb_dim=time_dim,
            groups=groups
        )
        self.mid_attn = SpatialTransformer(
            mid_dim,
            context_dim,
            groups=groups
        )
        self.mid_block2 = ResBlock(
            mid_dim,
            time_emb_dim=time_dim,
            groups=groups
        )

        # decoder (ResBlock + Transformer + Upsampling)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == 0
            self.decoder.append(
                nn.ModuleList(
                    [
                        ResBlock(
                            # concat using unet
                            dim_out * 2,
                            out_channel=dim_in,
                            time_emb_dim=time_dim,
                            groups=groups
                        ),
                        SpatialTransformer(
                            dim_in,
                            context_dim,
                            groups=groups
                        ),
                        Upsample(dim_in, dim_in)
                        if not is_last
                        else nn.Identity(),
                    ]
                )
            )

        # final conv block
        self.out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            nn.GroupNorm(groups, dim),
            nn.SiLU(),
            nn.Conv2d(dim, self.out_dim, 1),
        )

    def forward(self, x, label, time):
        """
        Args:
            x (tensor): input image of shape B x C x H x W
            label (iterable of long): input label of size B
            time (float): input time step
        """

        # first conv
        x = self.conv_in(x)
        # time embedding
        t = self.time_embd(time)
        # label embedding
        c = self.label_embd(label).unsqueeze(1)
        # cache
        h = []

        # encoder
        for resblock, transformer, downsample in self.encoder:
            x = resblock(x, t)
            x = transformer(x, c)
            x = downsample(x)
            h.append(x)

        # middle block
        x = self.mid_block1(x, t)
        x = self.mid_attn(x, c)
        x = self.mid_block2(x, t)

        # decoder
        for resblock, transformer, upsample in self.decoder:
            x = torch.cat((x, h.pop()), dim=1)
            x = resblock(x, t)
            x = transformer(x, c)
            x = upsample(x)

        x = self.final_conv(x)
        return x
