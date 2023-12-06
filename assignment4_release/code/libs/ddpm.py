# Implementation of DDPM described in https://arxiv.org/abs/2006.11239
# Reference: https://huggingface.co/blog/annotated-diffusion

import torch
from torch import nn
import torch.nn.functional as F

from .unet import UNet


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model"""

    def __init__(
        self,
        img_shape=(3, 32, 32),
        timesteps=100,
        dim=64,
        context_dim=64,
        num_classes=10,
        dim_mults=(1, 2, 4),
    ):
        """
        Args:
            img_shape (tuple/list of int): shape of input image (C x H x W)
            timesteps (int): number of timesteps in the diffusion process
            dim (int): base feature dimension in UNet
            context_dim (int): condition dimension (embedding of the label) in UNet
            num_classes (int): number of classes used for conditioning
            dim_mults (tuple/list of int): multiplier of feature dimensions in UNet
                length of this list specifies #blockes in UNet encoder/decoder
                e.g., (1, 2, 4) -> 3 blocks with output dims of 1x, 2x, 4x
                w.r.t. the base feature dim
        """

        super().__init__()

        assert len(img_shape) == 3
        self.timesteps = timesteps
        self.img_shape = img_shape
        betas = self.linear_beta_schedule(timesteps)
        alpha_vars = self.compute_alpha_vars(betas)
        self.betas = betas
        self.sqrt_recip_alphas = alpha_vars[0]
        self.sqrt_alphas_cumprod = alpha_vars[1]
        self.sqrt_one_minus_alphas_cumprod = alpha_vars[2]
        self.posterior_variance = alpha_vars[3]

        # the denoising model using UNet (conditioned on input label)
        self.model = UNet(
            dim, context_dim, num_classes, dim_mults=dim_mults, channels=img_shape[0]
        )

    @torch.no_grad()
    def linear_beta_schedule(self, timesteps):
        """
        linear schedule as in the paper (Sec 4)
        """
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)

    @torch.no_grad()
    def compute_alpha_vars(self, betas):
        """
        compute vars related to alphas from betas
        """
        # define alphas
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        return (
            sqrt_recip_alphas,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            posterior_variance
        )

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Helper function to match the dimensions.
        It sets a[t[i]] for every element t[i] in t and expands the results
        into x_shape
        """

        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    @torch.no_grad()
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process (using the nice property from Gaussian)
        It adds noise to a starting image and outputs its noisy version at
        an arbitary time step t
        """

        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        """
        Fill in the missing code here. See Equation 4 in the paper.
        """
        # x_t =
        # return x_t

    # compute the simplified loss
    def compute_loss(self, x_start, label, t, noise=None):
        """
        Compute loss for training the model.
        Fill in the missing code here. Algorithm 1 line 5 in the paper.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # return loss

    # sampling using the denosing process (single step)
    @torch.no_grad()
    def p_sample(self, x, label, t, t_index):
        """
        Denoise a noisy image at time step t
        """

        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)

        """
        Fill in the missing code here. See Equation 11 in the paper.
        """
        # model_mean =

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(
                self.posterior_variance, t, x.shape
            )
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    # sampling using the denosing process (reversing the forward process)
    @torch.no_grad()
    def p_sample_loop(self, labels):
        """
        Sampling from DDPM (algorithm 2)
        """

        device = next(self.model.parameters()).device
        # shape of the results
        shape = [len(labels)] + list(self.img_shape)
        # start from pure noise (for each example in the batch)
        imgs = torch.randn(shape, device=device)

        # draw samples
        for i in reversed(range(0, self.timesteps)):
            imgs = self.p_sample(
                imgs,
                labels,
                torch.full((len(labels),), i, device=device, dtype=torch.long),
                i
            )
        # clip the pixel values within range
        imgs.clamp_(min=-1.0, max=1.0)
        return imgs
