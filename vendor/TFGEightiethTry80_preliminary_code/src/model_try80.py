"""Try 80 - joint prior-anchored multi-task residual GMM model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


TASKS = ("path_loss", "delay_spread", "angular_spread")
REGIONS = ("los", "nlos")


def _group_norm(channels: int, groups: int = 8) -> nn.GroupNorm:
    g = min(groups, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(max(g, 1), channels)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.gn1 = _group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = _group_norm(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride) if (in_ch != out_ch or stride != 1) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.silu(self.gn1(self.conv1(x)))
        y = self.gn2(self.conv2(y))
        return F.silu(y + self.skip(x))


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, feat_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, feat_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class SharedEncoder(nn.Module):
    def __init__(self, in_channels: int, base: int, cond_dim: int) -> None:
        super().__init__()
        chs = [base, base * 2, base * 2, base * 4, base * 4]
        self.stem = ConvBlock(in_channels, chs[0])
        self.down1 = ConvBlock(chs[0], chs[1], stride=2)
        self.down2 = ConvBlock(chs[1], chs[2], stride=2)
        self.down3 = ConvBlock(chs[2], chs[3], stride=2)
        self.down4 = ConvBlock(chs[3], chs[4], stride=2)
        self.film_mid = FiLM(cond_dim, chs[2])
        self.film_deep = FiLM(cond_dim, chs[4])
        self.out_channels = tuple(chs)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.film_mid(self.down2(x1), cond)
        x3 = self.down3(x2)
        x4 = self.film_deep(self.down4(x3), cond)
        return x0, x1, x2, x3, x4


class SharedDecoder(nn.Module):
    def __init__(self, enc_channels: Tuple[int, ...], cond_dim: int, dropout: float) -> None:
        super().__init__()
        c0, c1, c2, c3, c4 = enc_channels
        self.up4 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = ConvBlock(c3 + c3, c3)
        self.film3 = FiLM(cond_dim, c3)
        self.up3 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2)
        self.up2 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1)
        self.film1 = FiLM(cond_dim, c1)
        self.up1 = nn.ConvTranspose2d(c1, c0, 2, stride=2)
        self.dec0 = ConvBlock(c0 + c0, c0)
        self.dropout = nn.Dropout2d(dropout)
        self.out_channels = c0

    def forward(self, xs: Tuple[torch.Tensor, ...], cond: torch.Tensor) -> torch.Tensor:
        x0, x1, x2, x3, x4 = xs
        u3 = self._crop_or_pad_to(self.up4(x4), x3)
        d3 = self.film3(self.dec3(torch.cat([u3, x3], dim=1)), cond)
        u2 = self._crop_or_pad_to(self.up3(d3), x2)
        d2 = self.dec2(torch.cat([u2, x2], dim=1))
        u1 = self._crop_or_pad_to(self.up2(d2), x1)
        d1 = self.film1(self.dec1(torch.cat([u1, x1], dim=1)), cond)
        u0 = self._crop_or_pad_to(self.up1(d1), x0)
        d0 = self.dec0(torch.cat([u0, x0], dim=1))
        return self.dropout(d0)

    @staticmethod
    def _crop_or_pad_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        dh = ref.shape[-2] - x.shape[-2]
        dw = ref.shape[-1] - x.shape[-1]
        pad = [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2]
        if any(p < 0 for p in pad):
            top = max(0, -pad[2])
            bot = x.shape[-2] - max(0, -pad[3])
            left = max(0, -pad[0])
            right = x.shape[-1] - max(0, -pad[1])
            return x[..., top:bot, left:right]
        return F.pad(x, pad)


class TaskGlobalHead(nn.Module):
    def __init__(self, feat_dim: int, cond_dim: int, num_components: int) -> None:
        super().__init__()
        self.num_components = num_components
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + cond_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 2 * num_components * 3),
        )

    def forward(self, pooled: torch.Tensor, cond: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.mlp(torch.cat([pooled, cond], dim=-1))
        b = raw.shape[0]
        raw = raw.view(b, 2, self.num_components, 3)
        pi_logits = raw[..., 0]
        delta_mu_raw = raw[..., 1]
        sigma_raw = raw[..., 2]
        return {
            "pi_logits": pi_logits,
            "delta_mu_raw": delta_mu_raw,
            "sigma_raw": sigma_raw,
        }


class TaskLocalHead(nn.Module):
    def __init__(self, feat_channels: int, num_components: int) -> None:
        super().__init__()
        out_channels = 2 * (num_components + 3)
        self.net = nn.Sequential(
            ConvBlock(feat_channels, feat_channels),
            nn.Conv2d(feat_channels, out_channels, 1),
        )
        self.num_components = num_components

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.net(feat)
        b, _, h, w = raw.shape
        k = self.num_components
        raw = raw.view(b, 2, k + 3, h, w)
        return {
            "p_logits": raw[:, :, :k],
            "local_delta_raw": raw[:, :, k : k + 1],
            "alpha_logits": raw[:, :, k + 1 : k + 2],
            "sigma_tilde_raw": raw[:, :, k + 2 : k + 3],
        }


@dataclass
class Try80ModelConfig:
    in_channels: int = 9
    cond_dim: int = 128
    height_embed_dim: int = 32
    base_width: int = 96
    num_components: int = 3
    decoder_dropout: float = 0.10
    alpha_bias: float = -2.0
    sigma_min: float = 0.05
    sigma_max: float = 3.00
    path_residual_los_max: float = 2.0
    path_residual_nlos_max: float = 4.0
    delay_residual_los_max: float = 30.0
    delay_residual_nlos_max: float = 40.0
    angular_residual_los_max: float = 9.0
    angular_residual_nlos_max: float = 13.0


class Try80Model(nn.Module):
    def __init__(self, cfg: Try80ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.height_mlp = nn.Sequential(
            nn.Linear(cfg.height_embed_dim, cfg.cond_dim),
            nn.SiLU(),
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
        )
        self.encoder = SharedEncoder(cfg.in_channels, base=cfg.base_width, cond_dim=cfg.cond_dim)
        self.decoder = SharedDecoder(self.encoder.out_channels, cond_dim=cfg.cond_dim, dropout=cfg.decoder_dropout)
        feat_dim = self.encoder.out_channels[-1]
        decoder_ch = self.decoder.out_channels

        self.global_heads = nn.ModuleDict({task: TaskGlobalHead(feat_dim, cfg.cond_dim, cfg.num_components) for task in TASKS})
        self.local_heads = nn.ModuleDict({task: TaskLocalHead(decoder_ch, cfg.num_components) for task in TASKS})

        self.cond_fuse = nn.Sequential(
            nn.Linear(cfg.cond_dim * 2, cfg.cond_dim),
            nn.SiLU(),
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
        )

        self.residual_max = {
            "path_loss": torch.tensor([cfg.path_residual_los_max, cfg.path_residual_nlos_max], dtype=torch.float32),
            "delay_spread": torch.tensor([cfg.delay_residual_los_max, cfg.delay_residual_nlos_max], dtype=torch.float32),
            "angular_spread": torch.tensor([cfg.angular_residual_los_max, cfg.angular_residual_nlos_max], dtype=torch.float32),
        }

    def forward(
        self,
        inputs: torch.Tensor,
        height_embed: torch.Tensor,
        priors_trans: Dict[str, torch.Tensor],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        h_cond = self.height_mlp(height_embed)
        xs = self.encoder(inputs, h_cond)
        pooled = xs[-1].mean(dim=(-1, -2))
        cond = self.cond_fuse(torch.cat([h_cond, pooled[:, : self.cfg.cond_dim]], dim=-1))
        feat = self.decoder(xs, cond)

        los_mask = inputs[:, 1:2]
        nlos_mask = inputs[:, 2:3]
        region_masks = torch.stack([los_mask, nlos_mask], dim=1)  # (B, 2, 1, H, W)

        outputs: Dict[str, Dict[str, torch.Tensor]] = {}
        for task in TASKS:
            global_raw = self.global_heads[task](pooled, cond)
            local_raw = self.local_heads[task](feat)
            outputs[task] = self._decode_task(task, priors_trans[task], region_masks, global_raw, local_raw)
        return outputs

    def _decode_task(
        self,
        task: str,
        prior_trans: torch.Tensor,
        region_masks: torch.Tensor,
        global_raw: Dict[str, torch.Tensor],
        local_raw: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        prior_trans = torch.nan_to_num(prior_trans, nan=0.0, posinf=0.0, neginf=0.0)
        b, _, h, w = prior_trans.shape
        k = self.cfg.num_components
        residual_max_native = self.residual_max[task].to(prior_trans.device).view(1, 2, 1, 1, 1)

        pi = F.softmax(global_raw["pi_logits"], dim=-1)  # (B, 2, K)
        global_delta_native = torch.tanh(global_raw["delta_mu_raw"]).unsqueeze(-1).unsqueeze(-1) * residual_max_native
        sigma = self.cfg.sigma_min + torch.sigmoid(global_raw["sigma_raw"]) * (self.cfg.sigma_max - self.cfg.sigma_min)
        sigma = sigma.unsqueeze(-1).unsqueeze(-1)

        p = F.softmax(local_raw["p_logits"], dim=2)
        local_delta_native = torch.tanh(local_raw["local_delta_raw"]) * residual_max_native
        alpha = torch.sigmoid(local_raw["alpha_logits"] + self.cfg.alpha_bias)
        sigma_tilde = self.cfg.sigma_min + torch.sigmoid(local_raw["sigma_tilde_raw"]) * (self.cfg.sigma_max - self.cfg.sigma_min)

        prior_region = prior_trans.unsqueeze(1).expand(b, 2, k, h, w)
        delta_total_native = torch.clamp(
            global_delta_native + local_delta_native,
            min=-residual_max_native,
            max=residual_max_native,
        )
        # Residual caps are specified in native units. Spread targets are still
        # modeled with Gaussian NLL in log1p space, so their capped native means
        # are mapped back to transformed space after the residual is assembled.
        if task == "path_loss":
            global_delta = global_delta_native
            local_delta = local_delta_native
            mu = prior_region + alpha * delta_total_native
        else:
            prior_native = torch.expm1(prior_trans).clamp_min(0.0)
            prior_native_region = prior_native.unsqueeze(1).expand(b, 2, k, h, w)
            mu_native = (prior_native_region + alpha * delta_total_native).clamp_min(0.0)
            mu = torch.log1p(mu_native)
            global_anchor_native = (prior_native_region + global_delta_native).clamp_min(0.0)
            local_anchor_native = (prior_native_region + local_delta_native).clamp_min(0.0)
            global_delta = torch.log1p(global_anchor_native) - prior_region
            local_delta = torch.log1p(local_anchor_native) - prior_region
        sigma_total = torch.sqrt(torch.clamp(sigma.pow(2) + sigma_tilde.pow(2), min=1.0e-8))

        pred_region = (p * mu).sum(dim=2)
        pred = (pred_region * region_masks.squeeze(2)).sum(dim=1, keepdim=True)
        prior_blended = (prior_trans.unsqueeze(1) * region_masks).sum(dim=1)

        return {
            "pi": pi,
            "p": p,
            "global_delta": global_delta,
            "local_delta": local_delta,
            "global_delta_native": global_delta_native,
            "local_delta_native": local_delta_native,
            "alpha": alpha,
            "mu": mu,
            "sigma": sigma_total,
            "pred_trans": pred,
            "pred_region_trans": pred_region,
            "prior_trans": prior_blended,
        }
