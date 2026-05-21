import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def adaptive_kernel_size(channel, gamma=2, b=1):
    """Adaptive channel kernel size (odd), similar to ECA: k = |(log2(C)/gamma + b)| rounded to nearest odd."""
    t = int(abs((math.log2(channel) / gamma) + b))
    k = t if t % 2 == 1 else t + 1
    # Ensure kernel size is at least 1.
    return max(1, k)


class GCSEAttention(nn.Module):
    """
    Group-Channel-Spatial Excitation (GCSE).

    Design:
    - Multi-stat squeeze: channel avg, max, std
    - Channel interaction: adaptive 1D conv (ECA-style) along channel dimension
    - Spatial guidance: depthwise conv to get spatial response, then group spatial pooling
    - Residual scaling: y = x * (1 + alpha * scale), alpha is learnable for stability
    """

    def __init__(self, channels, groups=8, gamma=2, b=1, dw_kernel=3, eps=1e-6):
        """
        Args:
          - channels: input channels C
          - groups: number of channel groups for spatial guidance
          - gamma, b: hyperparameters for adaptive channel kernel size
          - dw_kernel: depthwise spatial kernel size (odd)
          - eps: numerical stability term
        """
        super().__init__()
        self.C = channels
        self.G = min(groups, channels)  # Ensure G does not exceed C.
        self.eps = eps

        # Fuse (avg, max, std) along the "stat" dimension:
        # stack to (N, 3, C), then 1x1 Conv1d -> (N, 1, C)
        self.fuse_stats = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1, bias=True)

        # Channel interaction: 1D conv over channel dimension (input shape (N, 1, C)).
        # Kernel size k adapts to C; padding keeps length unchanged.
        k = adaptive_kernel_size(channels, gamma=gamma, b=b)
        padding = (k - 1) // 2
        self.chan_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k, padding=padding, bias=False)

        # Spatial guidance path: depthwise conv per channel.
        self.spatial_dw = nn.Conv2d(channels, channels, kernel_size=dw_kernel, padding=dw_kernel // 2, groups=channels, bias=True)

        # Per-group learnable scalars.
        self.group_scalers = nn.Parameter(torch.ones(self.G), requires_grad=True)

        # Learnable residual scaling alpha (init 0 for easy hot-swap/compatibility).
        self.alpha = nn.Parameter(torch.zeros(1))

        # BN on fused channel descriptor for stability.
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        """
        Args:
          - x: (N, C, H, W)
        Returns:
          - out: (N, C, H, W)
        """
        N, C, H, W = x.shape

        # 1) Multi-stat squeeze: channel avg / max / std over spatial dims.
        gap = F.adaptive_avg_pool2d(x, 1).view(N, C)
        gmp = F.adaptive_max_pool2d(x, 1).view(N, C)
        # Channel std over spatial dims; unbiased=False for population std.
        gstd = x.view(N, C, -1).std(dim=2, unbiased=False)

        # Stack stats -> (N, 3, C)
        stats = torch.stack([gap, gmp, gstd], dim=1)
        # 1x1 Conv1d fuse -> (N, 1, C) then squeeze -> (N, C)
        desc = self.fuse_stats(stats).squeeze(1)

        # Optional normalization: BN over channel descriptor.
        desc_bn = self.bn(desc.unsqueeze(-1)).squeeze(-1)

        # 2) Channel-local interaction: (N, C) -> (N, 1, C) -> 1D conv
        d = desc_bn.unsqueeze(1)
        e = self.chan_conv(d).squeeze(1)  # (N, C) with local channel correlation

        # 3) Spatial guidance: depthwise conv -> group spatial pooling
        s_map = self.spatial_dw(x)

        G = self.G
        if G == 1:
            # Single group: spatial mean then channel mean -> (N, 1)
            grp = s_map.view(N, C, -1).mean(dim=2).mean(dim=1, keepdim=True)
            grp_expand = grp.repeat(1, C)
        else:
            # Split channels into G groups; per-group spatial mean then channel mean -> (N, G)
            per_group = (C + G - 1) // G
            grp_vals = []
            for g in range(G):
                start = g * per_group
                end = min(start + per_group, C)
                if start >= end:
                    # Rare corner case: empty group.
                    grp_vals.append(torch.zeros(N, device=x.device))
                else:
                    sub = s_map[:, start:end, :, :].contiguous().view(N, end - start, -1)
                    gmean = sub.mean(dim=2).mean(dim=1)
                    grp_vals.append(gmean)

            # Stack to (N, G)
            grp_stack = torch.stack(grp_vals, dim=1)
            # Scale per group (G,) -> (1, G) -> (N, G)
            grp_scaled = grp_stack * self.group_scalers.unsqueeze(0)

            # Expand group scalars back to channel dimension.
            grp_expand = x.new_zeros((N, C))
            for g in range(G):
                start = g * per_group
                end = min(start + per_group, C)
                if start < end:
                    grp_expand[:, start:end] = grp_scaled[:, g:g + 1].repeat(1, end - start)

        # 4) Combine and activate: e * grp_expand -> sigmoid -> scale
        combined = e * grp_expand
        scale = torch.sigmoid(combined).view(N, C, 1, 1)

        # Residual scaling output.
        out = x * (1.0 + self.alpha * scale)

        return out
