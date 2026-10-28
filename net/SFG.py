import torch
import torch.nn as nn
import torch.nn.functional as F

class TeLU(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.exp(x))

class SpatialFocusGenerator(nn.Module):
    def __init__(self, in_channels=768, kernel_size=3, reduction=4, multi_kernel=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.multi_kernel = multi_kernel
        self.telu = TeLU()

        hidden_dim = max(in_channels // reduction, 16)

        self.channel_embed = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),               # [B, C, 1, 1]
            nn.Conv2d(in_channels, hidden_dim, 1), # 降维
            # nn.ReLU(inplace=True)
            self.telu
        )

        out_dim = kernel_size**2
        if multi_kernel:
            out_dim *= 2  # 生成两套 kernel

        self.kernel_generator = nn.Conv2d(hidden_dim, out_dim, 1)

        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        B, C, H, W = x.shape

        z = self.channel_embed(x)   # [B, hidden, 1, 1]

        kernels = self.kernel_generator(z)  # [B, k*k*(multi), 1, 1]

        if self.multi_kernel:
            k = self.kernel_size
            kernels = kernels.view(B, 2, 1, k, k)   # [B, 2, 1, k, k]
            kernels = kernels.mean(1)               # 融合两套 kernel
        else:
            kernels = kernels.view(B, 1, self.kernel_size, self.kernel_size)

        x_mean = x.mean(dim=1, keepdim=True)   # [B, 1, H, W]
        x_max, _ = x.max(dim=1, keepdim=True)  # [B, 1, H, W]
        x_feat = (x_mean + x_max) / 2          # 融合

        x_feat = x_feat.view(1, B, H, W)
        att = F.conv2d(x_feat, kernels, padding=self.kernel_size // 2, groups=B)

        att = att.view(B, 1, H, W)
        att = self.bn(att)
        att = torch.sigmoid(att)

        return x * att


if __name__ == "__main__":
    x = torch.randn(1, 768, 12, 12)

    sfg = SpatialFocusGenerator(in_channels=768)

    output = sfg(x)

    print("Input shape:", x.shape)      # [B, C, H, W]
    print("Output shape:", output.shape)  # [B, C, H, W]