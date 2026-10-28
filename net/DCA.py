import torch
import torch.nn as nn
import torch.nn.functional as F

class TeLU(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.exp(x))


class DynamicConvolutionalAggregator(nn.Module):
    def __init__(self, k, inplanes, planes, m=4, padding=None, stride=1,
                 use_kernel_softmax=True, alpha_init=0.1, l2_norm=True, scale=1.0):
        super().__init__()
        self.k = k
        self.inplanes = inplanes
        self.planes = planes
        self.half = inplanes // 2
        self.stride = stride
        self.pad = padding
        self.use_kernel_softmax = use_kernel_softmax
        self.l2_norm = l2_norm
        self.scale = scale

        self.conv_reduce = nn.Conv2d(self.half, max(8, self.half // m), 1, bias=True)
        self.conv_kernel = nn.Conv2d(max(8, self.half // m), self.half * (k * k), 1, bias=True)

        self.conv_static = nn.Conv2d(self.half, self.half, kernel_size=3, padding=1, bias=True)

        self.fuse = nn.Conv2d(self.half * 2, planes, 1, bias=True)
        self.bn_out = nn.BatchNorm2d(planes)

        self.proj_residual = nn.Conv2d(inplanes, planes, 1, bias=False) if planes != inplanes else None

        self.conv_y = nn.Conv1d(inplanes, 2 * planes, kernel_size=1)
        self.conv_w = nn.Conv1d(planes, 2 * planes, kernel_size=1)
        self.conv_fusion = nn.Conv1d(4 * planes, 2 * planes, kernel_size=1)
        self.bn_fusion = nn.BatchNorm1d(2 * planes)
        self.relu = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        self.telu = TeLU()

        nn.init.normal_(self.conv_kernel.weight, std=1e-3)
        if self.conv_kernel.bias is not None:
            nn.init.constant_(self.conv_kernel.bias, 0.)
        nn.init.constant_(self.alpha, alpha_init)

    def forward(self, x, y):
        """
        x: [B, C, H, W],
        y: [B, C, L],
        """
        B, C, H, W = x.shape
        assert C == self.inplanes, "Input channel mismatch"

        x1 = x[:, :self.half]
        x2 = x[:, self.half:]

        g = self.avg_pool(x1)
        g = self.conv_reduce(g)
        kernel = self.conv_kernel(g).view(B, self.half, self.k * self.k)

        if self.use_kernel_softmax:
            kernel = F.softmax(kernel, dim=2)
        else:
            kernel = torch.tanh(kernel) * 0.1

        padding = (self.k - 1) // 2 if self.pad is None else self.pad
        patches = F.unfold(x1, kernel_size=self.k, padding=padding, stride=self.stride)
        patches = patches.view(B, self.half, self.k * self.k, -1)

        kernel_exp = kernel.unsqueeze(-1).expand_as(patches)
        out_dw = (patches * kernel_exp).sum(dim=2)
        H_out = (H + 2 * padding - (self.k - 1) - 1) // self.stride + 1
        W_out = (W + 2 * padding - (self.k - 1) - 1) // self.stride + 1
        out_dw = out_dw.view(B, self.half, H_out, W_out)

        y_static = self.conv_static(x2)
        y_cat = torch.cat([out_dw, y_static], dim=1)

        y_fuse = self.fuse(y_cat)
        y_fuse = self.bn_out(y_fuse)
        y_fuse = F.silu(y_fuse)

        x_res = self.proj_residual(x) if self.proj_residual is not None else x
        if x_res.shape[-2:] != y_fuse.shape[-2:]:
            x_res = F.interpolate(x_res, size=y_fuse.shape[-2:], mode='bilinear', align_corners=False)

        out = x_res + self.alpha * y_fuse

        W = out.flatten(2)  # [B, C, H*W]
        if self.l2_norm:
            W = F.normalize(W, dim=1)
        W = W * (1 / self.scale)
        W = F.softmax(W, dim=2)

        y_proj = self.conv_y(y)      # [B, 2*planes, L]
        W_proj = self.conv_w(W)      # [B, 2*planes, L]
        fused = torch.cat([y_proj, W_proj], dim=1)  # [B, 4*planes, L]
        fused = self.conv_fusion(fused)
        fused = self.bn_fusion(fused)
        fused = self.telu(fused)

        return fused


# ------------------ 测试代码 ------------------
if __name__ == "__main__":
    batch_size = 12
    in_channels = 768
    out_channels = 768
    H, W = 12, 12

    x = torch.randn(batch_size, in_channels, H, W)
    y = torch.randn(batch_size, in_channels, H*W)

    model = DynamicConvolutionalAggregator(k=3, inplanes=in_channels, planes=out_channels//4)

    z = model(x,y)

    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"z shape: {z.shape}")
