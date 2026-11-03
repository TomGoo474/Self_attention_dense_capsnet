import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- utils
def squash(x, dim=-1, eps=1e-8):
    n2 = (x ** 2).sum(dim, keepdim=True)
    return (n2 / (1 + n2)) * x / torch.sqrt(n2 + eps)


# ---------------------------------------------------------------- 1. Convolutional layer
class ConvLayer(nn.Module):
    """227 -> 114 -> 57 -> 29, 得到 conv feature map"""
    def __init__(self, out_ch=64):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 7, 2, 3, bias=False),
                                   nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.conv2 = nn.Sequential(nn.Conv2d(32, out_ch, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv2(self.pool(self.conv1(x)))


# ---------------------------------------------------------------- 2. Self-attention layer
class SelfAttention(nn.Module):
    """Q/K/V 三个 1x1 卷积分支 -> attention map -> 与 V 相乘得 self-attention map -> 残差融合"""
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.q = nn.Conv2d(ch, ch // reduction, 1)
        self.k = nn.Conv2d(ch, ch // reduction, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).flatten(2).transpose(1, 2)           # B,N,C/8
        k = self.k(x).flatten(2)                            # B,C/8,N
        attn = torch.softmax(q @ k, dim=-1)                 # attention map  B,N,N
        v = self.v(x).flatten(2)                            # B,C,N
        out = (v @ attn.transpose(1, 2)).view(B, C, H, W)  # self-attention map
        return x + self.gamma * out


# ---------------------------------------------------------------- 3. Dense block layer
class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth, bn_size=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, bn_size * growth, 1, bias=False),
            nn.BatchNorm2d(bn_size * growth), nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth, growth, 3, padding=1, bias=False))

    def forward(self, x):
        return torch.cat([x, self.net(x)], 1)


class DenseBlock(nn.Sequential):
    def __init__(self, in_ch, n_layers=6, growth=32):
        super().__init__(*[DenseLayer(in_ch + i * growth, growth) for i in range(n_layers)])
        self.out_ch = in_ch + n_layers * growth


class Transition(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
                         nn.Conv2d(in_ch, out_ch, 1, bias=False),
                         nn.AvgPool2d(2, 2))


# ---------------------------------------------------------------- 4. Locally-constrained dynamic routing
class PrimaryCaps(nn.Module):
    """卷积 -> (B, n_types, caps_dim, H, W)"""
    def __init__(self, in_ch, n_types=32, caps_dim=8, stride=2):
        super().__init__()
        self.n_types, self.caps_dim = n_types, caps_dim
        self.conv = nn.Conv2d(in_ch, n_types * caps_dim, 3, stride, 1)

    def forward(self, x):
        x = self.conv(x)
        B, _, H, W = x.shape
        x = x.view(B, self.n_types, self.caps_dim, H, W)
        return squash(x, dim=2)


class LocalCapsConv(nn.Module):
    """
    Secondary Caps Layer: 局部约束动态路由。
    每个输出位置只与 k x k 邻域内的低层胶囊相连, 权重在空间位置间共享。
    """
    def __init__(self, in_types=32, in_dim=8, out_types=32, out_dim=16,
                 k=3, stride=1, iters=3):
        super().__init__()
        self.it, self.ot, self.od, self.k, self.s, self.iters = in_types, out_types, out_dim, k, stride, iters
        self.in_dim = in_dim
        self.W = nn.Parameter(0.05 * torch.randn(in_types, k * k, out_types, out_dim, in_dim))

    def forward(self, x):                                     # B,it,in_dim,H,W
        B, it, d, H, W = x.shape
        x = x.reshape(B, it * d, H, W)
        cols = F.unfold(x, self.k, stride=self.s)              # B, it*d*k*k, L
        L = cols.shape[-1]
        cols = cols.view(B, it, d, self.k * self.k, L)
        u_hat = torch.einsum('bidkl,ikjed->bikjel', cols, self.W)  # B,it,kk,ot,od,L
        u_hat = u_hat.reshape(B, it * self.k * self.k, self.ot, self.od, L)

        b = torch.zeros(B, u_hat.shape[1], self.ot, 1, L, device=x.device)
        for r in range(self.iters):
            c = torch.softmax(b, dim=2)
            s = (c * u_hat).sum(1)                             # B,ot,od,L
            v = squash(s, dim=2)
            if r < self.iters - 1:
                b = b + (u_hat * v.unsqueeze(1)).sum(3, keepdim=True)
        Ho = (H - self.k) // self.s + 1
        return v.view(B, self.ot, self.od, Ho, -1)


class ClassCaps(nn.Module):
   
    def __init__(self, in_types=32, in_dim=16, num_classes=4, out_dim=16, iters=3,
                 auto_scale=False, target_norm=1.0):
        super().__init__()
        self.nc, self.od, self.iters = num_classes, out_dim, iters
        self.auto_scale, self.target_norm = auto_scale, target_norm
        self._calibrated = False
        self.W = nn.Parameter(0.05 * torch.randn(in_types, num_classes, out_dim, in_dim))
        if auto_scale:
            self.register_buffer("scale", torch.zeros(1))      # 0 = 尚未标定

    def forward(self, x):                                      # B,it,d,H,W
        B, it, d, H, W = x.shape
        x = x.permute(0, 1, 3, 4, 2).reshape(B, it, H * W, d)
        u_hat = torch.einsum('bind,ijed->binje', x, self.W).reshape(B, it * H * W, self.nc, self.od)
        if self.auto_scale:
            if not self._calibrated:
                if self.scale.item() == 0:                     
                    with torch.no_grad():
                        s0 = u_hat.float().sum(1) / self.nc    # 第一次路由, c = 1/num_classes
                        n = s0.norm(dim=-1).mean().clamp_min(1e-8)
                        self.scale.fill_(self.target_norm / n.item())
                self._calibrated = True
            u_hat = u_hat * self.scale
        b = torch.zeros(B, u_hat.shape[1], self.nc, device=x.device)
        for r in range(self.iters):
            c = torch.softmax(b, dim=2).unsqueeze(-1)
            v = squash((c * u_hat).sum(1), dim=-1)             # B,nc,od
            if r < self.iters - 1:
                b = b + (u_hat * v.unsqueeze(1)).sum(-1)
        return v


# ---------------------------------------------------------------- 5. Reconstruction
class Decoder(nn.Module):
    def __init__(self, num_classes=4, caps_dim=16, base=7, ch=128, out_size=227):
        super().__init__()
        self.base, self.ch, self.out_size = base, ch, out_size
        self.fc = nn.Linear(num_classes * caps_dim, ch * base * base)
        def up(i, o):
            return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False),
                                 nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.net = nn.Sequential(                 # 7 -> 14 -> 28 -> 56 -> 112 -> 224
            nn.Conv2d(ch, ch, 1),                 # Conv(1x1)
            up(ch, 160), up(160, 64), up(64, 32), up(32, 16), up(16, 8),
            nn.Conv2d(8, 3, 3, padding=1), nn.Sigmoid())

    def forward(self, v, y):
        mask = F.one_hot(y, v.shape[1]).unsqueeze(-1).float()
        z = (v * mask).flatten(1)
        x = self.fc(z).view(-1, self.ch, self.base, self.base)
        x = self.net(x)
        return F.interpolate(x, size=(self.out_size,) * 2, mode='bilinear', align_corners=False)


# ---------------------------------------------------------------- Full model
class SADenseCapsNet(nn.Module):
    def __init__(self, num_classes=4, growth=48, stem_ch=64, trans_ch=256):
        super().__init__()
        self.conv = ConvLayer(stem_ch)
        self.sa = SelfAttention(stem_ch)
        self.db1 = DenseBlock(stem_ch, 6, growth)
        self.trans = Transition(self.db1.out_ch, trans_ch)
        self.db2 = DenseBlock(trans_ch, 6, growth)
        self.bn = nn.BatchNorm2d(self.db2.out_ch)
        self.primary = PrimaryCaps(self.db2.out_ch, 32, 8, stride=2)      # 14x14 -> 7x7
        self.secondary = LocalCapsConv(32, 8, 32, 16, k=3, stride=1)      # 7x7 -> 5x5
        self.classify = ClassCaps(32, 16, num_classes, 16)
        self.decoder = Decoder(num_classes, 16)

    def forward(self, x, y=None):
        f = self.sa(self.conv(x))
        f = self.db2(self.trans(self.db1(f)))
        f = F.relu(self.bn(f))
        v = self.classify(self.secondary(self.primary(f)))    # B,num_classes,16
        logits = v.norm(dim=-1)                                # 胶囊长度 = 类别概率
        if y is None:
            y = logits.argmax(1)
        return logits, self.decoder(v, y)


def margin_loss(logits, y, m_pos=0.9, m_neg=0.1, lam=0.5):
    t = F.one_hot(y, logits.shape[1]).float()
    return (t * F.relu(m_pos - logits) ** 2 + lam * (1 - t) * F.relu(logits - m_neg) ** 2).sum(1).mean()


def total_loss(logits, recon, x, y, alpha=0.392):
    return margin_loss(logits, y) + alpha * F.mse_loss(recon, x)


if __name__ == "__main__":
    model = SADenseCapsNet(num_classes=4)
    x, y = torch.randn(2, 3, 227, 227), torch.tensor([0, 3])
    logits, recon = model(x, y)
    print("logits:", logits.shape, " recon:", recon.shape)
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")  
