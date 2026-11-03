import torch
import torch.nn as nn
import torch.nn.functional as F

from sa_dense_capsnet import (squash, ClassCaps, PrimaryCaps,
                              LocalCapsConv, SADenseCapsNet, margin_loss)


# ================================================================ 
class _Normalize(nn.Module):
    """输入 [0,1] -> ImageNet 均值方差归一化"""
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def _init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)


# ================================================================ AlexNet
class AlexNet(nn.Module):
    """227 -> conv1(11,s4) 55 -> pool 27 -> conv2 27 -> pool 13 -> conv3/4/5 13 -> pool 6
    """
    def __init__(self, num_classes=4, dropout=0.5):
        super().__init__()
        self.norm = _Normalize()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),                                                   # 55 -> 27
            nn.Conv2d(96, 256, 5, padding=2), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),                                                   # 27 -> 13
            nn.Conv2d(256, 384, 3, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))                                                   # 13 -> 6
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes))
        _init_weights(self)

    def forward(self, x, y=None):
        x = self.features(self.norm(x))
        return self.classifier(torch.flatten(x, 1)), None


# ================================================================ VGG16-BN
class VGG16(nn.Module):

    CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
           512, 512, 512, "M", 512, 512, 512, "M"]

    def __init__(self, num_classes=4, dropout=0.5):
        super().__init__()
        self.norm = _Normalize()
        layers, in_ch = [], 3
        for v in self.CFG:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [nn.Conv2d(in_ch, v, 3, padding=1, bias=False),
                           nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_ch = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(4096, num_classes))
        _init_weights(self)

    def forward(self, x, y=None):
        x = self.features(self.norm(x))
        return self.classifier(torch.flatten(x, 1)), None


# ================================================================ 
class FCDecoder(nn.Module):

    def __init__(self, num_classes=4, caps_dim=16, size=64, out_size=227, hidden=(512, 1024)):
        super().__init__()
        self.size, self.out_size = size, out_size
        self.net = nn.Sequential(
            nn.Linear(num_classes * caps_dim, hidden[0]), nn.ReLU(inplace=True),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(inplace=True),
            nn.Linear(hidden[1], 3 * size * size), nn.Sigmoid())

    def forward(self, v, y):
        mask = F.one_hot(y, v.shape[1]).unsqueeze(-1).float()
        x = self.net((v * mask).flatten(1)).view(-1, 3, self.size, self.size)
        return F.interpolate(x, size=(self.out_size,) * 2, mode="bilinear", align_corners=False)


# ================================================================ CapsNet (Sabour et al., 2017, 适配 227x227)
class CapsNet(nn.Module):
  
    def __init__(self, num_classes=4, caps_grid=8, routing_iters=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, 9, stride=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True),   # 227 -> 110
            nn.MaxPool2d(2),                                                              # 110 -> 55
            nn.Conv2d(128, 256, 5, padding=2), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.prim = nn.Conv2d(256, 32 * 8, 9, stride=2)                                   # 55 -> 24
        self.pool = nn.AdaptiveAvgPool2d(caps_grid)                                       # 24 -> caps_grid
        self.classify = ClassCaps(32, 8, num_classes, 16, iters=routing_iters, auto_scale=True)
        # 解码器缩小到 32x32 输出 + (256, 602) 隐层, 使整网参数量约 8.2M
        self.decoder = FCDecoder(num_classes, 16, size=32, hidden=(256, 602))

    def forward(self, x, y=None):
        p = self.pool(self.prim(self.conv1(x)))
        B, _, H, W = p.shape
        v = self.classify(squash(p.view(B, 32, 8, H, W), dim=2))
        logits = v.norm(dim=-1)
        if y is None:
            y = logits.argmax(1)
        return logits, self.decoder(v, y)


# ================================================================ DCNet (Phaye et al., 2018)
class DenseConcat(nn.Module):
 
    def __init__(self, in_ch, n_layers=4, growth=128):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.BatchNorm2d(in_ch + i * growth), nn.ReLU(inplace=True),
                          nn.Conv2d(in_ch + i * growth, growth, 3, padding=1, bias=False))
            for i in range(n_layers)])
        self.out_ch = in_ch + n_layers * growth

    def forward(self, x):
        feats = [x]
        for layer in self.layers:
            feats.append(layer(torch.cat(feats, 1)))
        return torch.cat(feats, 1)


class DCStem(nn.Sequential):
    """227 -> 57 -> 29"""
    def __init__(self, ch=64):
        super().__init__(nn.Conv2d(3, ch, 7, 4, 3, bias=False), nn.BatchNorm2d(ch),
                         nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))


class DCNet(nn.Module):
  
    def __init__(self, num_classes=4):
        super().__init__()
        self.stem = DCStem(64)
        self.dense = DenseConcat(64, 4, 128)
        self.prim = nn.Conv2d(self.dense.out_ch, 32 * 8, 9, stride=2)    # 29 -> 11
        self.classify = ClassCaps(32, 8, num_classes, 16, auto_scale=True)   # 输入胶囊数很大, 需标定防饱和
        self.decoder = FCDecoder(num_classes, 16)

    def forward(self, x, y=None):
        f = self.dense(self.stem(x))
        p = self.prim(f)
        B, _, H, W = p.shape
        v = self.classify(squash(p.view(B, 32, 8, H, W), dim=2))
        logits = v.norm(dim=-1)
        if y is None:
            y = logits.argmax(1)
        return logits, self.decoder(v, y)





# ================================================================ 损失函数
def total_loss_fc(logits, recon, x, y, alpha=0.392, size=64):
   
    x_s = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    r_s = F.interpolate(recon, size=(size, size), mode="bilinear", align_corners=False)
    return margin_loss(logits, y) + alpha * F.mse_loss(r_s, x_s)


# ================================================================ 工厂函数
def build_model(name, num_classes=4):
    name = name.lower()
    models = {"alexnet": AlexNet, "vgg16": VGG16, "capsnet": CapsNet, "dcnet": DCNet,
               "ours": SADenseCapsNet}
    if name not in models:
        raise ValueError(f"model must be one of {list(models)}")
    return models[name](num_classes=num_classes)


if __name__ == "__main__":
    x, y = torch.randn(2, 3, 227, 227), torch.tensor([0, 3])
    for n in ["alexnet", "vgg16", "capsnet", "dcnet", "ours"]:
        m = build_model(n).eval()
        with torch.no_grad():
            logits, recon = m(x, y)
        print(f"{n:8s} logits {tuple(logits.shape)}  recon {None if recon is None else tuple(recon.shape)}  "
              f"params {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M")
