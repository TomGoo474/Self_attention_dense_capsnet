import argparse
import numpy as np
import torch
from torchvision import datasets, transforms

from train import build_model, CAPS_MODELS

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--model", default="ours")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--test-dir", default="test")
ap.add_argument("--img-size", type=int, default=227)
ap.add_argument("--batch-size", type=int, default=64)
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- 数据
tf = transforms.Compose([transforms.Resize((args.img_size,) * 2), transforms.ToTensor()])
ds = datasets.ImageFolder(Path(args.data) / args.test_dir, tf) if False else \
     datasets.ImageFolder(f"{args.data}/{args.test_dir}", tf)
loader = torch.utils.data.DataLoader(ds, args.batch_size, shuffle=False, num_workers=2)
classes = ds.classes

# ---- 加载模型
ck = torch.load(args.ckpt, map_location=device)
model = build_model(args.model, len(classes)).to(device)
model.load_state_dict(ck["model"])
model.eval()
print(f"加载 epoch {ck['epoch']+1}, 记录准确率 {ck['acc']:.4f}, 类别 {classes}")

# ---- 推理
scores = []
with torch.no_grad():
    for x, _ in loader:
        logits, _ = model(x.to(device), None)
        logits = logits.float()
        s = logits if args.model in CAPS_MODELS else torch.softmax(logits, 1)
        scores.append(s.cpu())
scores = torch.cat(scores).numpy()
preds = scores.argmax(1)
labels = np.array(ds.targets)
print(f"测试集准确率: {(preds == labels).mean():.4f}  ({(preds == labels).sum()}/{len(labels)})")

# ---- 混淆矩阵 (行=真实, 列=预测)
k = len(classes)
cm = np.zeros((k, k), int)
for t, q in zip(labels, preds):
    cm[t, q] += 1

w = max(8, max(len(c) for c in classes) + 2)
print("\n混淆矩阵 (行=真实, 列=预测)")
print(" " * w + "".join(f"{c:>{w}}" for c in classes))
for i, c in enumerate(classes):
    print(f"{c:<{w}}" + "".join(f"{cm[i, j]:>{w}d}" for j in range(k)))

# ---- Precision / Recall / F1
tp = np.diag(cm).astype(float)
prec = np.divide(tp, cm.sum(0), out=np.zeros(k), where=cm.sum(0) > 0)
rec = np.divide(tp, cm.sum(1), out=np.zeros(k), where=cm.sum(1) > 0)
f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)

print(f"\n{'class':<{w}}{'precision':>12}{'recall':>12}{'f1':>12}{'support':>10}")
for i, c in enumerate(classes):
    print(f"{c:<{w}}{prec[i]:>12.4f}{rec[i]:>12.4f}{f1[i]:>12.4f}{cm[i].sum():>10d}")
print(f"{'macro avg':<{w}}{prec.mean():>12.4f}{rec.mean():>12.4f}{f1.mean():>12.4f}{cm.sum():>10d}")
