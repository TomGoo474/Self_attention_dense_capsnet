import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sa_dense_capsnet import SADenseCapsNet, total_loss
from models_baseline import CapsNet, DCNet
from models_alexnet_vgg import AlexNet, VGG16

CAPS_MODELS = {"capsnet", "dcnet", "mdcnet", "ours"}
ALL_MODELS = ["alexnet", "vgg16", "vgg16_bn", "capsnet", "dcnet", "ours"]
ALIASES = {"vgg": "vgg16", "caps": "capsnet"}


# ---------------------------------------------------------------- 模型工厂
def build_model(name, num_classes):
    if name == "alexnet":
        return AlexNet(num_classes)
    if name == "vgg16":
        return VGG16(num_classes, use_bn=True)
    if name == "vgg16_bn":
        return VGG16(num_classes, use_bn=True)
    if name == "capsnet":
        return CapsNet(num_classes)
    if name == "dcnet":
        return DCNet(num_classes)
    if name == "ours":
        return SADenseCapsNet(num_classes)
    raise ValueError(f"未知模型 {name}, 可选: {ALL_MODELS}")


# ---------------------------------------------------------------- 数据
def build_loaders(args):
    size = args.img_size
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor()])                       # 范围 [0,1], 与模型约定一致
    test_tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

    root = Path(args.data)
    if not args.no_aug:
        tr_set = datasets.ImageFolder(root / "train", train_tf)
    else:
        tr_set = datasets.ImageFolder(root / "train", test_tf)
    te_set = datasets.ImageFolder(root / args.test_dir, test_tf)
    assert tr_set.classes == te_set.classes, \
        f"train/test 类别不一致: {tr_set.classes} vs {te_set.classes}"

    pin = torch.cuda.is_available()
    tr = DataLoader(tr_set, args.batch_size, shuffle=True, num_workers=args.workers,
                    pin_memory=pin, drop_last=len(tr_set) > args.batch_size,
                    persistent_workers=args.workers > 0)
    te = DataLoader(te_set, args.batch_size, shuffle=False, num_workers=args.workers,
                    pin_memory=pin, persistent_workers=args.workers > 0)
    return tr, te, tr_set.classes


# ---------------------------------------------------------------- 损失 / 评估
def compute_loss(model_name, logits, recon, x, y, alpha):
    if model_name in CAPS_MODELS:
        return total_loss(logits, recon, x, y, alpha)       # margin + alpha * MSE(recon)
    return F.cross_entropy(logits, y)


def run_epoch(model, loader, device, name, alpha, optimizer=None, scaler=None, amp=False):
    train = optimizer is not None
    model.train(train)
    tot_loss, correct, n = 0.0, 0, 0
    preds, labels = [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                # 训练时用真实标签做重建掩码, 测试时 y=None 用预测类别
                logits, recon = model(x, y if train else None)
            logits = logits.float()
            recon = recon.float() if recon is not None else None
            loss = compute_loss(name, logits, recon, x, y, alpha)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
            p = logits.argmax(1)
            bs = y.size(0)
            tot_loss += loss.item() * bs
            correct += (p == y).sum().item()
            n += bs
            preds.append(p.cpu())
            labels.append(y.cpu())
    return tot_loss / n, correct / n, torch.cat(preds).numpy(), torch.cat(labels).numpy()


def metrics(preds, labels, classes):
    k = len(classes)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return {"acc": float(tp.sum() / cm.sum()),
            "macro_precision": float(prec.mean()), "macro_recall": float(rec.mean()),
            "macro_f1": float(f1.mean()),
            "per_class": {c: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i])}
                          for i, c in enumerate(classes)},
            "confusion_matrix": cm.tolist()}


def evaluate_ckpt(model, ckpt_path, te, device, name, args, classes):
    """加载 checkpoint 做完整测试, 返回指标字典"""
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"])
    _, _, preds, labels = run_epoch(model, te, device, name, args.alpha, amp=args._use_amp)
    res = metrics(preds, labels, classes)
    res.update({"model": name, "checkpoint": ckpt_path.name, "epoch": ck["epoch"] + 1})
    return res


# ---------------------------------------------------------------- 单个模型的完整训练流程
def train_one(name, args, device, tr, te, classes):
    out = Path(args.out) if args.out else Path(args.data) / "runs"
    run_dir = out / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    if args.resume:
        run_dir = Path(args.resume).parent                  # 续训则写回原目录
    run_dir.mkdir(parents=True, exist_ok=True)
    save_args = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    (run_dir / "args.json").write_text(json.dumps(save_args, ensure_ascii=False, indent=2))

    model = build_model(name, len(classes)).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    lr = args.lr
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=lr * 0.01)
    use_amp = args._use_amp
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    start_ep, best_acc, best_ep = 0, 0.0, -1
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_ep, best_acc, best_ep = ck["epoch"] + 1, ck["best_acc"], ck["best_ep"]
        print(f"[resume] 从 epoch {start_ep} 继续")

    print(f"\n===== {name} | params {n_params:.2f}M | lr {lr} | amp {use_amp} | 输出 {run_dir} =====")
    log_path = run_dir / "log.csv"
    new_log = not (args.resume and log_path.exists())
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["epoch", "lr", "train_loss", "train_acc", "test_loss", "test_acc", "time_s"])

    for ep in range(start_ep, args.epochs):
        t0 = time.time()
        trl, tra, _, _ = run_epoch(model, tr, device, name, args.alpha, opt, scaler, use_amp)
        tel, tea, _, _ = run_epoch(model, te, device, name, args.alpha, amp=use_amp)
        cur_lr = opt.param_groups[0]["lr"]
        sched.step()
        dt = time.time() - t0
        flag = ""
        if tea > best_acc:
            best_acc, best_ep, flag = tea, ep, " *best"
            torch.save({"model": model.state_dict(), "classes": classes, "epoch": ep,
                        "acc": tea}, run_dir / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best_acc": best_acc, "best_ep": best_ep,
                    "classes": classes}, run_dir / "last.pt")
        log.writerow([ep + 1, f"{cur_lr:.2e}", f"{trl:.4f}", f"{tra:.4f}", f"{tel:.4f}", f"{tea:.4f}", f"{dt:.1f}"])
        log_f.flush()
        print(f"[{name}] ep {ep + 1:3d}/{args.epochs} lr {cur_lr:.1e} | "
              f"train loss {trl:.4f} acc {tra:.4f} | test loss {tel:.4f} acc {tea:.4f} | {dt:.0f}s{flag}")
    log_f.close()

    # 用最佳权重做最终测试, 输出详细指标
    res = evaluate_ckpt(model, run_dir / "best.pt", te, device, name, args, classes)
    res.update({"params_M": n_params, "best_epoch": best_ep + 1})
    (run_dir / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"[{name}] 最佳 epoch {best_ep + 1} | acc {res['acc']:.4f} | "
          f"P {res['macro_precision']:.4f} R {res['macro_recall']:.4f} F1 {res['macro_f1']:.4f}")
    return res


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="胶囊网络 / AlexNet / VGG 统一训练脚本")
    ap.add_argument("--data", required=True, help="数据根目录, 下含 train/ test/")
    ap.add_argument("--model", default="ours",
                    help=f"模型名, 可逗号分隔多个, 或 all。可选: {ALL_MODELS}")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="学习率, 所有模型统一")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=0.392, help="重建损失权重 (仅胶囊模型)")
    ap.add_argument("--img-size", type=int, default=227)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", help="混合精度 (胶囊路由可能不稳定, 默认关)")
    ap.add_argument("--no-aug", action="store_true", help="关闭数据增强")
    ap.add_argument("--test-dir", default="test", help="验证/测试集子目录名")
    ap.add_argument("--out", default="", help="输出目录, 默认 <data>/runs")
    ap.add_argument("--resume", default="", help="last.pt 路径, 续训 (仅单模型)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    names = ALL_MODELS if args.model == "all" else \
        [ALIASES.get(m.strip().lower(), m.strip().lower()) for m in args.model.split(",")]
    for m in names:
        if m not in ALL_MODELS:
            raise SystemExit(f"未知模型 {m}, 可选: {ALL_MODELS}")
    if args.resume and len(names) != 1:
        raise SystemExit("--resume 只能配合单个模型使用")
    args._use_amp = args.amp and device.type == "cuda"

    tr, te, classes = build_loaders(args)
    print(f"设备 {device} | 类别 {classes} | 训练 {len(tr.dataset)} 张 | 测试 {len(te.dataset)} 张")

    results = [train_one(m, args, device, tr, te, classes) for m in names]

    if len(results) > 1:
        out = Path(args.out) if args.out else Path(args.data) / "runs"
        summ = out / f"summary_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        with open(summ, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "params_M", "best_epoch", "acc",
                        "macro_precision", "macro_recall", "macro_f1"])
            for r in results:
                w.writerow([r["model"], f"{r['params_M']:.2f}", r["best_epoch"], f"{r['acc']:.4f}",
                            f"{r['macro_precision']:.4f}", f"{r['macro_recall']:.4f}", f"{r['macro_f1']:.4f}"])
        print(f"\n对比汇总已保存: {summ}")


if __name__ == "__main__":
    main()
