"""Split-CIFAR-100 on frozen ResNet-18 (ImageNet) features.
Features are computed ONCE on the AWS box (`python -m flycritic.cl.data --cache`) into data/cifar100_r18_{train,test}.pt
so that every method below is a classifier/memory on identical 512-d inputs. Nothing here runs on the laptop except
`synthetic_features` (random tensors) for smoke tests."""
import os, argparse, torch

N_CLASSES, FEAT_DIM = 100, 512

def cache_features(out_dir="data", root="data/cifar100", device=None, batch=256):
    import torchvision, torchvision.transforms as T
    from torch.utils.data import DataLoader
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    net = torchvision.models.resnet18(weights=w); net.fc = torch.nn.Identity(); net.eval().to(device)
    tf = T.Compose([T.Resize(224), T.ToTensor(), T.Normalize(w.meta["mean"] if "mean" in w.meta else [0.485, 0.456, 0.406],
                                                                w.meta["std"] if "std" in w.meta else [0.229, 0.224, 0.225])])
    os.makedirs(out_dir, exist_ok=True)
    for split, train in (("train", True), ("test", False)):
        ds = torchvision.datasets.CIFAR100(root, train=train, download=True, transform=tf)
        xs, ys = [], []
        with torch.no_grad():
            for xb, yb in DataLoader(ds, batch_size=batch, num_workers=4):
                xs.append(net(xb.to(device)).float().cpu()); ys.append(yb)
        x, y = torch.cat(xs), torch.cat(ys)
        torch.save({"x": x, "y": y}, f"{out_dir}/cifar100_r18_{split}.pt"); print(split, tuple(x.shape))

def load_features(split, data_dir="data"):
    d = torch.load(f"{data_dir}/cifar100_r18_{split}.pt"); return d["x"].float(), d["y"].long()

def synthetic_features(n_tasks=3, n_classes_per_task=10, n_per_class=50, d=FEAT_DIM, seed=0, n_test_per_class=20):
    """Random class-clustered features (for smoke tests only). Returns (xtr, ytr, xte, yte) with classes 0..n-1."""
    g = torch.Generator().manual_seed(seed); C = n_tasks * n_classes_per_task
    centers = torch.randn(C, d, generator=g) * 1.5
    def make(n):
        y = torch.arange(C).repeat_interleave(n); x = centers[y] + torch.randn(len(y), d, generator=g)
        return x, y
    xtr, ytr = make(n_per_class); xte, yte = make(n_test_per_class); return xtr, ytr, xte, yte

def make_tasks(n_classes, n_tasks, seed):
    """Class-order permutation by seed -> list of class-index tensors, one per task."""
    g = torch.Generator().manual_seed(seed); perm = torch.randperm(n_classes, generator=g)
    return list(perm.view(n_tasks, -1))

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--cache", action="store_true"); p.add_argument("--device", default=None); a = p.parse_args()
    if a.cache: cache_features(device=a.device)
