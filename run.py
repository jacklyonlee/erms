import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import ModelNet40
from model.pointmlp import PointMLP
from utils import compute_integrated_gradients, compute_saliency_map


def _get_pretrained_model():
    ckpt = torch.load("pointmlp.pth")
    net = PointMLP().cuda()
    net.load_state_dict(ckpt)
    return net


def _get_loader():
    return DataLoader(
        ModelNet40(partition="test", num_points=1024),
        num_workers=1,
        batch_size=16,
        shuffle=False,
        drop_last=False,
    )


@torch.no_grad()
def _get_postive_samples(net, loader, N=10):
    samples = []
    for x, y in loader:
        x, y = x.cuda(), y.squeeze().cuda()
        _, preds = net(x).max(dim=1)
        t = preds == y
        if (nt := t.sum()) > 0:
            xt, yt = x[t].cpu(), y[t].cpu()
            for i in range(nt):
                samples.append((xt[i : i + 1], yt[i : i + 1]))
                if len(samples) >= N:
                    return samples


def _get_attributions(net, samples, func, args=()):
    for x, _ in tqdm(samples):
        mask = func(net, x.cuda(), *args)
        print(mask.shape)


net = _get_pretrained_model()
loader = _get_loader()
samples = _get_postive_samples(net, loader)
_get_attributions(net, samples, compute_saliency_map)
_get_attributions(
    net, samples, compute_integrated_gradients, torch.zeros((1, 1024, 3)).cuda()
)
