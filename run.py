from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import ModelNet40
from model.pointmlp import PointMLP
from plot import plot_attribution
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
def _get_postive_samples(net, loader, include_classes, N=16):
    samples = []
    for x, y in loader:
        x, y = x.cuda(), y.squeeze().cuda()
        _, preds = net(x).max(dim=1)
        if (nt := (t := preds == y).sum()) > 0:
            xt, yt = x[t].cpu(), y[t].cpu()
            for i in range(nt):
                if (classname := ModelNet40.classes[yt[i]]) in include_classes:
                    samples.append(
                        (
                            xt[i : i + 1],
                            yt[i : i + 1],
                            classname,
                        )
                    )
                if len(samples) >= N:
                    return samples


def _get_attributions(net, samples, func, filename):
    for i, (x, _, c) in enumerate(tqdm(samples)):
        a = func(net, x.cuda()).cpu()
        plot_attribution(
            x.numpy().squeeze(),
            a.numpy().squeeze(),
            f"{filename}-{c}-{i}",
        )


net = _get_pretrained_model()
loader = _get_loader()
classes_of_interest = ("airplane", "chair", "table")
samples = _get_postive_samples(net, loader, classes_of_interest)
_get_attributions(net, samples, compute_saliency_map, "./out/saliency")
_get_attributions(
    net,
    samples,
    partial(compute_integrated_gradients, baseline=torch.zeros((1, 1024, 3)).cuda()),
    "./out/ig-zeros",
)