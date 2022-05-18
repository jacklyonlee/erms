from functools import partial

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

import erms
from data import ModelNet40
from model.pointmlp import PointMLP
from plot import plot_pc


def _get_pretrained_model():
    ckpt = torch.load("pointmlp.pth")
    net = PointMLP().cuda()
    net.load_state_dict(ckpt)
    return net.eval()


def _get_loader():
    return DataLoader(
        ModelNet40(partition="test", num_points=1024),
        num_workers=1,
        batch_size=16,
        shuffle=False,
        drop_last=False,
    )


@torch.no_grad()
def _get_postive_samples(net, loader, include_classes, N=32):
    samples = []
    for x, y in loader:
        x, y = x.cuda(), y.squeeze().cuda()
        _, p = net(x).max(dim=1)
        if (nt := (t := p == y).sum()) > 0:
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
        plot_pc(
            x.numpy().squeeze(),
            f"{filename}-{c}-{i}",
            mask=a.numpy().squeeze(),
        )


def _get_baseline(name):
    return {
        "zeros": lambda: torch.zeros((1, 1024, 3)),
        "randn": lambda: torch.randn(1, 1024, 3) / 4,
        "sphere": (
            lambda: (
                (x := torch.randn(1, 1024, 3))
                / x.norm(
                    dim=-1,
                    keepdim=True,
                )
            )
            / 2
        ),
    }[name]().cuda()


def _get_attacks(net, samples, func, filename):
    for i, (x, y, c) in enumerate(tqdm(samples)):
        x, y = x.cuda(), y.cuda()
        x_adv = func(net, x, y)
        with torch.no_grad():
            _, p_adv = net(x_adv).max(dim=1)
        if p_adv != y:
            plot_pc(
                x.cpu().numpy().squeeze(),
                f"{filename}-{c}-{i}-x",
            )
            plot_pc(
                x_adv.cpu().numpy().squeeze(),
                f"{filename}-{c}-{i}-x_adv",
                mask=(x_adv - x).cpu().numpy().squeeze(),
            )


def _eval_attacks(net, loader, func):
    Y, P, P_adv = [], [], []
    norms = ((0, []), (1, []), (2, []), (np.inf, []))
    N = 0
    for x, y in tqdm(loader):
        x, y = x.cuda(), y.cuda().squeeze()
        x_adv = func(net, x, y)
        with torch.no_grad():
            _, p = net(x).max(dim=1)
            _, p_adv = net(x_adv).max(dim=1)
        Y.append(y.cpu().numpy())
        P.append(p.detach().cpu().numpy())
        P_adv.append(p_adv.detach().cpu().numpy())
        if (f := p_adv != y).sum() > 0:
            d = x_adv[f] - x[f]
            for ord_, n in norms:
                n.append(d.norm(p=ord_, dim=(1, 2)).cpu())
        N += 1
        if N > 5:
            break
    Y = np.concatenate(Y)
    P = np.concatenate(P)
    P_adv = np.concatenate(P_adv)
    acc = metrics.accuracy_score(Y, P) * 100
    acc_adv = metrics.accuracy_score(Y, P_adv) * 100
    norms = (torch.cat(n).mean().item() for _, n in norms)
    return (acc, acc_adv, *norms)


def main():
    net = _get_pretrained_model()
    loader = _get_loader()
    samples = _get_postive_samples(
        net,
        loader,
        ("airplane", "chair", "table"),
    )

    # compute & save saliency maps
    _get_attributions(
        net,
        samples,
        erms.compute_saliency_map,
        "./out/saliency",
    )

    # compute & save integrated gradients
    for name in ("zeros", "randn", "sphere"):
        plot_pc(
            _get_baseline(name).cpu().numpy().squeeze(),
            f"./out/baseline-{name}",
        )
        _get_attributions(
            net,
            samples,
            partial(
                erms.compute_integrated_gradients,
                baseline=_get_baseline(name),
            ),
            f"./out/ig-{name}",
        )

    # evaluate and save adversarial attacks
    for name, attack in (
        ("gn", erms.compute_gn_attack),
        ("fgsm", erms.compute_fgsm_attack),
        ("pgd-l2", erms.compute_pgd_l2_attack),
        ("pgd-linf", erms.compute_pgd_linf_attack),
    ):
        _get_attacks(net, samples, attack, f"./out/adv-{name}")
        acc, acc_adv, n_0, n_1, n_2, n_inf = _eval_attacks(net, loader, attack)
        print(
            f"{name} | acc: {acc:.3f}, acc_adv: {acc_adv:.3f}, "
            f"L_0: {n_0:.3f}, L_1: {n_1:.3f}, "
            f"L_2: {n_2:.3f}, L_inf: {n_inf:.3f}"
        )


if __name__ == "__main__":
    main()
