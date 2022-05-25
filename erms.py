import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm
from trulens.nn.attribution import InputAttribution, IntegratedGradients
from trulens.nn.models import get_model_wrapper
from trulens.utils.tru_logger import get_logger

get_logger().setLevel(logging.ERROR)  # disable trulens logging


def _compute_attribution(net, x, attribution, *args, **kwargs):
    return attribution(
        get_model_wrapper(
            net,
            input_shape=x.shape[1:],
        ),
        *args,
        **kwargs,
    ).attributions(x)


def compute_class_saliency(net, x):
    return _compute_attribution(net, x, InputAttribution)


def compute_integrated_gradients(net, x, baseline):
    return _compute_attribution(
        net,
        x,
        IntegratedGradients,
        baseline,
        resolution=30,
    )


def compute_class_model(net, y_idx, baseline, n_steps=100):
    x, xs = baseline.clone().detach(), []
    x.requires_grad = True
    opt = torch.optim.Adam([x], lr=5e-3, weight_decay=1e-5)
    with tqdm(range(n_steps)) as t:
        for _ in t:
            o = F.softmax(net(x), dim=-1)
            loss = -torch.mean(o[:, y_idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            xs.append(x.clone().detach().cpu().numpy().squeeze())
            t.set_description(f"L:{loss.item():.3f}|N:{x.norm().item():.3f}")
    return xs


def compute_gn_attack(net, x, _, std=1e-2):
    x = x.clone().detach()
    x_adv = x + std * torch.randn_like(x)
    return x_adv.detach()


def _compute_fgsm_attack(net, x, y, alpha):
    loss = F.cross_entropy(net(x), y)
    grad = torch.autograd.grad(
        loss,
        x,
        retain_graph=False,
        create_graph=False,
    )[0]
    x = x + alpha * grad.sign()
    return x.detach()


def compute_fgsm_attack(net, x, y, alpha=1e-2):
    x = x.clone().detach()
    y = y.clone().detach()
    x.requires_grad = True
    return _compute_fgsm_attack(net, x, y, alpha)


def _compute_pgd_attack(net, x, y, alpha, n_steps, clip):
    x = x.clone().detach()
    y = y.clone().detach()
    x_adv = x.detach()
    for _ in range(n_steps):
        x_adv.requires_grad = True
        x_adv = _compute_fgsm_attack(net, x_adv, y, alpha)
        x_adv = (x + clip(x_adv - x)).detach()
    return x_adv


def compute_pgd_l1_attack(
    net,
    x,
    y,
    eps=0.2,
    alpha=1e-2,
    n_steps=10,
):
    def clip(d):
        B, N, C = d.shape
        d = d.view(N, -1)
        mask = (torch.norm(d, p=1, dim=1) < eps).float().unsqueeze(1)
        mu, _ = torch.sort(torch.abs(d), dim=1, descending=True)
        cumsum = torch.cumsum(mu, dim=1)
        arange = torch.arange(1, d.shape[1] + 1, device=d.device)
        rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
        theta = (cumsum[torch.arange(d.shape[0]), rho.cpu() - 1] - eps) / rho
        proj = (torch.abs(d) - theta.unsqueeze(1)).clamp(min=0)
        d = mask * d + (1 - mask) * proj * torch.sign(d)
        return d.view(B, N, C)

    return _compute_pgd_attack(net, x, y, alpha, n_steps, clip)


def compute_pgd_l2_attack(
    net,
    x,
    y,
    eps=0.4,
    alpha=1e-2,
    n_steps=10,
):
    def clip(d):
        n = torch.norm(d, dim=(1, 2), keepdim=True)
        f = torch.min(eps / n, torch.ones_like(n))
        return d * f

    return _compute_pgd_attack(net, x, y, alpha, n_steps, clip)


def compute_pgd_linf_attack(
    net,
    x,
    y,
    eps=1e-2,
    alpha=1e-2,
    n_steps=10,
):
    def clip(d):
        return torch.clamp(d, min=-eps, max=eps)

    return _compute_pgd_attack(net, x, y, alpha, n_steps, clip)
