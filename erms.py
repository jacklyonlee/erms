import logging
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from trulens.nn.attribution import (
    AttributionMethod,
    InputAttribution,
    IntegratedGradients,
)
from trulens.nn.models import get_model_wrapper
from trulens.utils.tru_logger import get_logger

get_logger().setLevel(logging.ERROR)  # disable trulens logging


def _compute_attribution(
    net: nn.Module,
    x: torch.Tensor,
    attribution: AttributionMethod,
    *args,
    **kwargs,
) -> torch.Tensor:
    return attribution(
        get_model_wrapper(
            net,
            input_shape=x.shape[1:],
        ),
        *args,
        **kwargs,
    ).attributions(x)


def compute_class_saliency(
    net: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    return _compute_attribution(net, x, InputAttribution)


def compute_integrated_gradients(
    net: nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
) -> torch.Tensor:
    return _compute_attribution(
        net,
        x,
        IntegratedGradients,
        baseline,
        resolution=30,
    )


def compute_class_model(
    net: nn.Module,
    y_idx: int,
    baseline: torch.Tensor,
    n_steps: int = 100,
) -> Sequence[np.ndarray]:
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


def compute_gn_attack(
    net: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    std: float = 1e-2,
) -> torch.Tensor:
    x = x.clone().detach()
    x_adv = x + std * torch.randn_like(x)
    return x_adv.detach()


def _compute_fgsm_attack(
    net: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    loss = F.cross_entropy(net(x), y)
    grad = torch.autograd.grad(
        loss,
        x,
        retain_graph=False,
        create_graph=False,
    )[0]
    x = x + alpha * grad.sign()
    return x.detach()


def compute_fgsm_attack(
    net: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1e-2,
) -> torch.Tensor:
    x = x.clone().detach()
    y = y.clone().detach()
    x.requires_grad = True
    return _compute_fgsm_attack(net, x, y, alpha)


def _compute_pgd_attack(
    net: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    n_steps: int,
    clip: Callable,
) -> torch.Tensor:
    x = x.clone().detach()
    y = y.clone().detach()
    x_adv = x.detach()
    for _ in range(n_steps):
        x_adv.requires_grad = True
        x_adv = _compute_fgsm_attack(net, x_adv, y, alpha)
        x_adv = (x + clip(x_adv - x)).detach()
    return x_adv


def compute_pgd_l1_attack(
    net: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.2,
    alpha: float = 1e-2,
    n_steps: int = 10,
):
    def clip(d: torch.Tensor) -> torch.Tensor:
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
    net: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.4,
    alpha: float = 1e-2,
    n_steps: int = 10,
) -> torch.Tensor:
    def clip(d: torch.Tensor) -> torch.Tensor:
        n = torch.norm(d, dim=(1, 2), keepdim=True)
        f = torch.min(eps / n, torch.ones_like(n))
        return d * f

    return _compute_pgd_attack(net, x, y, alpha, n_steps, clip)


def compute_pgd_linf_attack(
    net: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-2,
    alpha: float = 1e-2,
    n_steps: int = 10,
) -> torch.Tensor:
    def clip(d: torch.Tensor) -> torch.Tensor:
        return torch.clamp(d, min=-eps, max=eps)

    return _compute_pgd_attack(net, x, y, alpha, n_steps, clip)
