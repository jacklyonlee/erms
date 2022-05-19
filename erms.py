import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm
from trulens.nn.attribution import InputAttribution, IntegratedGradients
from trulens.nn.models import get_model_wrapper
from trulens.utils.tru_logger import get_logger

get_logger().setLevel(logging.ERROR)  # disable trulens logging


def compute_saliency_map(net, x):
    return InputAttribution(
        get_model_wrapper(
            net,
            input_shape=x.shape[1:],
        )
    ).attributions(x)


def compute_integrated_gradients(net, x, baseline):
    return IntegratedGradients(
        get_model_wrapper(
            net,
            input_shape=x.shape[1:],
        ),
        baseline,
        resolution=30,
    ).attributions(x)


def compute_class_saliency_map(net, y_idx, baseline, n_steps=100):
    x, xs = baseline.clone().detach(), []
    x.requires_grad = True
    opt = torch.optim.Adam([x], lr=5e-3, weight_decay=1e-5)
    with tqdm(range(n_steps)) as t:
        for _ in t:
            loss = -torch.mean(net(x)[:, y_idx])
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


def _compute_fgsm(net, x, y, alpha):
    loss = F.cross_entropy(net(x), y)
    grad = torch.autograd.grad(
        loss,
        x,
        retain_graph=False,
        create_graph=False,
    )[0]
    x = x + alpha * grad.sign()
    return x.detach()


def compute_fgsm_attack(net, x, y, eps=1e-2):
    x = x.clone().detach()
    y = y.clone().detach()
    x.requires_grad = True
    return _compute_fgsm(net, x, y, eps)


def _compute_pgd_attack(net, x, y, alpha, n_steps, clip):
    x = x.clone().detach()
    y = y.clone().detach()
    x_adv = x.detach()
    for _ in range(n_steps):
        x_adv.requires_grad = True
        x_adv = _compute_fgsm(net, x_adv, y, alpha)
        x_adv = (x + clip(x_adv - x)).detach()
    return x_adv


def compute_pgd_linf_attack(
    net,
    x,
    y,
    eps=1e-2,
    alpha=1e-2,
    n_steps=10,
):
    return _compute_pgd_attack(
        net,
        x,
        y,
        alpha,
        n_steps,
        lambda d: torch.clamp(
            d,
            min=-eps,
            max=eps,
        ),
    )


def compute_pgd_l2_attack(
    net,
    x,
    y,
    eps=0.5,
    alpha=0.2,
    n_steps=10,
):
    def clip(d):
        n = torch.norm(d, dim=(1, 2), keepdim=True)
        f = torch.min(eps / n, torch.ones_like(n))
        return d * f

    return _compute_pgd_attack(net, x, y, alpha, n_steps, clip)
