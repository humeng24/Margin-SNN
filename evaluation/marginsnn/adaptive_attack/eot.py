import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F


def grad_find(model, data, target, sample_size=30):
    data.requires_grad = True
    ensemble_xs = data.expand([sample_size, 3, 32, 32])
    labels = target.expand([sample_size])
    logits = model(ensemble_xs)

    _, preds = torch.max(logits.data, 1)

    loss = F.cross_entropy(logits, labels)
    loss.backward()

    gradient = data.grad
    data.requires_grad = False

    return preds, gradient


def EOT_FGSM(model, data, target, eps=8/255, data_min=0, data_max=1, sample_size=30):
    # model.eval()
    x_adv = data.clone()

    preds, gradient = grad_find(model, x_adv, target, sample_size)

    with torch.no_grad():
        x_adv += eps * torch.sign(gradient)
        x_adv.clamp_(data_min, data_max)

    return x_adv


def EOT_PGD(model, data, target, eps=8/255, steps=10, alpha=1/255, d_min=0, d_max=1,
            sample_size=30, random_start=True):
    # model.eval()
    x_adv = data.clone()

    data_max = data + eps
    data_min = data - eps
    data_max.clamp_(d_min, d_max)
    data_min.clamp_(d_min, d_max)

    if random_start:
        with torch.no_grad():
            x_adv.data = data + x_adv.uniform_(-1 * eps, eps)
            x_adv.data.clamp_(d_min, d_max)

    for _ in range(steps):
        preds, gradient = grad_find(model, x_adv, target, sample_size)

        with torch.no_grad():
            x_adv += alpha * torch.sign(gradient)
            x_adv.data = torch.max(torch.min(x_adv, data_max),
                                   data_min)

    return x_adv