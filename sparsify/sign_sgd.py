import torch
from torch.optim import Optimizer


class SignSGD(Optimizer):
    """Steepest descent in the L-infty norm. From <https://arxiv.org/abs/1802.04434>"""

    def __init__(self, params, lr: float = 1e-3):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr}
        super(SignSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: None = None) -> None:
        assert closure is None, "Closure is not supported."

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is not None:
                    p.add_(p.grad.sign(), alpha=-lr)
