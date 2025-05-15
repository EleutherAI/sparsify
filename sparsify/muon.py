"""
Adapted from https://github.com/KellerJordan/Muon/blob/master/muon.py
Modifications by Nora Belrose
"""

import torch
import torch.distributed as dist
from torch import Tensor


def quintic_newtonschulz(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero.
    For the purpose of minimizing steps, it turns out to be empirically effective to
    keep increasing the slope at zero even beyond the point where the iteration no
    longer converges all the way to one everywhere on the interval. This iteration
    therefore does not produce UV^T but rather something like US'V^T where S' is
    diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    # batched implementation by @scottjmaddox, put into practice by @YouJiacheng
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        # quintic strategy adapted from suggestion by @jxbz, @leloykun, @YouJiacheng
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon is a generalized steepest descent optimizer using the spectral norm on the
    matrix-valued parameters. This means it always updates in the direction which
    locally reduces the loss as much as possible, while constraining the update to have
    a spectral norm given by the learning rate. It achieves this using a Newton-Schulz
    iteration to orthogonalize the stochastic gradient (or momentum buffer) for each
    matrix in the model before taking a step.

    The spectral norm is an intuitive heuristic because, roughly speaking, it measures
    the maximum change to the activations of a layer that can be caused by a change to
    its weights. By constraining the worst-case change to the activations, we ensure
    that we do not desta

    TThis optimizer is unlikely to work well with small batch sizes, since it strongly
    magnifies small singular values, which will be noisy given a small minibatch.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD.
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.1,
        ns_steps: int = 5,
        ddp: bool = True,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        self.rank = dist.get_rank() if dist.is_initialized() and ddp else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() and ddp else 1

        # Distributed Data Parallel (DDP) setup
        if dist.is_initialized() and ddp:
            param_groups = []

            # Check that the user isn't doing some weird model parallelism
            devices = {p.device for p in params}
            device = next(iter(devices))
            assert len(devices) == 1, "Muon does not support model parallelism."

            # Group parameters by their device and number of elements. For each group,
            # we pre-allocate a buffer to store the updates from all ranks.
            for size in {p.numel() for p in params}:
                b = torch.empty(
                    self.world_size, size, dtype=torch.bfloat16, device=device
                )
                group = dict(
                    params=[p for p in params if p.numel() == size],
                    update_buffer=b,
                    update_buffer_views=[b[i] for i in range(self.world_size)],
                )
                param_groups.append(group)

            super().__init__(param_groups, defaults)
        else:
            super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params: list[Tensor] = group["params"]

            # Apply decoupled weight decay to all parameters. This doesn't require any
            # communication, since it's a simple element-wise operation.
            if group["weight_decay"] > 0.0:
                for p in params:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

            # These will be None / empty list if we're not using DDP
            update_buffer: Tensor | None = group.get("update_buffer", None)
            update_buffer_views: list[Tensor] = group.get("update_buffer_views", [])

            beta = group["momentum"]
            handle = None
            params_world = None

            def update_prev():  # optimized implementation contributed by @YouJiacheng
                assert handle is not None and params_world is not None
                handle.wait()

                for p_world, g_world in zip(params_world, update_buffer_views):
                    # Heuristic from <https://arxiv.org/abs/2502.16982>
                    scale = 0.2 * max(p_world.shape) ** 0.5
                    p_world.add_(g_world.view_as(p_world), alpha=-group["lr"] * scale)

            for i in range(0, len(params), self.world_size):
                # Compute Muon update
                if i + self.rank < len(params):
                    p = params[i + self.rank]
                    state = self.state[p]

                    g = p.grad
                    assert g is not None

                    # Apply momentum
                    if beta > 0.0:
                        if "exp_avg" not in state:
                            state["exp_avg"] = torch.zeros_like(g)

                        buf: Tensor = state["exp_avg"].lerp_(g, 1 - beta)
                        g = g.lerp_(buf, beta) if group["nesterov"] else buf

                    if g.ndim == 4:  # for the case of conv filters
                        g = g.view(len(g), -1)

                    g = quintic_newtonschulz(g, steps=group["ns_steps"])
                else:
                    g = update_buffer_views[self.rank]

                if self.world_size > 1:
                    # async all_gather instead of sync all_reduce by @YouJiacheng
                    if i > 0:
                        update_prev()

                    handle = dist.all_gather_into_tensor(
                        update_buffer, g.flatten(), async_op=True
                    )
                    params_world = params[i : i + self.world_size]
                else:
                    scale = 0.2 * max(params[i].shape) ** 0.5
                    params[i].add_(g, alpha=-group["lr"] * scale)

            if self.world_size > 1:
                update_prev()
