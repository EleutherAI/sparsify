from dataclasses import dataclass, asdict
from sparsify.xformers_decoder import xformers_embedding_bag
from safetensors.torch import load_model, save_model
from math import ceil, log2
from pathlib import Path
from torch import nn
import torch
import json


def decoder_impl(indices, weights, dictionary):
    return xformers_embedding_bag(
        indices, dictionary, weights, mode="sum"
    )


def grad_pursuit_update_step(signal, weights, dictionary, eps=1e-3):
    residual = signal - torch.einsum('fv,bf->bv', dictionary, weights)
    selected_features = (weights != 0)
    inner_products = torch.einsum('fv,bv->bf', dictionary, residual)
    idx = inner_products.argmax(dim=1)
    selected_features = torch.scatter(selected_features, 1, idx[:, None], torch.full((idx.shape[0], 1), True, device=idx.device))

    grad = selected_features * inner_products
    c = torch.einsum('bf,fv->bv', grad, dictionary)
    c_square_norm = torch.einsum('bv,bv->b', c, c)
    step_size = torch.einsum('bv,bv->b', c, residual) / torch.maximum(c_square_norm, torch.full_like(c_square_norm, eps))
    weights = weights + step_size[:, None] * grad
    weights = torch.nn.functional.relu(weights)
    return weights

def grad_pursuit(signal, dictionary, target_l0):
    if target_l0 < 0 or target_l0 > dictionary.shape[0]:
        raise ValueError(f"target_l0 must be in [0, {dictionary.shape[0]}], got {target_l0}")
    weights = torch.zeros((signal.shape[0], dictionary.shape[0]), dtype=signal.dtype, device=signal.device)
    for i in range(target_l0):
        weights = grad_pursuit_update_step(signal, weights, dictionary)
    return weights.topk(target_l0, dim=-1)


@dataclass
class ITDAConfig:
    d_model: int
    target_l0: int
    loss_threshold: float
    add_error: bool = False
    subtract_mean: bool = False
    fvu_loss: bool = True
    start_size: int = 0
    error_k: int = None
    skip_connection: bool = False
    preprocessing_steps: int = 1



def dictionary_size_transform(x):
    return 2 ** ceil(max(4, log2(x)))


@dataclass
class ITDAOutput:
    weights: torch.Tensor
    indices: torch.Tensor
    x_reconstructed: torch.Tensor
    y_reconstructed: torch.Tensor
    losses: torch.Tensor
    skip_y: torch.Tensor | None = None


class ITDA(nn.Module):
    def __init__(self, config: ITDAConfig,
                 dtype=torch.float32, device=None,
                 initial_size=None):
        super().__init__()
        if initial_size is None:
            initial_size = config.start_size
        self.xs = nn.Parameter(
            torch.empty((initial_size, config.d_model),
                        dtype=dtype, device=device),
            requires_grad=False
        )
        self.ys = nn.Parameter(
            torch.empty((initial_size, config.d_model),
                        dtype=dtype, device=device),
            requires_grad=False
        )
        self.device = device
        self.mean_x, self.mean_y = (
            nn.Parameter(torch.zeros(config.d_model, dtype=dtype, device=device),
                         requires_grad=False) for _ in range(2)
        )
        self.weight = nn.Parameter(
            torch.empty((config.d_model, config.d_model),
                        dtype=dtype, device=device),
            requires_grad=False)
        self.weight_data = []
        self.steps = 0
        self.dictionary_size = 0
        self.config = config
        self.dtype = dtype
    
    def save_to_disk(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        save_model(self, str(path / "sae.safetensors"))
        with open(path / "cfg.json", "w") as f:
            json.dump(
                {
                    **asdict(self.config),
                    "dictionary_size": self.dictionary_size,
                    "steps": self.steps,
                },
                f,
            )
        return path
    
    @staticmethod
    def load_from_disk(path, device: str | torch.device | None = "cpu"):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "cfg.json") as f:
            config = json.load(f)
            itda_config = config | {}
            itda_config.pop("dictionary_size")
            itda_config.pop("steps")
            cfg = ITDAConfig(**itda_config)
        
        itda = ITDA(cfg, device=device,
                    initial_size=dictionary_size_transform(config["dictionary_size"]))
        load_model(itda, str(path / "sae.safetensors"),
                   strict=False, device=device)
        itda.dictionary_size = config["dictionary_size"]
        itda.steps = config["steps"]
        return itda
    
    def forward(self, x, y, target_l0=None):
        assert x.ndim == y.ndim == 2
        x, y = x.to(self.dtype), y.to(self.dtype)
        assert self.steps >= self.config.preprocessing_steps
        if self.config.skip_connection:
            skip_y = (x - self.mean_x) @ self.weight
        if self.dictionary_size == 0:
            if self.config.subtract_mean:
                x_reconstructed = self.mean_x.broadcast_to(*x.shape[:-1], -1)
                y_reconstructed = self.mean_y.broadcast_to(*y.shape[:-1], -1)
            else:
                x_reconstructed = y_reconstructed = \
                    torch.zeros_like(y)
            weights = torch.zeros((y.shape[0], self.config.target_l0), dtype=y.dtype, device=y.device)
            indices = torch.zeros((y.shape[0], self.config.target_l0), dtype=torch.long, device=y.device)
        else:
            if target_l0 is None:
                target_l0 = self.config.target_l0
            if self.config.subtract_mean:
                x = x - self.mean_x
            weights, indices = grad_pursuit(x, self.xs[:self.dictionary_size], min(target_l0, self.dictionary_size))
            x_reconstructed = decoder_impl(indices, weights, self.xs[:self.dictionary_size])
            y_reconstructed = decoder_impl(indices, weights, self.ys[:self.dictionary_size])
            if self.config.subtract_mean:
                x_reconstructed = x_reconstructed + self.mean_x
                y_reconstructed = y_reconstructed + self.mean_y
        if self.config.skip_connection:
            y_reconstructed = y_reconstructed + skip_y
        if self.config.fvu_loss:
            l2_loss = (y - y_reconstructed).pow(2).sum(-1)
            total_variance = (y - self.mean_y).pow(2).sum(-1).mean()
            losses = l2_loss / total_variance
        else:
            losses = (y_reconstructed - y).pow(2).sum(-1)
        return ITDAOutput(
            weights=weights,
            indices=indices,
            x_reconstructed=x_reconstructed,
            y_reconstructed=y_reconstructed,
            losses=losses,
            skip_y=skip_y if self.config.skip_connection else None
        )
       
    @torch.no_grad()
    def step(self, x, y):
        assert x.ndim == y.ndim == 2
        x, y = x.to(self.dtype), y.to(self.dtype)
        if self.steps < self.config.preprocessing_steps:
            if self.steps == 0 or self.mean_x is None or self.mean_y is None:
                self.mean_x = nn.Parameter(torch.zeros_like(x[0]), requires_grad=False)
                self.mean_y = nn.Parameter(torch.zeros_like(y[0]), requires_grad=False)
            self.mean_x.data.mul_(self.steps / (self.steps + 1)).add_(x.mean(0) / (self.steps + 1))
            self.mean_y.data.mul_(self.steps / (self.steps + 1)).add_(y.mean(0) / (self.steps + 1))
            self.steps += 1
            if self.config.skip_connection:
                self.weight_data.append((x, y))
            return None
        if self.steps == self.config.preprocessing_steps and self.config.skip_connection:
            assert self.config.subtract_mean
            all_x = torch.cat([x for x, _ in self.weight_data])
            all_y = torch.cat([y for _, y in self.weight_data])
            x_train = all_x - self.mean_x
            y_train = all_y - self.mean_y
            weight = torch.linalg.lstsq(x_train, y_train).solution
            self.weight.data = weight.clone()
            del self.weight_data
        self.steps += 1
        out_0 = self(x, y)
        should_be_added = out_0.losses > self.config.loss_threshold
        if self.config.add_error:
            if self.config.error_k is None:
                added_x = x - out_0.x_reconstructed
                added_y = y - out_0.y_reconstructed
            else:
                out_1 = self(x, y, target_l0=self.config.error_k)
                added_x = x - out_1.x_reconstructed
                added_y = y - out_1.y_reconstructed
        else:
            added_x = x
            added_y = y
            if self.config.subtract_mean:
                added_x = added_x - self.mean_x
                added_y = added_y - self.mean_y
            if self.config.skip_connection:
                added_y = added_y - out_0.skip_y
        added_x = added_x[should_be_added]
        added_y = added_y[should_be_added]
        added_y = added_y / added_x.norm(dim=-1, keepdim=True)
        added_x = added_x / added_x.norm(dim=-1, keepdim=True)
        n_added = added_x.shape[0]
        if n_added:
            if self.dictionary_size + n_added > self.xs.shape[0]:
                old_xs, old_ys = self.xs, self.ys
                new_size = dictionary_size_transform(self.dictionary_size + n_added)
                self.xs = nn.Parameter(
                    torch.empty((new_size, self.config.d_model), device=self.device, dtype=self.dtype)
                )
                self.ys = nn.Parameter(
                    torch.empty((new_size, self.config.d_model), device=self.device, dtype=self.dtype)
                )
                self.xs[:self.dictionary_size] = old_xs[:self.dictionary_size]
                self.ys[:self.dictionary_size] = old_ys[:self.dictionary_size]
                del old_xs, old_ys
            self.xs[self.dictionary_size:self.dictionary_size + n_added] = added_x
            self.ys[self.dictionary_size:self.dictionary_size + n_added] = added_y
            self.dictionary_size += n_added
        return out_0
    