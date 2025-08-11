import os
from collections import defaultdict
from dataclasses import asdict
from fnmatch import fnmatchcase
from glob import glob
from typing import Sized

import torch
import torch.distributed as dist
from datasets import Dataset as HfDataset
from natsort import natsorted
from safetensors.torch import load_model
from schedulefree import ScheduleFreeWrapper
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, get_linear_schedule_with_warmup

from .config import TrainConfig
from .data import MemmapDataset
from .muon import Muon
from .sign_sgd import SignSGD
from .sparse_coder import SparseCoder
from .utils import get_layer_list, resolve_widths, set_submodule


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        dataset: HfDataset | MemmapDataset,
        model: PreTrainedModel,
    ):
        # Store the whole model, including any potential causal LM wrapper
        self.model = model

        if cfg.hookpoints:
            assert not cfg.layers, "Cannot specify both `hookpoints` and `layers`."

            # Replace wildcard patterns
            raw_hookpoints = []
            for name, _ in model.base_model.named_modules():
                if any(fnmatchcase(name, pat) for pat in cfg.hookpoints):
                    raw_hookpoints.append(name)

            # Natural sort to impose a consistent order
            cfg.hookpoints = natsorted(raw_hookpoints)
        else:
            # If no layers are specified, train on all of them
            if not cfg.layers:
                N = model.config.num_hidden_layers
                cfg.layers = list(range(0, N))

            # Now convert layers to hookpoints
            layers_name, _ = get_layer_list(model)
            cfg.hookpoints = [f"{layers_name}.{i}" for i in cfg.layers]

        cfg.hookpoints = cfg.hookpoints[:: cfg.layer_stride]

        self.cfg = cfg
        self.dataset = dataset
        self.distribute_modules()

        device = model.device
        input_widths = resolve_widths(model, cfg.hookpoints)
        unique_widths = set(input_widths.values())

        if cfg.distribute_modules and len(unique_widths) > 1:
            # dist.all_to_all requires tensors to have the same shape across ranks
            raise ValueError(
                f"All modules must output tensors of the same shape when using "
                f"`distribute_modules=True`, got {unique_widths}"
            )

        # Initialize all the SAEs
        print(f"Initializing SAEs with random seed(s) {cfg.init_seeds}")
        self.saes = {}
        for hook in self.local_hookpoints():
            for seed in cfg.init_seeds:
                torch.manual_seed(seed)

                # Add suffix to the name to disambiguate multiple seeds
                name = f"{hook}/seed{seed}" if len(cfg.init_seeds) > 1 else hook
                self.saes[name] = SparseCoder(
                    input_widths[hook], cfg.sae, device, dtype=torch.float32
                )

        assert isinstance(dataset, Sized)
        num_batches = len(dataset) // cfg.batch_size

        match cfg.optimizer:
            case "adam":
                try:
                    from bitsandbytes.optim import Adam8bit as Adam

                    print("Using 8-bit Adam from bitsandbytes")
                except ImportError:
                    from torch.optim import Adam

                    print(
                        "bitsandbytes 8-bit Adam not available, using torch.optim.Adam"
                    )
                    print("Run `pip install bitsandbytes` for less memory usage.")

                pgs = [
                    dict(
                        params=sae.parameters(),
                        lr=cfg.lr or 2e-4 / (sae.num_latents / (2**14)) ** 0.5,
                    )
                    for sae in self.saes.values()
                ]
                # For logging purposes
                lrs = [f"{lr:.2e}" for lr in sorted(set(pg["lr"] for pg in pgs))]

                adam = Adam(pgs)
                self.optimizers = [adam]
                self.lr_schedulers = [
                    get_linear_schedule_with_warmup(
                        adam, cfg.lr_warmup_steps, num_batches
                    )
                ]
            case "muon":
                params = {p for sae in self.saes.values() for p in sae.parameters()}
                muon_params = {p for p in params if p.ndim >= 2}
                lrs = [f"{cfg.lr or 2e-3:.2e}"]

                self.optimizers = [
                    Muon(
                        muon_params,
                        # Muon LR is independent of the number of latents
                        lr=cfg.lr or 2e-3,
                        # Muon distributes the work of the Newton-Schulz iterations
                        # across all ranks for DDP but this doesn't make sense when
                        # we're distributing modules across ranks
                        ddp=not cfg.distribute_modules,
                    ),
                    torch.optim.Adam(params - muon_params, lr=cfg.lr or 2e-3),
                ]
                self.lr_schedulers = [
                    get_linear_schedule_with_warmup(self.optimizers[0], 0, num_batches),
                    get_linear_schedule_with_warmup(
                        self.optimizers[1], cfg.lr_warmup_steps, num_batches
                    ),
                ]
            case "signum":
                from schedulefree import ScheduleFreeWrapper

                pgs = [
                    dict(
                        params=sae.parameters(),
                        lr=cfg.lr or 5e-3 / (sae.num_latents / (2**14)) ** 0.5,
                    )
                    for sae in self.saes.values()
                ]
                lrs = [f"{lr:.2e}" for lr in sorted(set(pg["lr"] for pg in pgs))]

                opt = ScheduleFreeWrapper(SignSGD(pgs), momentum=0.95)
                opt.train()

                self.optimizers = [opt]
                self.lr_schedulers = []
            case other:
                raise ValueError(f"Unknown optimizer '{other}'")

        print(f"Learning rates: {lrs}" if len(lrs) > 1 else f"Learning rate: {lrs[0]}")
        self.global_step = 0
        self.num_tokens_since_fired = {
            name: torch.zeros(sae.num_latents, device=device, dtype=torch.long)
            for name, sae in self.saes.items()
        }
        self.exclude_tokens = torch.tensor(
            self.cfg.exclude_tokens, device=device, dtype=torch.long
        )

        num_latents = list(self.saes.values())[0].num_latents
        self.initial_k = min(num_latents, round(list(input_widths.values())[0] * 10))
        self.final_k = self.cfg.sae.k

        self.best_loss = (
            {name: float("inf") for name in self.local_hookpoints()}
            if self.cfg.loss_fn == "fvu"
            else float("inf")
        )

    def load_state(self, path: str):
        """Load the trainer state from disk."""
        device = self.model.device

        # Load the train state first so we can print the step number
        train_state = torch.load(
            f"{path}/state.pt", map_location=device, weights_only=True
        )
        self.global_step = train_state["global_step"]

        for file in glob(f"{path}/rank_*_state.pt"):
            rank_state = torch.load(file, map_location=device, weights_only=True)

            for k in self.local_hookpoints():
                if k in rank_state["num_tokens_since_fired"]:
                    self.num_tokens_since_fired[k] = rank_state[
                        "num_tokens_since_fired"
                    ][k]

                if not isinstance(rank_state["best_loss"], dict):
                    self.best_loss = rank_state["best_loss"]
                elif k in rank_state["best_loss"]:
                    self.best_loss[k] = rank_state["best_loss"][k]  # type: ignore

        print(
            f"\033[92mResuming training at step {self.global_step} from '{path}'\033[0m"
        )

        for i, scheduler in enumerate(self.lr_schedulers):
            lr_state = torch.load(
                f"{path}/lr_scheduler_{i}.pt", map_location=device, weights_only=True
            )
            scheduler.load_state_dict(lr_state)

        for i, optimizer in enumerate(self.optimizers):
            opt_state = torch.load(
                f"{path}/optimizer_{i}.pt", map_location=device, weights_only=True
            )
            optimizer.load_state_dict(opt_state)

        for name, sae in self.saes.items():
            load_model(sae, f"{path}/{name}/sae.safetensors", device=str(device))

    def get_current_k(self) -> int:
        """Get the current k value based on a linear decay schedule."""
        if self.global_step >= self.cfg.k_decay_steps:
            return self.final_k

        progress = self.global_step / self.cfg.k_decay_steps
        return round(self.initial_k * (1 - progress) + self.final_k * progress)

    def fit(self):
        # Use Tensor Cores even for fp32 matmuls
        torch.set_float32_matmul_precision("high")

        # Make sure the model is frozen
        self.model.requires_grad_(False)

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        ddp = dist.is_initialized() and not self.cfg.distribute_modules

        wandb = None
        if self.cfg.log_to_wandb and rank_zero:
            try:
                import wandb

                wandb.init(
                    entity=os.environ.get("WANDB_ENTITY", None),
                    name=self.cfg.run_name,
                    project=os.environ.get("WANDB_PROJECT", "sparsify"),
                    config=asdict(self.cfg),
                    save_code=True,
                )
            except (AttributeError, ImportError):
                print("Weights & Biases not available, skipping logging.")
                print("Run `pip install -U wandb` if you want to use it.")
                self.cfg.log_to_wandb = False

        num_sae_params = sum(
            p.numel() for s in self.saes.values() for p in s.parameters()
        )
        num_model_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of SAE parameters: {num_sae_params:_}")
        print(f"Number of model parameters: {num_model_params:_}")

        num_batches = len(self.dataset) // self.cfg.batch_size
        if self.global_step > 0:
            assert hasattr(self.dataset, "select"), "Dataset must implement `select`"

            n = self.global_step * self.cfg.batch_size
            ds = self.dataset.select(range(n, len(self.dataset)))  # type: ignore
        else:
            ds = self.dataset

        device = self.model.device
        dl = DataLoader(
            ds,  # type: ignore
            batch_size=self.cfg.batch_size,
            # NOTE: We do not shuffle here for reproducibility; the dataset should
            # be shuffled before passing it to the trainer.
            shuffle=False,
        )
        pbar = tqdm(
            desc="Training",
            disable=not rank_zero,
            initial=self.global_step,
            total=num_batches,
        )

        did_fire = {
            name: torch.zeros(sae.num_latents, device=device, dtype=torch.bool)
            for name, sae in self.saes.items()
        }

        tokens_mask: torch.Tensor

        acc_steps = self.cfg.grad_acc_steps * self.cfg.micro_acc_steps
        denom = acc_steps * self.cfg.wandb_log_frequency
        num_tokens_in_step = 0

        # For logging purposes
        avg_auxk_loss = defaultdict(float)
        avg_fvu = defaultdict(float)
        avg_multi_topk_fvu = defaultdict(float)
        avg_ce = 0.0
        avg_kl = 0.0
        avg_losses = (
            {name: float("inf") for name in self.local_hookpoints()}
            if self.cfg.loss_fn == "fvu"
            else float("inf")
        )

        if self.cfg.loss_fn == "ce":
            batch = next(iter(dl))
            x = batch["input_ids"].to(device)

            clean_loss = self.model(x, labels=x).loss
            self.maybe_all_reduce(clean_loss)
            if rank_zero:
                print(f"Initial CE loss: {clean_loss.item():.4f}")

            # If doing end-to-end transcoders, then we don't actually want to run the
            # modules that we're replacing
            if self.cfg.sae.transcode:
                for point in self.cfg.hookpoints:
                    set_submodule(self.model.base_model, point, nn.Identity())

        name_to_module = {
            name: self.model.base_model.get_submodule(name)
            for name in self.cfg.hookpoints
        }
        maybe_wrapped: dict[str, DDP] | dict[str, SparseCoder] = {}
        module_to_name = {v: k for k, v in name_to_module.items()}

        def hook(module: nn.Module, inputs, outputs):
            aux_out = None

            # Maybe unpack tuple inputs and outputs
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            if isinstance(outputs, tuple):
                outputs, *aux_out = outputs
            mask = tokens_mask

            # Name may optionally contain a suffix of the form /seedN where N is an
            # integer. We only care about the part before the slash.
            name, _, _ = module_to_name[module].partition("/")

            # Remember the original output shape since we'll need it for e2e training
            out_shape = outputs.shape

            # Scatter and gather the hidden states across ranks if necessary
            if self.cfg.distribute_modules:
                world_outputs = outputs.new_empty(
                    outputs.shape[0] * dist.get_world_size(), *outputs.shape[1:]
                )
                dist.all_gather_into_tensor(world_outputs, outputs)
                outputs = world_outputs

                # Don't bother with the communication overhead if we're autoencoding
                if self.cfg.sae.transcode:
                    world_inputs = inputs.new_empty(
                        inputs.shape[0] * dist.get_world_size(), *inputs.shape[1:]
                    )
                    dist.all_gather_into_tensor(world_inputs, inputs)
                    inputs = world_inputs

                world_mask = mask.new_empty(
                    mask.shape[0] * dist.get_world_size(), *mask.shape[1:]
                )
                dist.all_gather_into_tensor(world_mask, mask)
                mask = world_mask.bool()

                if name not in self.module_plan[dist.get_rank()]:
                    return

            # Flatten the batch and sequence dimensions
            outputs = outputs.flatten(0, 1)
            inputs = inputs.flatten(0, 1) if self.cfg.sae.transcode else outputs
            mask = mask.flatten(0, 1)

            # Remove tokens not used for training
            all_outputs = outputs.detach().clone()
            outputs = outputs[mask]
            inputs = inputs[mask]

            # On the first iteration, initialize the encoder and decoder biases
            raw = self.saes[name]
            if self.global_step == 0 and not self.cfg.finetune:
                # Ensure the preactivations are centered at initialization
                # This is mathematically equivalent to Anthropic's proposal of
                # subtracting the decoder bias
                if self.cfg.sae.transcode:
                    mean = self.maybe_all_reduce(inputs.mean(0)).to(raw.dtype)
                    mean_image = -mean @ raw.encoder.weight.data.T
                    raw.encoder.bias.data = mean_image

                mean = self.maybe_all_reduce(outputs.mean(0))
                raw.b_dec.data = mean.to(raw.dtype)

            # Make sure the W_dec is still unit-norm if we're autoencoding
            if raw.cfg.normalize_decoder and not self.cfg.sae.transcode:
                raw.set_decoder_norm_to_unit_norm()

            wrapped = maybe_wrapped[name]
            out = wrapped(
                x=inputs,
                y=outputs,
                dead_mask=(
                    self.num_tokens_since_fired[name] > self.cfg.dead_feature_threshold
                    if self.cfg.auxk_alpha > 0
                    else None
                ),
            )

            # Update the did_fire mask
            did_fire[name][out.latent_indices.flatten()] = True
            self.maybe_all_reduce(did_fire[name], "max")  # max is boolean "any"

            if self.cfg.loss_fn in ("ce", "kl"):
                # Replace the normal output with the SAE output
                output = all_outputs.clone()
                output[mask] = out.sae_out.type_as(output)
                output = output.reshape(out_shape)
                return (output, *aux_out) if aux_out is not None else output

            # Metrics that only make sense for local
            avg_fvu[name] += float(self.maybe_all_reduce(out.fvu.detach()) / denom)
            if self.cfg.auxk_alpha > 0:
                avg_auxk_loss[name] += float(
                    self.maybe_all_reduce(out.auxk_loss.detach()) / denom
                )
            if self.cfg.sae.multi_topk:
                avg_multi_topk_fvu[name] += float(
                    self.maybe_all_reduce(out.multi_topk_fvu.detach()) / denom
                )

            # Do a "local" backward pass if we're not training end-to-end
            loss = (
                out.fvu + self.cfg.auxk_alpha * out.auxk_loss + out.multi_topk_fvu / 8
            )
            loss.div(acc_steps).backward()

        k = self.get_current_k()
        for name, sae in self.saes.items():
            sae.cfg.k = k

        for batch in dl:
            x = batch["input_ids"].to(device)
            tokens_mask = torch.isin(x, self.exclude_tokens, invert=True)

            if not maybe_wrapped:
                # Wrap the SAEs with Distributed Data Parallel. We have to do this
                # after we set the decoder bias, otherwise DDP will not register
                # gradients flowing to the bias after the first step.
                maybe_wrapped = (
                    {
                        name: DDP(sae, device_ids=[dist.get_rank()])
                        for name, sae in self.saes.items()
                    }
                    if ddp
                    else self.saes
                )

            # Bookkeeping for dead feature detection
            N = tokens_mask.sum().item()
            num_tokens_in_step += N

            # Compute clean logits if using KL loss
            clean_probs = (
                self.model(x).logits.softmax(dim=-1)
                if self.cfg.loss_fn == "kl"
                else None
            )

            # Forward pass on the model to get the next batch of activations
            handles = [
                mod.register_forward_hook(hook) for mod in name_to_module.values()
            ]
            try:
                match self.cfg.loss_fn:
                    case "ce":
                        ce = self.model(x, labels=x).loss
                        ce.div(acc_steps).backward()

                        avg_ce += float(self.maybe_all_reduce(ce.detach()) / denom)

                        avg_losses = avg_ce
                    case "kl":
                        dirty_lps = self.model(x).logits.log_softmax(dim=-1)
                        kl = -torch.sum(clean_probs * dirty_lps, dim=-1).mean()
                        kl.div(acc_steps).backward()

                        avg_kl += float(self.maybe_all_reduce(kl) / denom)
                        avg_losses = avg_kl
                    case "fvu":
                        self.model(x)
                        avg_losses = dict(avg_fvu)
                    case other:
                        raise ValueError(f"Unknown loss function '{other}'")
            finally:
                for handle in handles:
                    handle.remove()

            # Check if we need to actually do a training step
            step, substep = divmod(self.global_step + 1, self.cfg.grad_acc_steps)
            if substep == 0:
                if self.cfg.sae.normalize_decoder and not self.cfg.sae.transcode:
                    for sae in self.saes.values():
                        sae.remove_gradient_parallel_to_decoder_directions()

                for optimizer in self.optimizers:
                    optimizer.step()
                    optimizer.zero_grad()

                for scheduler in self.lr_schedulers:
                    scheduler.step()

                k = self.get_current_k()
                for name, sae in self.saes.items():
                    sae.cfg.k = k

                ###############
                with torch.no_grad():
                    # Update the dead feature mask
                    for name, counts in self.num_tokens_since_fired.items():
                        counts += num_tokens_in_step
                        counts[did_fire[name]] = 0

                    # Reset stats for this step
                    num_tokens_in_step = 0
                    for mask in did_fire.values():
                        mask.zero_()

                if (step + 1) % self.cfg.save_every == 0:
                    self.save()

                    if self.cfg.save_best:
                        self.save_best(avg_losses)

                if (
                    self.cfg.log_to_wandb
                    and (step + 1) % self.cfg.wandb_log_frequency == 0
                ):
                    info = {}
                    if self.cfg.loss_fn == "ce":
                        info["ce_loss"] = avg_ce
                    elif self.cfg.loss_fn == "kl":
                        info["kl_loss"] = avg_kl

                    for name in self.saes:
                        mask = (
                            self.num_tokens_since_fired[name]
                            > self.cfg.dead_feature_threshold
                        )

                        ratio = mask.mean(dtype=torch.float32).item()
                        info.update({f"dead_pct/{name}": ratio})
                        if self.cfg.loss_fn == "fvu":
                            info[f"fvu/{name}"] = avg_fvu[name]

                        if self.cfg.auxk_alpha > 0:
                            info[f"auxk/{name}"] = avg_auxk_loss[name]
                        if self.cfg.sae.multi_topk:
                            info[f"multi_topk_fvu/{name}"] = avg_multi_topk_fvu[name]

                    if self.cfg.distribute_modules:
                        outputs = [{} for _ in range(dist.get_world_size())]
                        dist.gather_object(info, outputs if rank_zero else None)
                        info.update({k: v for out in outputs for k, v in out.items()})

                    if rank_zero:
                        info["k"] = k

                        if wandb is not None:
                            wandb.log(info, step=step)

                avg_auxk_loss.clear()
                avg_fvu.clear()
                avg_multi_topk_fvu.clear()
                avg_ce = 0.0
                avg_kl = 0.0

            self.global_step += 1
            pbar.update()

        self.save()
        if self.cfg.save_best:
            self.save_best(avg_losses)

        pbar.close()

    def local_hookpoints(self) -> list[str]:
        return (
            self.module_plan[dist.get_rank()]
            if self.module_plan
            else self.cfg.hookpoints
        )

    def maybe_all_cat(self, x: Tensor) -> Tensor:
        """Concatenate a tensor across all processes."""
        if not dist.is_initialized() or self.cfg.distribute_modules:
            return x

        buffer = x.new_empty([dist.get_world_size() * x.shape[0], *x.shape[1:]])
        dist.all_gather_into_tensor(buffer, x)
        return buffer

    def maybe_all_reduce(self, x: Tensor, op: str = "mean") -> Tensor:
        if not dist.is_initialized() or self.cfg.distribute_modules:
            return x

        if op == "sum":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        elif op == "mean":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            x /= dist.get_world_size()
        elif op == "max":
            dist.all_reduce(x, op=dist.ReduceOp.MAX)
        else:
            raise ValueError(f"Unknown reduction op '{op}'")

        return x

    def distribute_modules(self):
        """Prepare a plan for distributing modules across ranks."""
        if not self.cfg.distribute_modules:
            self.module_plan = []
            print(f"Training on modules: {self.cfg.hookpoints}")
            return

        layers_per_rank, rem = divmod(len(self.cfg.hookpoints), dist.get_world_size())
        assert rem == 0, "Number of modules must be divisible by world size"

        # Each rank gets a subset of the layers
        self.module_plan = [
            self.cfg.hookpoints[start : start + layers_per_rank]
            for start in range(0, len(self.cfg.hookpoints), layers_per_rank)
        ]
        for rank, modules in enumerate(self.module_plan):
            print(f"Rank {rank} modules: {modules}")

    def _checkpoint(self, saes: dict[str, SparseCoder], path: str, rank_zero: bool):
        """Save SAEs and training state to disk."""
        print("Saving checkpoint")

        for optimizer in self.optimizers:
            if isinstance(optimizer, ScheduleFreeWrapper):
                optimizer.eval()

        for name, sae in saes.items():
            assert isinstance(sae, SparseCoder)

            sae.save_to_disk(f"{path}/{name}")

        if rank_zero:
            for i, scheduler in enumerate(self.lr_schedulers):
                torch.save(scheduler.state_dict(), f"{path}/lr_scheduler_{i}.pt")

            for i, optimizer in enumerate(self.optimizers):
                torch.save(optimizer.state_dict(), f"{path}/optimizer_{i}.pt")

            torch.save(
                {"global_step": self.global_step},
                f"{path}/state.pt",
            )

            self.cfg.save_json(f"{path}/config.json")

        for optimizer in self.optimizers:
            if isinstance(optimizer, ScheduleFreeWrapper):
                optimizer.train()

        rank = 0 if rank_zero else dist.get_rank()
        torch.save(
            {
                "num_tokens_since_fired": self.num_tokens_since_fired,
                "best_loss": self.best_loss,
            },
            f"{path}/rank_{rank}_state.pt",
        )

    def save(self):
        """Save the SAEs and training state to disk."""
        path = f'{self.cfg.save_dir}/{self.cfg.run_name or "unnamed"}'

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0

        if rank_zero or self.cfg.distribute_modules:
            self._checkpoint(self.saes, path, rank_zero)

        # Barrier to ensure all ranks have saved before continuing
        if dist.is_initialized():
            dist.barrier()

    def save_best(self, avg_loss: float | dict[str, float]):
        """Save individual sparse coders to disk if they have the lowest loss."""
        base_path = f'{self.cfg.save_dir}/{self.cfg.run_name or "unnamed"}/best'
        rank_zero = not dist.is_initialized() or dist.get_rank() == 0

        if isinstance(avg_loss, dict):
            for name in self.saes:
                if avg_loss[name] < self.best_loss[name]:  # type: ignore
                    self.best_loss[name] = avg_loss[name]  # type: ignore

                    if rank_zero or self.cfg.distribute_modules:
                        self._checkpoint(
                            {name: self.saes[name]}, f"{base_path}/{name}", rank_zero
                        )
        else:
            if avg_loss < self.best_loss:  # type: ignore
                self.best_loss = avg_loss  # type: ignore

                if rank_zero or self.cfg.distribute_modules:
                    self._checkpoint(self.saes, base_path, rank_zero)

        # Barrier to ensure all ranks have saved before continuing
        if dist.is_initialized():
            dist.barrier()


# Support old name for compatibility
SaeTrainer = Trainer
