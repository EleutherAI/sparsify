from collections import defaultdict
from dataclasses import asdict
from fnmatch import fnmatchcase
from typing import Sized
from glob import glob
import math

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
from transformers import (
    PreTrainedModel, get_linear_schedule_with_warmup
)

from .config import TrainConfig
from .data import MemmapDataset
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
            if cfg.layer_stride > 1:
                cfg.hookpoints = cfg.hookpoints[::cfg.layer_stride]
        else:
            # If no layers are specified, train on all of them
            if not cfg.layers:
                N = model.config.num_hidden_layers
                cfg.layers = list(range(0, N))

            # Now convert layers to hookpoints
            layers_name, _ = get_layer_list(model)
            cfg.hookpoints = [f"{layers_name}.{i}" for i in cfg.layers]
        
        # If doing end-to-end, then we don't actually want to run the modules that
        # we're replacing
        if cfg.end_to_end:
            for hook in cfg.hookpoints:
                set_submodule(model.base_model, hook, nn.Identity())

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
        base_lr = {
            'signum': 3e-3,
            'adam': 2e-4,
        }[cfg.optimizer]

        # Re-initialize the decoder for transcoder training. By default the Sae class
        # initializes the decoder with the transpose of the encoder.
        if cfg.transcode:
            for sae in self.saes.values():
                assert sae.W_dec is not None
                nn.init.kaiming_normal_(sae.W_dec.data)

        pgs = [
            {
                "params": sae.parameters(),
                # Auto-select LR using 1 / sqrt(d) scaling law from Fig 3 of the paper
                "lr": cfg.lr or base_lr / (sae.num_latents / (2**14)) ** 0.5,
            }
            for sae in self.saes.values()
        ]
        # Dedup the learning rates we're using, sort them, round to 2 decimal places
        lrs = [f"{lr:.2e}" for lr in sorted(set(pg["lr"] for pg in pgs))]
        print(f"Learning rates: {lrs}" if len(lrs) > 1 else f"Learning rate: {lrs[0]}")

        if cfg.optimizer == 'adam':
            try:
                from bitsandbytes.optim import Adam8bit as Adam

                print("Using 8-bit Adam from bitsandbytes")
            except ImportError:
                from torch.optim import Adam

                print("bitsandbytes 8-bit Adam not available, using torch.optim.Adam")
                print("Run `pip install bitsandbytes` for less memory usage.")

            assert isinstance(dataset, Sized)
            num_examples = len(dataset)

            self.optimizer = Adam(pgs)
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, cfg.lr_warmup_steps, num_examples // cfg.batch_size
            )
        else:
            self.optimizer = ScheduleFreeWrapper(SignSGD(pgs))
            self.optimizer.train()
            self.lr_scheduler = None
            
        self.global_step = 0
        self.num_tokens_since_fired = {
            name: torch.zeros(sae.num_latents, device=device, dtype=torch.long)
            for name, sae in self.saes.items()
        }

        num_latents = list(self.saes.values())[0].num_latents
        self.initial_k = min(num_latents, round(list(input_widths.values())[0] * 10))
        self.final_k = self.cfg.sae.k


    def load_state(self, path: str):
        """Load the trainer state from disk."""
        device = self.model.device

        # Load the train state first so we can print the step number
        train_state = torch.load(
            f"{path}/state.pt", map_location=device, weights_only=True
        )
        train_state["num_tokens_since_fired"] = {}

        for file in glob(f"{path}/rank_*_state.pt"):
            rank_train_state = torch.load(file, map_location=device)
            train_state["num_tokens_since_fired"].update(
                rank_train_state["num_tokens_since_fired"]
            )

        self.global_step = train_state["global_step"]
        self.num_tokens_since_fired = {
            k: train_state["num_tokens_since_fired"][k] for k in self.local_hookpoints()
        }

        print(
            f"\033[92mResuming training at step {self.global_step} from '{path}'\033[0m"
        )

        if self.lr_scheduler is not None:
            lr_state = torch.load(
                f"{path}/lr_scheduler.pt", map_location=device, weights_only=True
            )
            self.lr_scheduler.load_state_dict(lr_state)

        opt_state = torch.load(
            f"{path}/optimizer.pt", map_location=device, weights_only=True
        )
        self.optimizer.load_state_dict(opt_state)
        if self.cfg.optimizer == "signum":
            self.optimizer.train()

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

        if self.cfg.log_to_wandb and rank_zero:
            try:
                import wandb

                wandb.init(
                    name=self.cfg.run_name,
                    project="sae",
                    config=asdict(self.cfg),
                    save_code=True,
                )
            except ImportError:
                print("Weights & Biases not installed, skipping logging.")
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

        acc_steps = self.cfg.grad_acc_steps * self.cfg.micro_acc_steps
        denom = acc_steps * self.cfg.wandb_log_frequency
        num_tokens_in_step = 0

        # For logging purposes
        avg_auxk_loss = defaultdict(float)
        avg_fvu = defaultdict(float)
        avg_multi_topk_fvu = defaultdict(float)
        aux_loss = 0.0

        name_to_module = {
            name: self.model.base_model.get_submodule(name)
            for name in self.cfg.hookpoints
        }
        maybe_wrapped: dict[str, DDP] | dict[str, SparseCoder] = {}
        module_to_name = {v: k for k, v in name_to_module.items()}

        def hook(module: nn.Module, inputs, outputs):
            nonlocal aux_loss
            aux_out = None

            # Maybe unpack tuple inputs and outputs
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            if isinstance(outputs, tuple):
                outputs, *aux_out = outputs

            # Name may optionally contain a suffix of the form /seedN where N is an
            # integer. We only care about the part before the slash.
            name = module_to_name[module]

            # Remember the original output shape since we'll need it for e2e training
            out_shape = outputs.shape

            # Flatten the batch and sequence dimensions
            outputs = outputs.flatten(0, 1)
            inputs = inputs.flatten(0, 1) if self.cfg.transcode else outputs
            raw = self.saes[name]

            # On the first iteration, initialize the decoder bias
            if self.global_step == 0:
                mean = self.maybe_all_reduce(outputs.mean(0))
                raw.b_dec.data = mean.to(raw.dtype)
            
            # Make sure the W_dec is still unit-norm if we're autoencoding
            if raw.cfg.normalize_decoder and not self.cfg.transcode:
                raw.set_decoder_norm_to_unit_norm()

            wrapped = maybe_wrapped[name]
            out = wrapped(
                x=inputs,
                y=outputs,
                dead_mask=(
                    self.num_tokens_since_fired[name]
                    > self.cfg.dead_feature_threshold
                    if self.cfg.auxk_alpha > 0
                    else None
                ),
            )
            avg_fvu[name] += float(
                self.maybe_all_reduce(out.fvu.detach()) / denom
            )
            if self.cfg.auxk_alpha > 0:
                avg_auxk_loss[name] += float(
                    self.maybe_all_reduce(out.auxk_loss.detach()) / denom
                )
            if self.cfg.sae.multi_topk:
                avg_multi_topk_fvu[name] += float(
                    self.maybe_all_reduce(out.multi_topk_fvu.detach()) / denom
                )

            # Update the did_fire mask
            did_fire[name][out.latent_indices.flatten()] = True
            self.maybe_all_reduce(did_fire[name], "max")  # max is boolean "any"

            if self.cfg.end_to_end:
                #aux_loss += loss

                # Replace the normal output with the SAE output
                output = out.sae_out.reshape(out_shape).type_as(outputs)
                return (output, *aux_out) if aux_out is not None else output

            # Do a "local" backward pass if we're not training end-to-end
            loss = (
                out.fvu
                + self.cfg.auxk_alpha * out.auxk_loss
                + out.multi_topk_fvu / 8
            )
            loss.div(acc_steps).backward()

        k = self.get_current_k()
        for name, sae in self.saes.items():
            sae.cfg.k = k

        for batch in dl:
            if not maybe_wrapped:
                # Wrap the SAEs with Distributed Data Parallel. We have to do this
                # after we set the decoder bias, otherwise DDP will not register
                # gradients flowing to the bias after the first step.
                maybe_wrapped = (
                    {
                        name: DDP(
                            sae,
                            device_ids=[dist.get_rank()],
                            find_unused_parameters=self.cfg.end_to_end,
                        )
                        for name, sae in self.saes.items()
                    }
                    if ddp
                    else self.saes
                )

            # Bookkeeping for dead feature detection
            N = batch["input_ids"].numel()
            num_tokens_in_step += N

            # Forward pass on the model to get the next batch of activations
            handles = [
                mod.register_forward_hook(hook) for mod in name_to_module.values()
            ]
            try:
                ce_loss = None
                x = batch["input_ids"].to(device)

                # If we're training end-to-end, we need to pass the labels to the model
                # and do a backward pass through the entire model.
                if self.cfg.end_to_end:
                    ce_loss = self.model(x, labels=x).loss

                    alpha = self.cfg.local_loss_weight
                    loss = (
                        # Normalize the CE loss by log(vocab_size) to compare with FVU
                        (1 - alpha) * ce_loss / math.log(self.model.config.vocab_size) +
                        # Normalize the aux loss by the number of SAEs
                        alpha * aux_loss / len(self.saes)
                    )
                    loss.div(acc_steps).backward()
                else:
                    self.model(x)
            finally:
                aux_loss = 0.0
                for handle in handles:
                    handle.remove()

            if self.cfg.distribute_modules:
                input_dict = self.scatter_hiddens(input_dict)
                output_dict = self.scatter_hiddens(output_dict)

            for name, raw in self.saes.items():
                # Name may optionally contain a suffix of the form /seedN where N is an
                # integer. We only care about the part before the slash.
                hookpoint, _, _ = name.partition("/")

                # 'inputs' is distinct from outputs iff we're transcoding
                outputs = output_dict[hookpoint]
                inputs = input_dict.get(name, outputs)

                # On the first iteration, initialize the decoder bias
                if self.global_step == 0:
                    mean = self.maybe_all_reduce(outputs.mean(0))
                    raw.b_dec.data = mean.to(raw.dtype)

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

                # Make sure the W_dec is still unit-norm if we're autoencoding
                if raw.cfg.normalize_decoder and not self.cfg.sae.transcode:
                    raw.set_decoder_norm_to_unit_norm()

                acc_steps = self.cfg.grad_acc_steps * self.cfg.micro_acc_steps
                denom = acc_steps * self.cfg.wandb_log_frequency
                wrapped = maybe_wrapped[name]

                # Save memory by chunking the activations
                in_chunks = inputs.chunk(self.cfg.micro_acc_steps)
                out_chunks = outputs.chunk(self.cfg.micro_acc_steps)
                for in_chunk, out_chunk in zip(in_chunks, out_chunks):
                    out = wrapped(
                        x=in_chunk,
                        y=out_chunk,
                        dead_mask=(
                            self.num_tokens_since_fired[name]
                            > self.cfg.dead_feature_threshold
                            if self.cfg.auxk_alpha > 0
                            else None
                        ),
                    )

                    avg_fvu[name] += float(
                        self.maybe_all_reduce(out.fvu.detach()) / denom
                    )
                    if self.cfg.auxk_alpha > 0:
                        avg_auxk_loss[name] += float(
                            self.maybe_all_reduce(out.auxk_loss.detach()) / denom
                        )
                    if self.cfg.sae.multi_topk:
                        avg_multi_topk_fvu[name] += float(
                            self.maybe_all_reduce(out.multi_topk_fvu.detach()) / denom
                        )

                    loss = (
                        out.fvu
                        + self.cfg.auxk_alpha * out.auxk_loss
                        + out.multi_topk_fvu / 8
                    )
                    loss.div(acc_steps).backward()

                    # Update the did_fire mask
                    did_fire[name][out.latent_indices.flatten()] = True
                    self.maybe_all_reduce(did_fire[name], "max")  # max is boolean "any"

            # Check if we need to actually do a training step
            step, substep = divmod(self.global_step + 1, self.cfg.grad_acc_steps)
            if substep == 0:
                if self.cfg.sae.normalize_decoder and not self.cfg.sae.transcode:
                    for sae in self.saes.values():
                        sae.remove_gradient_parallel_to_decoder_directions()

                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

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

                if (
                    self.cfg.log_to_wandb
                    and (step + 1) % self.cfg.wandb_log_frequency == 0
                ):
                    info = {}
                    if ce_loss is not None:
                        info["ce_loss"] = ce_loss.item()

                    for name in self.saes:
                        mask = (
                            self.num_tokens_since_fired[name]
                            > self.cfg.dead_feature_threshold
                        )

                        info.update(
                            {
                                f"fvu/{name}": avg_fvu[name],
                                f"dead_pct/{name}": mask.mean(
                                    dtype=torch.float32
                                ).item(),
                            }
                        )
                        if self.cfg.auxk_alpha > 0:
                            info[f"auxk/{name}"] = avg_auxk_loss[name]
                        if self.cfg.sae.multi_topk:
                            info[f"multi_topk_fvu/{name}"] = avg_multi_topk_fvu[name]

                    avg_auxk_loss.clear()
                    avg_fvu.clear()
                    avg_multi_topk_fvu.clear()

                    if self.cfg.distribute_modules:
                        outputs = [{} for _ in range(dist.get_world_size())]
                        dist.gather_object(info, outputs if rank_zero else None)
                        info.update({k: v for out in outputs for k, v in out.items()})

                    if rank_zero:
                        info["k"] = k

                        wandb.log(info, step=step)

                if (step + 1) % self.cfg.save_every == 0:
                    self.save()

            self.global_step += 1
            pbar.update()

        self.save()
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

    def scatter_hiddens(self, hidden_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Scatter & gather the hidden states across ranks."""
        # Short-circuit if we have no data
        if not hidden_dict:
            return hidden_dict

        outputs = [
            # Add a new leading "layer" dimension to each tensor
            torch.stack([hidden_dict[hook] for hook in hookpoints], dim=1)
            for hookpoints in self.module_plan
        ]
        local_hooks = self.module_plan[dist.get_rank()]
        shape = next(iter(hidden_dict.values())).shape

        # Allocate one contiguous buffer to minimize memcpys
        buffer = outputs[0].new_empty(
            # The (micro)batch size times the world size
            shape[0] * dist.get_world_size(),
            # The number of layers we expect to receive
            len(local_hooks),
            # All other dimensions
            *shape[1:],
        )

        # Perform the all-to-all scatter
        inputs = buffer.split([len(output) for output in outputs])
        dist.all_to_all([x for x in inputs], outputs)

        # Return a list of results, one for each layer
        return {hook: buffer[:, i] for i, hook in enumerate(local_hooks)}

    def save(self):
        """Save the SAEs to disk."""

        path = self.cfg.run_name or "sae-ckpts"
        rank_zero = not dist.is_initialized() or dist.get_rank() == 0

        if rank_zero or self.cfg.distribute_modules:
            print("Saving checkpoint")

            for name, sae in self.saes.items():
                assert isinstance(sae, SparseCoder)

                sae.save_to_disk(f"{path}/{name}")

            rank = 0 if rank_zero else dist.get_rank()
            torch.save({"num_tokens_since_fired": self.num_tokens_since_fired}, f"{path}/rank_{rank}_state.pt")

        if rank_zero:
            if self.lr_scheduler is not None:
                torch.save(self.lr_scheduler.state_dict(), f"{path}/lr_scheduler.pt")
            
            if self.cfg.optimizer == "signum":
                self.optimizer.eval()
                torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pt")
                self.optimizer.train()
            else:
                torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pt")
            
            torch.save({"global_step": self.global_step}, f"{path}/state.pt")

            self.cfg.save_json(f"{path}/config.json")

        # Barrier to ensure all ranks have saved before continuing
        if dist.is_initialized():
            dist.barrier()

# Support old name for compatibility
SaeTrainer = Trainer
