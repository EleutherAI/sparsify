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
from .utils import get_layer_list, resolve_widths


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        dataset: HfDataset | MemmapDataset,
        model: PreTrainedModel,
    ):
        if cfg.hookpoints:
            assert not cfg.layers, "Cannot specify both `hookpoints` and `layers`."

            # Replace wildcard patterns
            raw_hookpoints = []
            for name, _ in model.named_modules():
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

        if not cfg.hookpoints:
            raise ValueError(
                f"""No valid hookpoints specified, available hookpoints:"""
                f"""\n {[item[0] for item in model.named_modules()]}"""
            )

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

        self.model = model

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

                adam = Adam(pgs, betas=(0., 0.999), )
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
                    torch.optim.Adam(params - muon_params, lr=cfg.lr or 2e-4, betas=(0., 0.999)),
                ]
                self.lr_schedulers = [
                    get_linear_schedule_with_warmup(
                        self.optimizers[0], 0, num_batches
                    ),
                    get_linear_schedule_with_warmup(
                        self.optimizers[1],
                        cfg.lr_warmup_steps,
                        num_batches
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
        self.save_count = 0

        num_latents = list(self.saes.values())[0].num_latents
        self.num_latents = num_latents
        self.initial_k = min(num_latents, round(list(input_widths.values())[0] * 10))
        self.final_k = self.cfg.sae.k
        self.best_loss = float("inf")

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

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        ddp = dist.is_initialized() and not self.cfg.distribute_modules

        wandb = None
        if self.cfg.log_to_wandb and rank_zero:
            try:
                import wandb

                wandb.init(
                    name=self.cfg.run_name,
                    project="sae",
                    config=asdict(self.cfg),
                    save_code=True,
                )
            except (AttributeError, ImportError):
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
        loss = torch.inf
        num_tokens_in_step = 0

        # For logging purposes
        avg_auxk_loss = defaultdict(float)
        avg_fvu = defaultdict(float)
        avg_multi_topk_fvu = defaultdict(float)

        input_dict: dict[str, Tensor] = {}
        output_dict: dict[str, Tensor] = {}
        name_to_module = {
            name: self.model.get_submodule(name) for name in self.cfg.hookpoints
        }
        maybe_wrapped: dict[str, DDP] | dict[str, SparseCoder] = {}
        module_to_name = {v: k for k, v in name_to_module.items()}

        def hook(module: nn.Module, inputs, outputs):
            # Maybe unpack tuple inputs and outputs
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            name = module_to_name[module]
            output_dict[name] = outputs.flatten(0, 1)

            # Remember the inputs if we're training a transcoder
            if self.cfg.sae.transcode:
                input_dict[name] = inputs.flatten(0, 1)

        k = self.get_current_k()
        for name, sae in self.saes.items():
            sae.cfg.k = k

        for batch in dl:
            input_dict.clear()
            output_dict.clear()

            # Bookkeeping for dead feature detection
            N = batch["input_ids"].numel()
            num_tokens_in_step += N

            # Forward pass on the model to get the next batch of activations
            handles = [
                mod.register_forward_hook(hook) for mod in name_to_module.values()
            ]
            try:
                with torch.no_grad():
                    self.model(batch["input_ids"].to(device))
            finally:
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

                if self.cfg.sae.activation == "jumprelu":
                    # normalize activations to mean squared L2 norm of 1
                    scaling_factor = torch.rsqrt(inputs.pow(2).sum(-1).mean(0))
                    # TODO test using one scaling factor for both inputs and outputs when transcoding
                    inputs = inputs * scaling_factor

                    if self.cfg.sae.transcode:
                        scaling_factor = torch.rsqrt(outputs.pow(2).sum(-1).mean(0))
                    
                    outputs = outputs * scaling_factor

                if not self.cfg.sae.activation == "jumprelu":
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

                    # print("out.l0", out.l0)
                    # print("out.fvu", out.fvu)

                    loss = (
                        out.fvu # rescaled MSE
                        + out.l0 * self.get_current_lambda()
                    )
                    # print("l0", out.per_example_l0)
                    # print("lambda", self.get_current_lambda())

                    loss.div(acc_steps).backward()

                    # Update the did_fire mask
                    did_fire[name][out.did_fire] = True
                    self.maybe_all_reduce(did_fire[name], "max")  # max is boolean "any"

            # Check if we need to actually do a training step
            step, substep = divmod(self.global_step + 1, self.cfg.grad_acc_steps)
            if substep == 0:
                if (
                    self.cfg.sae.normalize_decoder
                    and not self.cfg.sae.transcode
                    and not self.cfg.sae.activation == "jumprelu"
                ):
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

                if (
                    self.cfg.log_to_wandb
                    and (step + 1) % self.cfg.wandb_log_frequency == 0
                ):
                    info = {}

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
                        info["k"] = out.per_example_l0

                        if wandb is not None:
                            wandb.log(info, step=step)

                if (step + 1) % self.cfg.save_every == 0 and loss < self.best_loss:
                    self.best_loss = loss
                    self.save()

            self.global_step += 1
            pbar.update()

        if loss < self.best_loss:
            self.best_loss = loss
            self.save()

        pbar.close()


    def get_current_lambda(self) -> float:
        lmda = 20

        if self.global_step >= self.cfg.lambda_warmup_steps:
            return lmda

        progress = self.global_step / self.cfg.lambda_warmup_steps
        starting_val = 0.1

        return starting_val + (lmda - starting_val) * progress

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

        path = f"checkpoints/{self.cfg.run_name}" or "checkpoints/unnamed"
        path = f"{path}-{self.save_count % 2}"
        self.save_count += 1
        print("saving to", path)

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0

        for optimizer in self.optimizers:
            if hasattr(optimizer, "eval"):
                optimizer.eval()  # type: ignore

        if rank_zero or self.cfg.distribute_modules:
            print("Saving checkpoint")

            for name, sae in self.saes.items():
                assert isinstance(sae, SparseCoder)

                sae.save_to_disk(f"{path}/{name}")

            rank = 0 if rank_zero else dist.get_rank()
            torch.save(
                {"num_tokens_since_fired": self.num_tokens_since_fired},
                f"{path}/rank_{rank}_state.pt",
            )

        if rank_zero:
            for i, scheduler in enumerate(self.lr_schedulers):
                torch.save(scheduler.state_dict(), f"{path}/lr_scheduler_{i}.pt")

            for i, optimizer in enumerate(self.optimizers):
                print("saving optimizer", i)
                torch.save(optimizer.state_dict(), f"{path}/optimizer_{i}.pt")


            print("saving state")
            torch.save({"global_step": self.global_step}, f"{path}/state.pt")

            print("saving config")
            self.cfg.save_json(f"{path}/config.json")

        for optimizer in self.optimizers:
            if hasattr(optimizer, "train"):
                optimizer.train()  # type: ignore

        # Barrier to ensure all ranks have saved before continuing
        if dist.is_initialized():
            dist.barrier()


# Support old name for compatibility
SaeTrainer = Trainer
