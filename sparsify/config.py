from dataclasses import dataclass
from functools import partial
from typing import Literal

from simple_parsing import Serializable, list_field


@dataclass
class SparseCoderConfig(Serializable):
    """
    Configuration for training a sparse coder on a language model.
    """

    activation: Literal["groupmax", "topk"] = "topk"
    """Activation function to use."""

    expansion_factor: int = 32
    """Multiple of the input dimension to use as the sparse coder dimension."""

    normalize_decoder: bool = True
    """Normalize the decoder weights to have unit norm."""

    num_latents: int = 0
    """Number of latents to use. If 0, use `expansion_factor`."""

    k: int = 32
    """Number of nonzero features."""

    multi_topk: bool = False
    """Use Multi-TopK loss."""

    skip_connection: bool = False
    """Include a linear skip connection."""

    transcode: bool = False
    """Whether we want to predict the output of a module given its input."""

    n_targets: int = 0
    """Number of targets to predict. Only used if `transcode` is True."""

    n_sources: int = 0
    """Number of cross-layer sources writing to this layer. Only used if
    `transcode` is True and `cross_layer` is not 0."""

    normalize_io: bool = False
    """Normalize the input and output of the sparse coder."""

    divide_cross_layer: bool = False
    """Divide the preceding layer skip connections by the number of layers."""

    train_post_encoder: bool = True
    """Train the post-encoder bias."""

    post_encoder_scale: bool = False
    """Train a scale for post-encoder layers."""

    coalesce_topk: Literal["none", "concat", "per-layer", "group"] = "none"
    """How to combine values and indices across layers."""

    topk_coalesced: bool = True
    """Whether to actually apply topk to the coalesced values."""

    @property
    def do_coalesce_topk(self):
        return self.coalesce_topk != "none"


# Support different naming conventions for the same configuration
SaeConfig = SparseCoderConfig
TranscoderConfig = partial(SparseCoderConfig, transcode=True)


@dataclass
class TrainConfig(Serializable):
    sae: SparseCoderConfig

    batch_size: int = 32
    """Batch size measured in sequences."""

    grad_acc_steps: int = 1
    """Number of steps over which to accumulate gradients."""

    micro_acc_steps: int = 1
    """Chunk the activations into this number of microbatches for training."""

    loss_fn: Literal["ce", "fvu", "kl"] = "fvu"
    """Loss function to use for training the sparse coders.

    - `ce`: Cross-entropy loss of the final model logits.
    - `fvu`: Fraction of variance explained.
    - `kl`: KL divergence of the final model logits w.r.t. the original logits.
    """

    remove_transcoded_modules: bool = False
    """Don't run modules that are replaced for transcoders with CE loss."""

    optimizer: Literal["adam", "muon", "signum"] = "signum"
    """Optimizer to use."""

    lr: float | None = None
    """Base LR. If None, it is automatically chosen based on the number of latents."""

    lr_warmup_steps: int = 1000
    """Number of steps over which to warm up the learning rate. Only used if
    `optimizer` is `adam`."""

    force_lr_warmup: bool = False
    """Force the learning rate warmup even if `optimizer` is not `adam`."""

    k_decay_steps: int = 0
    """Number of steps over which to decay the number of active latents. Starts at
    input width * 10 and decays to k. Experimental feature."""

    auxk_alpha: float = 0.0
    """Weight of the auxiliary loss term."""

    dead_feature_threshold: int = 10_000_000
    """Number of tokens after which a feature is considered dead."""

    hookpoints: list[str] = list_field()
    """List of hookpoints to train sparse coders on."""

    init_seeds: list[int] = list_field(0)
    """List of random seeds to use for initialization. If more than one, train a sparse
    coder for each seed."""

    layers: list[int] = list_field()
    """List of layer indices to train sparse coders on."""

    per_layer_k: list[int] = list_field()
    """List of k values to use for each layer."""

    layer_stride: int = 1
    """Stride between layers to train sparse coders on."""

    cross_layer: int = 0
    """How many layers ahead to train the sparse coder on.
    If 0, train only on the same layer."""

    tp: int = 1
    """Number of tensor parallel ranks to use."""

    @property
    def distribute_modules(self) -> bool:
        """Whether to distribute the modules across ranks."""
        return self.tp > 1

    save_every: int = 1000
    """Save sparse coders every `save_every` steps."""

    save_best: bool = False
    """Save the best checkpoint found for each hookpoint."""

    finetune: str | None = None
    """Finetune the sparse coders from a pretrained checkpoint."""

    restart_epoch: bool = False
    """Start loading the dataset from the beginning after loading a checkpoint."""

    log_to_wandb: bool = True
    run_name: str | None = None
    wandb_log_frequency: int = 1

    save_dir: str = "checkpoints"

    def __post_init__(self):
        """Validate the configuration."""
        if self.layers and self.layer_stride != 1:
            raise ValueError("Cannot specify both `layers` and `layer_stride`.")

        if not self.init_seeds:
            raise ValueError("Must specify at least one random seed.")
