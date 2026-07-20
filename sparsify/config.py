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

    embed_skip: bool = False
    """Include a learned affine skip connection from the model's embedding
    activations (rather than this hookpoint's own input) to the sparse coder
    output, as in a skip transcoder. Meant to control for aspects of the
    reconstruction explainable by generic token/dataset geometry already
    present at the embedding layer."""

    gram_lookup: bool = False
    """Subtract a frozen n-gram conditional-mean activation from the target (and, for
    autoencoders, from the encoder input) before training, reconstructing the residual
    ``y - mu[gram_id]``. This is a null model for local-context/token geometry: whatever
    the frozen table explains is provably not SAE computation. The table (mean activations
    per suffix n-gram) is built offline by ``sparsify.ngram_stats`` and supplied per batch
    by the trainer via the ``gram_means`` argument to ``forward``. Generalizes
    ``embed_skip`` and the Tokenized-SAE per-token bias. Kept orthogonal to ``embed_skip``."""

    lookup_max_order: int = 4
    """Maximum n-gram order used by ``gram_lookup``. 0 subtracts the global mean only
    (a near-baseline), 1 is a frozen per-token bias, 2-4 use the longest-suffix table
    restricted to that order via the backoff-parent ancestor remap."""

    gram_table_hash: str | None = None
    """Manifest hash of the n-gram table this coder was trained against (provenance)."""

    transcode: bool = False
    """Whether we want to predict the output of a module given its input."""


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

    optimizer: Literal["adam", "muon", "signum"] = "signum"
    """Optimizer to use."""

    lr: float | None = None
    """Base LR. If None, it is automatically chosen based on the number of latents."""

    lr_warmup_steps: int = 1000
    """Number of steps over which to warm up the learning rate. Only used if
    `optimizer` is `adam`."""

    k_decay_steps: int = 0
    """Number of steps over which to decay the number of active latents. Starts at
    input width * 10 and decays to k. Experimental feature."""

    auxk_alpha: float = 0.0
    """Weight of the auxiliary loss term."""

    dead_feature_threshold: int = 10_000_000
    """Number of tokens after which a feature is considered dead."""

    exclude_tokens: list[int] = list_field()
    """List of tokens to ignore during sparse coders training."""

    hookpoints: list[str] = list_field()
    """List of hookpoints to train sparse coders on."""

    init_seeds: list[int] = list_field(0)
    """List of random seeds to use for initialization. If more than one, train a sparse
    coder for each seed."""

    layers: list[int] = list_field()
    """List of layer indices to train sparse coders on."""

    layer_stride: int = 1
    """Stride between layers to train sparse coders on."""

    distribute_modules: bool = False
    """Store one copy of each sparse coder, instead of copying them across devices."""

    save_every: int = 1000
    """Save sparse coders every `save_every` steps."""

    save_best: bool = False
    """Save the best checkpoint found for each hookpoint."""

    finetune: str | None = None
    """Finetune the sparse coders from a pretrained checkpoint."""

    log_to_wandb: bool = True
    run_name: str | None = None
    wandb_log_frequency: int = 1

    save_dir: str = "checkpoints"

    gram_table_path: str | None = None
    """Path to a frozen n-gram table (built by ``sparsify.ngram_stats``). Required when
    any sparse coder sets ``sae.gram_lookup=True``; ignored otherwise."""

    def __post_init__(self):
        """Validate the configuration."""
        if self.layers and self.layer_stride != 1:
            raise ValueError("Cannot specify both `layers` and `layer_stride`.")

        if self.distribute_modules and self.loss_fn in ("ce", "kl"):
            raise ValueError(
                "Distributing modules across ranks is not compatible with the "
                "cross-entropy or KL divergence losses."
            )

        if not self.init_seeds:
            raise ValueError("Must specify at least one random seed.")
