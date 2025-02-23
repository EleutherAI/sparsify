from dataclasses import dataclass

from simple_parsing import Serializable, list_field

from .monet import MonetConfig

@dataclass
class SaeConfig(Serializable):
    """
    Configuration for training a sparse autoencoder on a language model.
    """

    expansion_factor: int = 32
    """Multiple of the input dimension to use as the SAE dimension."""

    normalize_decoder: bool = True
    """Normalize the decoder weights to have unit norm."""
    
    monet: bool = False
    """Use Monet architecture (router + MLP) for the SAE."""
    
    monet_config: MonetConfig = MonetConfig()
    """Configuration for the Monet architecture."""
    
    encoder_monarch: bool = False
    """Use Monarch matrix for the encoder."""
    
    monarch_inner_dim: int = 8
    """Dimension of the Monarch group."""
    monarch_mul: int = 32
    """Dense expansion factor for the Monarch matmul."""
    
    encoder_mpnet: bool = False
    """MPNet (matching pursuit) encoder."""
    
    encoder_halut: bool = False
    """Whether to use hashed lookup tables for the encoder weights."""
    
    # uv run python -m sparsify EleutherAI/pythia-160m togethercomputer/RedPajama-Data-1T-Sample --encoder_pkm=True --transcode=True --run_name pkm_saes/with_pkm_transcoder
    # uv run python -m sparsify "HuggingFaceTB/SmolLM2-135M" "HuggingFaceFW/fineweb" --subset sample-10BT --transcode=True --skip_connection=True --layers 3 6 9
    encoder_pkm: bool = False
    """Whether to use Product Key Memory for the encoder weights."""

    pkm_pad: bool = False
    """Pad the PKM encoder to a power of 2."""

    topk_separate: bool = True
    """Faster top-k for PKM by separating the top-k operation."""

    pkm_bias: bool = True
    """Non-decomposed bias for PKM."""
    
    pkm_init_scale: float = 1.0
    """Scale factor for PKM encoder initialization."""
    
    encoder_kron: bool = False
    """Use Kronecker product for the encoder."""
    
    kron_in_group: int = 2
    """Number of input groups for the Kronecker product."""
    
    kron_out_group: int = 4
    """Number of output groups for the Kronecker product."""
    
    kron_u: int = 4
    """Number of matrices to mix for the Kronecker product."""
    
    kron_lora: float = 1.0
    """How much to reduce the dimensionality of the input to the Kronecker product."""
    
    @property
    def encoder_weird(self):
        return self.encoder_monarch or self.encoder_mpnet or self.encoder_halut or self.encoder_pkm or self.encoder_kron
    
    decoder_xformers: bool = True
    """Xformers implementation for the decoder."""

    num_latents: int = 0
    """Number of latents to use. If 0, use `expansion_factor`."""

    k: int = 32
    """Number of nonzero features."""

    multi_topk: bool = False
    """Use Multi-TopK loss."""

    skip_connection: bool = False
    """Include a linear skip connection."""


@dataclass
class TrainConfig(Serializable):
    sae: SaeConfig

    batch_size: int = 8
    """Batch size measured in sequences."""

    grad_acc_steps: int = 1
    """Number of steps over which to accumulate gradients."""

    micro_acc_steps: int = 1
    """Chunk the activations into this number of microbatches for SAE training."""

    lr: float | None = None
    """Base LR. If None, it is automatically chosen based on the number of latents."""

    sfssgd: bool = False
    """Schedule-free SignSGD."""
    
    sf_momentum: float = 0.9
    """Momentum for Schedule-Free optimizers."""

    lr_warmup_steps: int = 1000

    auxk_alpha: float = 0.0
    """Weight of the auxiliary loss term."""

    dead_feature_threshold: int = 10_000_000
    """Number of tokens after which a feature is considered dead."""

    hookpoints: list[str] = list_field()
    """List of hookpoints to train SAEs on."""

    init_seeds: list[int] = list_field(0)
    """List of random seeds to use for initialization. If more than one, train an SAE
    for each seed."""

    layers: list[int] = list_field()
    """List of layer indices to train SAEs on."""

    layer_stride: int = 1
    """Stride between layers to train SAEs on."""

    transcode: bool = False
    """Predict the output of a module given its input."""

    distribute_modules: bool = False
    """Store a single copy of each SAE, instead of copying them across devices."""

    save_every: int = 1000
    """Save SAEs every `save_every` steps."""

    log_to_wandb: bool = True
    run_name: str | None = None
    wandb_log_frequency: int = 1

    def __post_init__(self):
        assert not (
            self.layers and self.layer_stride != 1
        ), "Cannot specify both `layers` and `layer_stride`."

        assert len(self.init_seeds) > 0, "Must specify at least one random seed."
