from dataclasses import dataclass, field
from typing import List
import yaml, os

@dataclass
class ASEConfig:
    n_scales: int = 8
    n_filters: int = 16
    filter_len: int = 64
    latent_dim: int = 256
    dropout: float = 0.0

@dataclass
class SSSRConfig:
    state_dim: int = 256
    d_inner: int = 512
    dt_min: float = 0.001
    dt_max: float = 0.1
    n_heads: int = 8
    hebbian_lr: float = 1e-4
    stability_eps: float = 1e-3

@dataclass
class CRGConfig:
    n_nodes: int = 64
    sparsity_lambda: float = 0.05
    dag_lambda: float = 1.0
    n_lags: int = 5
    update_interval: int = 100
    max_edges: int = 256

@dataclass
class HMBConfig:
    buffer_size: int = 512
    archive_size: int = 4096
    embed_dim: int = 256
    compress_dim: int = 64
    surprise_threshold: float = 2.0
    consolidation_interval: int = 1000

@dataclass
class SHCALConfig:
    plasticity_rate: float = 1e-4
    ewc_lambda: float = 100.0
    fisher_samples: int = 200
    prune_threshold: float = 1e-3
    grow_threshold: float = 0.1

@dataclass
class DAHConfig:
    n_domains: int = 32
    adapter_rank: int = 16
    adapter_scale: float = 0.01
    meta_dim: int = 128

@dataclass
class ESEConfig:
    max_rules: int = 64
    max_depth: int = 5
    min_samples: int = 20
    cf_steps: int = 50
    cf_lr: float = 0.01

@dataclass
class HTDConfig:
    n_levels: int = 4
    time_constants: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    bottleneck_dim: int = 64

@dataclass
class SafetyConfig:
    cbf_gamma: float = 0.5
    lipschitz_bound: float = 10.0
    n_constraints: int = 8
    certified_radius: float = 0.1

@dataclass
class TrainingConfig:
    lr: float = 3e-4
    min_lr: float = 1e-6
    warmup_steps: int = 1000
    max_steps: int = 100000
    batch_size: int = 32
    seq_len: int = 1024
    grad_clip: float = 1.0
    spectral_clip: float = 2.0
    beta_hmb: float = 0.1
    gamma_crg: float = 0.01
    delta_ewc: float = 0.1
    epsilon_conformal: float = 0.05
    weight_decay: float = 0.01
    checkpoint_dir: str = "checkpoints"

@dataclass
class MultiTaskConfig:
    forecast_horizon: int = 10
    n_channels: int = 1

@dataclass
class ModelConfig:
    input_dim: int = 64
    output_dim: int = 64
    n_classes: int = 0
    dtype: str = "float32"
    device: str = "cpu"
    ase: ASEConfig = field(default_factory=ASEConfig)
    sssr: SSSRConfig = field(default_factory=SSSRConfig)
    crg: CRGConfig = field(default_factory=CRGConfig)
    hmb: HMBConfig = field(default_factory=HMBConfig)
    shcal: SHCALConfig = field(default_factory=SHCALConfig)
    dah: DAHConfig = field(default_factory=DAHConfig)
    ese: ESEConfig = field(default_factory=ESEConfig)
    htd: HTDConfig = field(default_factory=HTDConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    multitask: MultiTaskConfig = field(default_factory=MultiTaskConfig)

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """Construct a ModelConfig from environment variables, with defaults."""
        cfg = cls()
        cfg.input_dim = int(os.environ.get("VULGARIS_INPUT_DIM", 9))
        cfg.output_dim = int(os.environ.get("VULGARIS_OUTPUT_DIM", 1))
        cfg.n_classes = int(os.environ.get("VULGARIS_N_CLASSES", 5))
        cfg.ase.latent_dim = int(os.environ.get("VULGARIS_D_MODEL", 64))
        cfg.ase.n_filters = int(os.environ.get("VULGARIS_N_FILTERS", 16))
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str):
        import dataclasses
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False)
