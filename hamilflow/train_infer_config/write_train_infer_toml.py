from dataclasses import dataclass, field, fields, is_dataclass, asdict
from pathlib import Path
from typing import Optional, List, Callable, Literal, get_type_hints, get_args, get_origin
import copy
import tomli_w

# ============================== SYSTEM ==============================

@dataclass
class InferSystemConfig:
    note: str = "Enjoy DeepH-pack! ;-)"
    device: str = "gpu*8:0"
    float_type: Literal["bf16", "tf32", "fp32", "fp64"] = "fp32"
    random_seed: int = 137
    log_level: Literal["debug", "info", "warning", "critical"] = "info"
    jax_memory_preallocate: bool = True

@dataclass
class TrainSystemConfig(InferSystemConfig):
    show_train_process_bar: bool = True

# ================================ DATA ===============================

@dataclass
class DftConfig:
    data_dir_depth: int = 0
    validation_check: Optional[bool] = None  # infer-only quirk key, not in guide's table at all — flag if you actually use it

@dataclass
class GraphConfig:
    dataset_name: str = "DATASET-DEMO"
    graph_type: Literal["H", "HS", "Rho", "Sap", "S"] = "H"  # infer restricts further, see InferGraphConfig
    storage_type: Literal["memory", "disk"] = "memory"
    common_orbital_types: str = ""
    parallel_num: int = -1
    only_save_graph: bool = False

@dataclass
class InferGraphConfig(GraphConfig):
    graph_type: Literal["Sap", "S"] = "S"
    dataset_name: str = "INFER-DEMO"

@dataclass
class ModelSaveConfig:
    best: bool = True
    latest: bool = True
    latest_interval: int = 100
    latest_num: int = 10

@dataclass
class InferDataConfig:
    inputs_dir: str = "./user/should/set/this/inputs"
    outputs_dir: str = "./user/should/set/this/outputs"
    dft: DftConfig = field(default_factory=DftConfig)
    graph: InferGraphConfig = field(default_factory=InferGraphConfig)

@dataclass
class TrainDataConfig(InferDataConfig):
    inputs_dir: str = "<Invalid-Input>"   # required, no real default
    outputs_dir: str = "<Invalid-Input>"
    graph: GraphConfig = field(default_factory=GraphConfig)
    model_save: ModelSaveConfig = field(default_factory=ModelSaveConfig)

# ================================ MODEL ===============================

@dataclass
class InferModelConfig:
    model_dir: str = "<Invalid-Input>"
    load_model_type: Literal["best", "latest"] = "best"
    load_model_epoch: int = -1

@dataclass
class ModelAdvancedConfig:
    gaussian_basis_rmax: float = 7.5
    net_irreps: str = "<Invalid-Input>"
    num_blocks: int = 3
    consider_parity: bool = True
    standardize_gauge: bool = False

@dataclass
class TrainModelConfig:
    net_type: Literal["sparrow", "normal", "eagle", "accurate"] = "normal"
    target_type: Literal["H", "Rho"] = "H"
    loss_type: Literal["mae", "mse", "wmae", "huber", "ai2dft", "ai2dft_node", "hopad", "aims"] = "mse"
    advanced: ModelAdvancedConfig = field(default_factory=ModelAdvancedConfig)

# ============================== PROCESS ================================

@dataclass
class InferDataloaderConfig:
    batch_size: int = 1

@dataclass
class ProcessInferConfig:
    output_type: Literal["h5", "petsc"] = "h5"
    output_into: Literal["to_output", "to_input"] = "to_output"
    target_symmetrize: bool = True
    multi_way_jit_num: int = 1
    dataloader: InferDataloaderConfig = field(default_factory=InferDataloaderConfig)

@dataclass
class TrainDataloaderConfig:
    batch_size: int = 1
    train_size: int = 1
    validate_size: int = 0
    test_size: int = 0
    dataset_split_json: str = ""
    only_use_train_loss: bool = False

@dataclass
class OptimizerConfig:
    type: Literal["sgd", "adam", "adamw"] = "adamw"
    init_learning_rate: float = 2e-3
    clip_norm_factor: float = -1.0
    momentum: float = 0.8          # sgd only (informational)
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])  # adam/adamw only (informational)
    weight: float = 0.001          # adamw only (informational)
    eps: float = 1e-8              # adam/adamw only (informational)

@dataclass
class SchedulerConfig:
    min_learning_rate_scale: float = 1e-4
    type: Literal["reduce_on_plateau", "warmup_cosine_decay"] = "reduce_on_plateau"
    factor: float = 0.5            # reduce_on_plateau only (informational)
    patience: int = 500            # reduce_on_plateau only (informational)
    rtol: float = 0.05             # reduce_on_plateau only (informational)
    cooldown: int = 100            # reduce_on_plateau only (informational)
    accum_size: int = -1           # reduce_on_plateau only (informational)
    init_scale: float = 0.1        # warmup_cosine_decay only (informational)
    warmup_steps: int = 1000       # warmup_cosine_decay only (informational)
    decay_steps: int = int(2e5)    # warmup_cosine_decay only (informational)
    end_scale: float = -1.0        # warmup_cosine_decay only (informational)

@dataclass
class ContinuedConfig:
    enable: bool = False
    new_training_data: bool = False
    new_optimizer: bool = False
    previous_output_dir: str = "<Invalid-Input>"
    load_model_type: Literal["best", "latest"] = "latest"
    load_model_epoch: int = -1

@dataclass
class ProcessTrainConfig:
    max_epoch: int = 10000
    multi_way_jit_num: int = 1
    ahead_of_time_compile: bool = False
    dataloader: TrainDataloaderConfig = field(default_factory=TrainDataloaderConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    continued: ContinuedConfig = field(default_factory=ContinuedConfig)

# ============================ TOP LEVEL ================================

@dataclass
class InferConfig:
    system: InferSystemConfig = field(default_factory=InferSystemConfig)
    data: InferDataConfig = field(default_factory=InferDataConfig)
    model: InferModelConfig = field(default_factory=InferModelConfig)
    process: ProcessInferConfig = field(default_factory=ProcessInferConfig)

@dataclass
class TrainConfig:
    system: TrainSystemConfig = field(default_factory=TrainSystemConfig)
    data: TrainDataConfig = field(default_factory=TrainDataConfig)
    model: TrainModelConfig = field(default_factory=TrainModelConfig)
    process: ProcessTrainConfig = field(default_factory=ProcessTrainConfig)


# ============================ VALIDATION ================================

def _choices(tp) -> Optional[tuple]:
    return get_args(tp) if get_origin(tp) is Literal else None

def validate_keys(schema_cls, overrides: dict, path: str = "") -> None:
    """Check keys exist AND, for Literal-typed fields, that values are in the allowed set."""
    hints = get_type_hints(schema_cls)
    for key, value in overrides.items():
        full_key = f"{path}.{key}" if path else key
        if key not in hints:
            raise KeyError(f"Unknown config key: '{full_key}'")
        tp = hints[key]
        if isinstance(value, dict):
            nested_cls = tp
            if not is_dataclass(nested_cls):
                raise TypeError(f"'{full_key}' expected a scalar, got a table")
            validate_keys(nested_cls, value, full_key)
        else:
            choices = _choices(tp)
            if choices is not None and value not in choices:
                raise ValueError(f"'{full_key}' = {value!r} not in allowed set {choices}")


def apply_overrides(instance, overrides: dict):
    instance = copy.deepcopy(instance)
    for key, value in overrides.items():
        current = getattr(instance, key)
        if isinstance(value, dict) and is_dataclass(current):
            setattr(instance, key, apply_overrides(current, value))
        else:
            setattr(instance, key, value)
    return instance


def _to_toml_dict(config) -> dict:
    def strip_none(d: dict) -> dict:
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out[k] = strip_none(v)
            elif v is not None:
                out[k] = v
        return out
    return strip_none(asdict(config))


# ======================= HARD CONSTRAINTS (from guide's "must"/"cannot") =======================

@dataclass
class Rule:
    check: Callable[["TrainConfig"], bool]
    message: str

def validate_constraints(config, rules: List[Rule]) -> None:
    for rule in rules:
        if not rule.check(config):
            raise ValueError(rule.message)

def _has_odd_parity(irreps: str) -> bool:
    # crude but sufficient: e3nn irreps strings mark odd parity with a trailing 'o' per term
    return any(term.strip().endswith("o") for term in irreps.split("+"))

TRAIN_RULES: List[Rule] = [
    Rule(
        check=lambda c: c.model.net_type not in ("eagle", "owl") or c.model.advanced.consider_parity is False,
        message="net_type in {'eagle','owl'} requires model.advanced.consider_parity=False",
    ),
    Rule(
        check=lambda c: c.model.advanced.consider_parity or not _has_odd_parity(c.model.advanced.net_irreps),
        message="consider_parity=False forbids odd-parity ('o') terms in net_irreps",
    ),
    Rule(
        check=lambda c: not c.model.advanced.standardize_gauge or c.data.graph.graph_type == "HS",
        message="standardize_gauge=True requires data.graph.graph_type='HS'",
    ),
]


# ============================ WRITERS ================================

def write_train_toml(overrides: dict, out_dir, filename: str = "train.toml") -> Path:
    validate_keys(TrainConfig, overrides)
    config = apply_overrides(TrainConfig(), overrides)
    validate_constraints(config, TRAIN_RULES)
    out_path = Path(out_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        tomli_w.dump(_to_toml_dict(config), f)
    return out_path


def write_infer_toml(overrides: dict, out_dir, filename: str = "infer.toml") -> Path:
    validate_keys(InferConfig, overrides)
    config = apply_overrides(InferConfig(), overrides)
    out_path = Path(out_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        tomli_w.dump(_to_toml_dict(config), f)
    return out_path
