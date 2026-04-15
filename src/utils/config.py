"""
Configuration system: YAML file → Python dataclasses.
Supports dot-notation CLI overrides like `training.epochs=50`.
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class DataConfig:
    root_dir: str = "data/"
    split_csv: str = "split_unilateral.csv"
    label_csv: str = "labels.csv"
    sequences: List[str] = field(default_factory=lambda: ["Pre", "Post_1", "Post_2", "T2"])
    spatial_size: Tuple[int, int, int] = (128, 128, 32)
    fold: int = 0
    num_workers: int = 4
    batch_size: int = 4


@dataclass
class ModelConfig:
    architecture: str = "densenet121"
    in_channels: int = 4
    num_classes: int = 3
    dropout: float = 0.0
    pretrained: bool = False
    use_instancenorm: bool = False
    pretrain_path: str = ""


@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    scheduler_patience: int = 10
    early_stopping_patience: int = 15
    mixed_precision: bool = True
    class_weights: Optional[List[float]] = None
    label_smoothing: float = 0.0
    loss_type: str = "cross_entropy"  # "cross_entropy" or "focal"
    focal_gamma: float = 2.0
    oversample: bool = False


@dataclass
class AugmentationConfig:
    rand_flip_prob: float = 0.5
    rand_rotate90_prob: float = 0.5
    rand_affine_prob: float = 0.3
    rand_affine_rotate_range: float = 0.1745
    rand_affine_scale_range: List[float] = field(default_factory=lambda: [0.9, 1.1])
    rand_intensity_shift: float = 0.1
    rand_intensity_scale: float = 0.1
    use_percentile_norm: bool = False
    rand_gaussian_noise_prob: float = 0.0
    rand_gaussian_noise_std: float = 0.05
    derive_sub2: bool = False
    derive_washout: bool = False
    crop_foreground: bool = False


@dataclass
class CalibrationConfig:
    enabled: bool = False
    temperature_init: float = 1.5


@dataclass
class EvaluationConfig:
    sensitivity_threshold: float = 0.9
    specificity_threshold: float = 0.9


@dataclass
class PathsConfig:
    output_dir: str = "outputs/"
    checkpoint_dir: str = "outputs/checkpoints/"
    log_dir: str = "outputs/logs/"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    seed: int = 42
    device: str = "auto"


def _cast_value(value: str, target_type: type):
    """Cast a string CLI override to the target type."""
    if target_type == bool:
        return value.lower() in ("true", "1", "yes")
    if target_type == type(None):
        if value.lower() == "null" or value.lower() == "none":
            return None
        return value
    return target_type(value)


def _apply_overrides(cfg_dict: dict, overrides: List[str]) -> dict:
    """Apply dot-notation overrides like 'training.epochs=50' to the config dict."""
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override}")
        key, value = override.split("=", 1)
        parts = key.split(".")

        # Navigate to the parent dict
        d = cfg_dict
        for part in parts[:-1]:
            if part not in d:
                raise KeyError(f"Unknown config key: {key}")
            d = d[part]

        final_key = parts[-1]
        if final_key not in d:
            raise KeyError(f"Unknown config key: {key}")

        # Try to cast to the existing type
        existing = d[final_key]
        if existing is None:
            # For None values, try to parse as float list or leave as string
            if value.startswith("["):
                d[final_key] = yaml.safe_load(value)
            else:
                try:
                    d[final_key] = float(value)
                except ValueError:
                    d[final_key] = value
        elif isinstance(existing, list):
            d[final_key] = yaml.safe_load(value)
        elif isinstance(existing, bool):
            d[final_key] = value.lower() in ("true", "1", "yes")
        elif isinstance(existing, int):
            d[final_key] = int(value)
        elif isinstance(existing, float):
            d[final_key] = float(value)
        else:
            d[final_key] = value

    return cfg_dict


def _dict_to_config(d: dict) -> Config:
    """Convert a nested dict to a Config dataclass."""
    return Config(
        data=DataConfig(**d.get("data", {})),
        model=ModelConfig(**d.get("model", {})),
        training=TrainingConfig(**d.get("training", {})),
        augmentation=AugmentationConfig(**d.get("augmentation", {})),
        calibration=CalibrationConfig(**d.get("calibration", {})),
        evaluation=EvaluationConfig(**d.get("evaluation", {})),
        paths=PathsConfig(**d.get("paths", {})),
        seed=d.get("seed", 42),
        device=d.get("device", "auto"),
    )


def load_config(yaml_path: str, overrides: Optional[List[str]] = None) -> Config:
    """
    Load configuration from YAML file and apply optional CLI overrides.

    Args:
        yaml_path: Path to the YAML config file.
        overrides: List of 'key.subkey=value' strings.

    Returns:
        Config dataclass with all settings.
    """
    with open(yaml_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    if overrides:
        cfg_dict = _apply_overrides(cfg_dict, overrides)

    # Convert spatial_size list to tuple
    if "data" in cfg_dict and "spatial_size" in cfg_dict["data"]:
        cfg_dict["data"]["spatial_size"] = tuple(cfg_dict["data"]["spatial_size"])

    config = _dict_to_config(cfg_dict)

    # Ensure in_channels matches number of sequences + derived channels
    n_channels = len(config.data.sequences)
    if config.augmentation.derive_sub2:
        n_channels += 1
    if config.augmentation.derive_washout:
        n_channels += 1
    config.model.in_channels = n_channels

    # Create output directories
    for dir_path in [config.paths.output_dir, config.paths.checkpoint_dir, config.paths.log_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    return config
