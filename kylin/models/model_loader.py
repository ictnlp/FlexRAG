from dataclasses import field, make_dataclass
from typing import Optional

from kylin.utils import Choices

from .model_base import (
    EncoderBase,
    Encoders,
    GeneratorBase,
    Generators,
)


generator_fields = [
    ("generator_type", Optional[Choices(Generators.names)], field(default=None))
]
generator_fields += [
    (
        f"{Generators[name]['short_names'][0]}_config",
        Generators[name]["config_class"],
        field(default_factory=Generators[name]["config_class"]),
    )
    for name in Generators.mainnames
]
GeneratorConfig = make_dataclass("GeneratorConfig", generator_fields)


def load_generator(cfg: GeneratorConfig) -> GeneratorBase | None:  # type: ignore
    if cfg.generator_type is None:
        return None
    if cfg.generator_type in Generators:
        cfg_name = f"{Generators[cfg.generator_type]['short_names'][0]}_config"
        sub_cfg = getattr(cfg, cfg_name)
        return Generators[cfg.generator_type]["item"](sub_cfg)
    raise ValueError(f"Unknown generator type: {cfg.generator_type}")


encoder_fields = [
    ("encoder_type", Optional[Choices(Encoders.names)], field(default=None))
]
encoder_fields += [
    (
        f"{Encoders[name]['short_names'][0]}_config",
        Encoders[name]["config_class"],
        field(default_factory=Encoders[name]["config_class"]),
    )
    for name in Encoders.mainnames
]
EncoderConfig = make_dataclass("EncoderConfig", encoder_fields)


def load_encoder(cfg: EncoderConfig) -> EncoderBase | None:  # type: ignore
    if cfg.encoder_type is None:
        return None
    if cfg.encoder_type in Encoders:
        cfg_name = f"{Encoders[cfg.encoder_type]['short_names'][0]}_config"
        sub_cfg = getattr(cfg, cfg_name)
        return Encoders[cfg.encoder_type]["item"](sub_cfg)
    raise ValueError(f"Unknown encoder type: {cfg.encoder_type}")
