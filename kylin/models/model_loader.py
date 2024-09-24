from dataclasses import field, make_dataclass

from kylin.utils import Choices

from .model_base import (
    EncoderBase,
    Encoders,
    GeneratorBase,
    Generators,
    RankerBase,
    Rankers,
)


generator_fields = [
    ("generator_type", Choices(Generators.names), field(default=Generators.names[0]))
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


def load_generator(cfg: GeneratorConfig) -> GeneratorBase:  # type: ignore
    if cfg.generator_type in Generators:
        cfg_name = f"{Generators[cfg.generator_type]['short_names'][0]}_config"
        sub_cfg = getattr(cfg, cfg_name)
        return Generators[cfg.generator_type]["item"](sub_cfg)
    raise ValueError(f"Unknown generator type: {cfg.generator_type}")


encoder_fields = [
    ("encoder_type", Choices(Encoders.names), field(default=Encoders.names[0]))
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


def load_encoder(cfg: EncoderConfig) -> EncoderBase:  # type: ignore
    if cfg.encoder_type in Encoders:
        cfg_name = f"{Encoders[cfg.encoder_type]['short_names'][0]}_config"
        sub_cfg = getattr(cfg, cfg_name)
        return Encoders[cfg.encoder_type]["item"](sub_cfg)
    raise ValueError(f"Unknown encoder type: {cfg.encoder_type}")


ranker_fields = [
    ("ranker_type", Choices(Rankers.names), field(default=Rankers.names[0]))
]
ranker_fields += [
    (
        f"{Rankers[name]['short_names'][0]}_config",
        Rankers[name]["config_class"],
        field(default_factory=Rankers[name]["config_class"]),
    )
    for name in Rankers.mainnames
]
RankerConfig = make_dataclass("RankerConfig", ranker_fields)


def load_ranker(cfg: RankerConfig) -> RankerBase:  # type: ignore
    if cfg.ranker_type in Rankers:
        cfg_name = f"{Rankers[cfg.ranker_type]['short_names'][0]}_config"
        sub_cfg = getattr(cfg, cfg_name)
        return Rankers[cfg.ranker_type]["item"](sub_cfg)
    raise ValueError(f"Unknown ranker type: {cfg.ranker_type}")
