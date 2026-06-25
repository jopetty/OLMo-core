"""
Launch ladder experiments trained exclusively on a synthetic sensitivity dataset.

Unlike ``sensitivity_ladder.py``, this ladder does not mix in the standard Dolma/OLMo
pretraining distribution. The selected synthetic dataset is repeated to fill the full
Chinchilla token budget for the requested model size.

Typical usage:

    uv run src/scripts/train/ladder/synthetic_ladder.py launch \
      --size 20M \
      --model-type transformer \
      --mixture-dataset aperiodic_supervised_n200000000_v26_a50_m64_z1p2_s3 \
      --chinchilla-multiple 8 \
      --max-gpus 8 \
      --cluster ai2/jupiter \
      --workspace ai2/linear-rnns \
      --budget ai2/oe-other \
      --priority urgent

The default save layout is:

    /weka/.../synthetic-ladder/{size}/{model_type}/{dataset}/Cx{chinchilla_multiple}

By default the synthetic datasets are loaded from:

    /weka/oe-training-default/jacksonp/sensitivity-data/data/processed/{dataset}/*.npy
"""

import argparse
import math
from dataclasses import dataclass

from olmo_core.config import DType, StrEnum
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    ConcatAndChunkInstanceSourceConfig,
    InstanceFilterConfig,
    NumpyDocumentSourceConfig,
    SamplingInstanceSourceConfig,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.common import get_gpu_type, get_root_dir
from olmo_core.internal.ladder import get_requested_sizes, main
from olmo_core.io import join_path
from olmo_core.model_ladder import (
    DeviceMeshSpec,
    ModelLadder,
    WSDSChinchillaRunConfigurator,
)
from olmo_core.nn.attention import AttentionConfig, AttentionType, GateConfig, GateGranularity
from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
from olmo_core.nn.feed_forward import ActivationFunction, FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation
from olmo_core.nn.transformer.config import (
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)

from sensitivity_ladder import (
    HybridSmallSuiteModelConfigurator,
    SensitivityModelType,
    _attention_backend,
    _format_chinchilla_multiple,
    _wandb_tags,
)

SYNTHETIC_DATA_ROOT = "/weka/oe-training-default/jacksonp/sensitivity-data"

SYNTHETIC_DATASETS: dict[str, tuple[str, int]] = {
    # "aperiodic_supervised_n200000000_v26_a50_m64_z1p2_s3": (
    #     "data/processed/aperiodic_supervised_n200000000_v26_a50_m64_z1p2_s3",
    #     39_955_714_725,
    # ),
    # "aperiodic_supervised_n200000000_v26_a50_m64_z1p2_s3_gte2048": (
    #     "data/processed/aperiodic_supervised_n200000000_v26_a50_m64_z1p2_s3_gte2048",
    #     9_266_616_069,
    # ),
    "aperiodic_supervised_n200000000_v26_a50_m64_z1p2_s3_lt2048": (
        "data/processed/aperiodic_supervised_n200000000_v26_a50_m64_z1p2_s3_lt2048",
        30_689_098_656,
    ),
    # "aperiodic_unsupervised_n200000000_v26_a50_m64_z1p2_s2": (
    #     "data/processed/aperiodic_unsupervised_n200000000_v26_a50_m64_z1p2_s2",
    #     12_989_047_750,
    # ),
    # "aperiodic_unsupervised_n200000000_v26_a50_m64_z1p2_s2_gte512": (
    #     "data/processed/aperiodic_unsupervised_n200000000_v26_a50_m64_z1p2_s2_gte512",
    #     869_803_741,
    # ),
    "aperiodic_unsupervised_n200000000_v26_a50_m64_z1p2_s2_lt512": (
        "data/processed/aperiodic_unsupervised_n200000000_v26_a50_m64_z1p2_s2_lt512",
        12_119_244_009,
    ),
    # "periodic_supervised_n200000000_v26_a50_m64_z1p2_s5": (
    #     "data/processed/periodic_supervised_n200000000_v26_a50_m64_z1p2_s5",
    #     39_245_384_989,
    # ),
    # "periodic_supervised_n200000000_v26_a50_m64_z1p2_s5_gte2048": (
    #     "data/processed/periodic_supervised_n200000000_v26_a50_m64_z1p2_s5_gte2048",
    #     8_994_499_443,
    # ),
    "periodic_supervised_n200000000_v26_a50_m64_z1p2_s5_lt2048": (
        "data/processed/periodic_supervised_n200000000_v26_a50_m64_z1p2_s5_lt2048",
        30_250_885_546,
    ),
    # "periodic_unsupervised_n200000000_v26_a50_m64_z1p2_s4": (
    #     "data/processed/periodic_unsupervised_n200000000_v26_a50_m64_z1p2_s4",
    #     12_279_699_672,
    # ),
    # "periodic_unsupervised_n200000000_v26_a50_m64_z1p2_s4_gte512": (
    #     "data/processed/periodic_unsupervised_n200000000_v26_a50_m64_z1p2_s4_gte512",
    #     582_433_300,
    # ),
    "periodic_unsupervised_n200000000_v26_a50_m64_z1p2_s4_lt512": (
        "data/processed/periodic_unsupervised_n200000000_v26_a50_m64_z1p2_s4_lt512",
        11_697_266_372,
    ),
    # "r-trivial_supervised_n200000000_v26_a50_m64_z1p2_s1": (
    #     "data/processed/r-trivial_supervised_n200000000_v26_a50_m64_z1p2_s1",
    #     14_550_577_590,
    # ),
    # "r-trivial_supervised_n200000000_v26_a50_m64_z1p2_s1_gte2048": (
    #     "data/processed/r-trivial_supervised_n200000000_v26_a50_m64_z1p2_s1_gte2048",
    #     7_063_790,
    # ),
    "r-trivial_supervised_n200000000_v26_a50_m64_z1p2_s1_lt2048": (
        "data/processed/r-trivial_supervised_n200000000_v26_a50_m64_z1p2_s1_lt2048",
        14_543_513_800,
    ),
    # "r-trivial_unsupervised_n200000000_v26_a50_m64_z1p2_s0": (
    #     "data/processed/r-trivial_unsupervised_n200000000_v26_a50_m64_z1p2_s0",
    #     7_219_242_800,
    # ),
    # "r-trivial_unsupervised_n200000000_v26_a50_m64_z1p2_s0_gte256": (
    #     "data/processed/r-trivial_unsupervised_n200000000_v26_a50_m64_z1p2_s0_gte256",
    #     969_644_472,
    # ),
    "r-trivial_unsupervised_n200000000_v26_a50_m64_z1p2_s0_lt256": (
        "data/processed/r-trivial_unsupervised_n200000000_v26_a50_m64_z1p2_s0_lt256",
        6_249_598_328,
    ),
}


class SyntheticSize(StrEnum):
    size_20M = "20M"
    size_275M = "275M"
    size_810M = "810M"
    size_1_4B = "1.4B"


def _source_label(dataset: str) -> str:
    return dataset.replace("_n200000000_v26_a50_m64_z1p2_", "-")


class SyntheticModelConfigurator(HybridSmallSuiteModelConfigurator):
    """Add a tiny model configuration to the hybrid-small-suite ladder models."""

    def configure_rank_microbatch_size(
        self,
        *,
        size_spec: str,
        sequence_length: int,
        device_type: str,
    ) -> int:
        if size_spec != SyntheticSize.size_20M:
            return super().configure_rank_microbatch_size(
                size_spec=size_spec,
                sequence_length=sequence_length,
                device_type=device_type,
            )
        del device_type
        if self.rank_microbatch_size is not None:
            assert self.rank_microbatch_size > 0
            assert self.rank_microbatch_size % sequence_length == 0
            return self.rank_microbatch_size
        return 2 * sequence_length

    def configure_minimal_device_mesh_spec(
        self,
        *,
        size_spec: str,
        sequence_length: int,
        device_type: str,
    ) -> DeviceMeshSpec:
        if size_spec != SyntheticSize.size_20M:
            return super().configure_minimal_device_mesh_spec(
                size_spec=size_spec,
                sequence_length=sequence_length,
                device_type=device_type,
            )
        del sequence_length, device_type
        return DeviceMeshSpec(world_size=1, dp_world_size=None)

    def configure_model(
        self,
        *,
        size_spec: str,
        sequence_length: int,
        tokenizer: TokenizerConfig,
        device_type: str,
    ) -> TransformerConfig:
        if size_spec != SyntheticSize.size_20M:
            return super().configure_model(
                size_spec=size_spec,
                sequence_length=sequence_length,
                tokenizer=tokenizer,
                device_type=device_type,
            )
        if sequence_length != 8192:
            raise OLMoConfigurationError(
                "The synthetic 20M model config currently assumes sequence length 8192."
            )

        d_model = 128
        hidden_size = 1024
        n_layers = 14
        n_heads = 2
        head_dim = 64
        layer_norm = LayerNormConfig(
            name=LayerNormType.rms,
            eps=1e-6,
            bias=False,
            dtype=DType.float32,
        )
        feed_forward = FeedForwardConfig(
            hidden_size=hidden_size,
            bias=False,
            dtype=DType.float32,
            activation=ActivationFunction.silu,
        )
        attention_block = TransformerBlockConfig(
            name=TransformerBlockType.peri_norm,
            sequence_mixer=AttentionConfig(
                name=AttentionType.default,
                n_heads=n_heads,
                n_kv_heads=n_heads,
                head_dim=head_dim,
                bias=False,
                rope=None,
                gate=GateConfig(
                    granularity=GateGranularity.elementwise,
                    full_precision=True,
                ),
                qk_norm=layer_norm,
                use_head_qk_norm=True,
                backend=_attention_backend(device_type),
                dtype=DType.float32,
            ),
            feed_forward=feed_forward,
            layer_norm=layer_norm,
        )

        block_overrides: dict[int, TransformerBlockConfig] | None = None
        if self.model_type == SensitivityModelType.hybrid:
            block = TransformerBlockConfig(
                name=TransformerBlockType.peri_norm,
                sequence_mixer=GatedDeltaNetConfig(
                    n_heads=n_heads,
                    n_v_heads=n_heads,
                    head_dim=head_dim,
                    expand_v=2.0,
                    dtype=DType.float32,
                ),
                feed_forward=feed_forward,
                layer_norm=layer_norm,
            )
            block_overrides = {
                layer_idx: attention_block for layer_idx in range(n_layers) if layer_idx % 5 == 4
            }
        else:
            block = attention_block

        return TransformerConfig(
            d_model=d_model,
            vocab_size=tokenizer.padded_vocab_size(),
            n_layers=n_layers,
            block=block,
            lm_head=LMHeadConfig(
                loss_implementation=LMLossImplementation.default,
                layer_norm=layer_norm,
                bias=False,
                dtype=DType.float32,
            ),
            dtype=DType.float32,
            block_overrides=block_overrides,
            embed_scale=math.sqrt(d_model),
            embedding_norm=LayerNormConfig(
                name=LayerNormType.rms,
                eps=1e-6,
                bias=False,
            ),
            tie_word_embeddings=True,
        )


def _source_paths(args: argparse.Namespace) -> list[str]:
    if args.mixture_source_path:
        return args.mixture_source_path
    relative_path, _ = SYNTHETIC_DATASETS[args.mixture_dataset]
    return [str(join_path(args.mixture_dataset_root, relative_path, "*.npy"))]


def _synthetic_source(
    args: argparse.Namespace, tokenizer: TokenizerConfig
) -> ConcatAndChunkInstanceSourceConfig:
    label = _source_label(args.mixture_dataset)
    return ConcatAndChunkInstanceSourceConfig(
        sources=[
            NumpyDocumentSourceConfig(
                source_paths=_source_paths(args),
                tokenizer=tokenizer,
                expand_glob=True,
                source_group_size=-1,
                label=label,
            )
        ],
        sequence_length=args.sequence_length,
        label=label,
    )


def _model_configurator(args: argparse.Namespace) -> SyntheticModelConfigurator:
    return SyntheticModelConfigurator(
        model_type=str(args.model_type),
        rank_microbatch_size=None
        if args.rank_mbz is None
        else args.rank_mbz * args.sequence_length,
    )


@dataclass(kw_only=True)
class SyntheticLadder(ModelLadder):
    """Ladder recipe for synthetic-only pretraining experiments."""

    model_type: str
    mixture_dataset: str
    mixture_dataset_tokens: int
    training_tokens: int
    chinchilla_multiple: float

    def get_save_folder(self, size_spec: str) -> str:
        return str(
            join_path(
                self.dir,
                size_spec,
                self.model_type,
                self.mixture_dataset,
                f"Cx{_format_chinchilla_multiple(self.chinchilla_multiple)}",
            )
        )

    def _configure_trainer(self, size_spec: str, for_benchmarking: bool = False):
        config = super()._configure_trainer(size_spec, for_benchmarking=for_benchmarking)
        run_name = (
            f"{size_spec}/{self.model_type}/{self.mixture_dataset}/"
            f"Cx{_format_chinchilla_multiple(self.chinchilla_multiple)}"
        )
        if "wandb" in config.callbacks:
            config.callbacks["wandb"].name = run_name  # type: ignore[attr-defined]
            config.callbacks["wandb"].project = self.project or self.name  # type: ignore[attr-defined]
            config.callbacks["wandb"].group = f"{self.name}/{size_spec}/{self.model_type}"  # type: ignore[attr-defined]
            config.callbacks["wandb"].tags = _wandb_tags(  # type: ignore[attr-defined]
                f"size:{size_spec}",
                f"model_type:{self.model_type}",
                f"mixture_dataset:{_source_label(self.mixture_dataset)}",
                "data:synth-only",
                f"chinchilla_multiple:{_format_chinchilla_multiple(self.chinchilla_multiple)}",
                f"mixture_dataset_tokens:{self.mixture_dataset_tokens}",
                f"training_tokens:{self.training_tokens}",
            )
        if "slack_notifier" in config.callbacks:
            config.callbacks["slack_notifier"].name = run_name  # type: ignore[attr-defined]
        return config


def add_args(cmd: str, parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(
        cluster="ai2/jupiter",
        workspace="ai2/linear-rnns",
        budget="ai2/oe-other",
        priority="urgent",
    )
    if cmd == "launch-all":
        parser.set_defaults(_synthetic_launch_all=True)
    parser.add_argument(
        "--model-type",
        choices=list(SensitivityModelType),
        default=SensitivityModelType.transformer,
        help="Model family for this condition.",
    )
    parser.add_argument(
        "--mixture-dataset",
        choices=sorted(SYNTHETIC_DATASETS),
        required=cmd in {"dry-run", "benchmark", "launch-benchmark", "run", "launch", "metrics"},
        help="Synthetic dataset to use for all pretraining tokens.",
    )
    parser.add_argument(
        "--mixture-dataset-root",
        type=str,
        default=SYNTHETIC_DATA_ROOT,
        help="Root directory for the configured synthetic dataset paths.",
    )
    parser.add_argument(
        "--mixture-source-path",
        nargs="*",
        default=None,
        help=(
            "Tokenized synthetic .npy shard path(s) or glob(s). By default the configured "
            "relative dataset path is resolved under --mixture-dataset-root."
        ),
    )


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    if getattr(args, "_synthetic_launch_all", False):
        raise OLMoConfigurationError(
            "This ladder upsamples the synthetic dataset to a size-specific token budget. Use "
            "`launch` with --size for each size instead of `launch-all`."
        )

    tokenizer = TokenizerConfig.dolma2()
    sizes = get_requested_sizes(args)
    size_for_duration = sizes[0]
    if args.mixture_dataset is None:
        args.mixture_dataset = next(iter(SYNTHETIC_DATASETS))

    run_configurator = WSDSChinchillaRunConfigurator(
        chinchilla_multiple=args.chinchilla_multiple,
        lr_multiplier=args.lr_multiplier,
        stepped_schedule=args.stepped_schedule,
    )
    model_configurator = _model_configurator(args)
    model_config = model_configurator.configure_model(
        size_spec=str(size_for_duration),
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        device_type=get_gpu_type(args.cluster),
    )
    draft_ladder = ModelLadder(
        name=args.name,
        project=args.project,
        dir=str(join_path(get_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=sizes,
        max_devices=args.max_gpus,
        device_type=get_gpu_type(args.cluster),
        model_configurator=model_configurator,
        run_configurator=run_configurator,
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        instance_sources=[_synthetic_source(args, tokenizer)],
        data_loader=ComposableDataLoaderConfig(
            num_workers=8, instance_filter_config=InstanceFilterConfig()
        ),
    )
    global_batch_size, *_ = draft_ladder._configure_batch_size_and_num_devices(
        str(size_for_duration), model_config.num_non_embedding_params
    )
    training_tokens = run_configurator.configure_duration(
        model_config.num_non_embedding_params,
        global_batch_size,
    ).value

    source_label = _source_label(args.mixture_dataset)
    _, mixture_dataset_tokens = SYNTHETIC_DATASETS[args.mixture_dataset]
    return SyntheticLadder(
        name=args.name,
        project=args.project,
        dir=str(join_path(get_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=sizes,
        max_devices=args.max_gpus,
        device_type=get_gpu_type(args.cluster),
        model_configurator=model_configurator,
        run_configurator=run_configurator,
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        instance_sources=[
            SamplingInstanceSourceConfig(
                sources=[_synthetic_source(args, tokenizer)],
                max_tokens=training_tokens,
                label=source_label,
            )
        ],
        data_loader=ComposableDataLoaderConfig(
            num_workers=8, instance_filter_config=InstanceFilterConfig()
        ),
        model_type=str(args.model_type),
        mixture_dataset=args.mixture_dataset,
        mixture_dataset_tokens=mixture_dataset_tokens,
        training_tokens=training_tokens,
        chinchilla_multiple=args.chinchilla_multiple,
    )


if __name__ == "__main__":
    main(
        configure_ladder=configure_ladder,
        size_enum=SyntheticSize,
        default_name="synthetic-ladder",
        add_additional_args=add_args,
    )
