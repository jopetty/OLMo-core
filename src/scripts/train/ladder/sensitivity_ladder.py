"""
Launch ladder experiments that mix a small sensitivity dataset into normal OLMo pretraining.

Each run trains on the standard Dolma/OLMo distribution plus exactly one of the small
sensitivity datasets. The sensitivity dataset is sampled so that one pass through it is
spread over the configured Chinchilla duration. For example, a Cx8 run uses a thinner
mixture than a Cx4 run, but both see the full sensitivity dataset by the end.

Typical usage:

    uv run src/scripts/train/ladder/sensitivity_ladder.py launch \
      --size 190M \
      --model-type transformer \
      --mixture-dataset aperiodic_supervised_n10000_v26_a50_m64_z1p2_s3 \
      --chinchilla-multiple 8 \
      --max-gpus 8 \
      --cluster ai2/jupiter \
      --workspace ai2/linear-rnns \
      --budget ai2/oe-other \
      --priority urgent

The default save layout is:

    /weka/.../sensitivity-ladder/{size}/{model_type}/{dataset}/Cx{chinchilla_multiple}

By default the sensitivity datasets are loaded from:

    /weka/oe-training-default/jacksonp/datasets/sensitivity-data/data/processed/{dataset}/*.npy
"""

import argparse
import logging
from dataclasses import dataclass
from typing import Literal

from olmo_core.config import StrEnum
from olmo_core.data import DataMix, TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    ConcatAndChunkInstanceSourceConfig,
    InstanceFilterConfig,
    InstanceSourceConfig,
    MixingInstanceSourceConfig,
    MixingInstanceSourceSpecConfig,
    NumpyDocumentSourceConfig,
    NumpyDocumentSourceMixConfig,
)
from olmo_core.eval import task_groups
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.common import get_gpu_type, get_root_dir
from olmo_core.internal.ladder import get_requested_sizes, main
from olmo_core.io import join_path
from olmo_core.model_ladder import (
    ModelLadder,
    Olmo3ModelConfigurator,
    WSDSChinchillaRunConfigurator,
)
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerConfig

log = logging.getLogger(__name__)

SENSITIVITY_DATA_ROOT = (
    "/weka/oe-training-default/jacksonp/datasets/sensitivity-data/data/processed"
)

OLMO_EVAL_FORK_POST_SETUP = (
    "pip install --force-reinstall --no-deps "
    "'git+https://github.com/jopetty/olmo-eval.git@formal-langs'"
)

DEFAULT_EXTRA_EVAL_TASKS = [
    "formal_langs_cube_unique:v6",
    "formal_langs_cube_reassign_const:v6",
    "formal_langs_cube_reassign_var:v6",
]

SENSITIVITY_DATASETS = (
    "aperiodic_supervised_n10000_v26_a50_m64_z1p2_s3",
    "r-trivial_unsupervised_n10000_v26_a50_m64_z1p2_s0",
    "aperiodic_unsupervised_n10000_v26_a50_m64_z1p2_s2",
    "periodic_unsupervised_n10000_v26_a50_m64_z1p2_s4",
    "periodic_supervised_n10000_v26_a50_m64_z1p2_s5",
    "r-trivial_supervised_n10000_v26_a50_m64_z1p2_s1",
)

SENSITIVITY_DATASET_TOKENS: dict[str, int] = {
    "r-trivial_unsupervised_n10000_v26_a50_m64_z1p2_s0": 355_460,
    "r-trivial_supervised_n10000_v26_a50_m64_z1p2_s1": 711_005,
    "aperiodic_unsupervised_n10000_v26_a50_m64_z1p2_s2": 635_190,
    "aperiodic_supervised_n10000_v26_a50_m64_z1p2_s3": 1_988_130,
    "periodic_unsupervised_n10000_v26_a50_m64_z1p2_s4": 606_842,
    "periodic_supervised_n10000_v26_a50_m64_z1p2_s5": 1_955_320,
}


class SensitivityModelType(StrEnum):
    transformer = "transformer"
    hybrid = "hybrid"


def _format_chinchilla_multiple(chinchilla_multiple: float) -> str:
    return f"{chinchilla_multiple:g}"


def _source_label(dataset: str) -> str:
    return dataset.replace("_n10000_v26_a50_m64_z1p2_", "-")


def _source_paths(args: argparse.Namespace) -> list[str]:
    return args.mixture_source_path or [
        str(join_path(args.mixture_dataset_root, args.mixture_dataset, "*.npy"))
    ]


def _olmo_source(
    tokenizer: TokenizerConfig, sequence_length: int, mix_base_dir: str
) -> InstanceSourceConfig:
    return ConcatAndChunkInstanceSourceConfig(
        sources=[
            NumpyDocumentSourceMixConfig(
                tokenizer=tokenizer,
                mix=DataMix.OLMo_mix_0925,
                mix_base_dir=mix_base_dir,
            )
        ],
        sequence_length=sequence_length,
        label="olmo-mix-0925",
    )


def _sensitivity_source(
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


def _get_sensitivity_tokens(args: argparse.Namespace, tokenizer: TokenizerConfig) -> int:
    if args.mixture_dataset_tokens is not None:
        raw_tokens = args.mixture_dataset_tokens
    elif args.mixture_dataset in SENSITIVITY_DATASET_TOKENS:
        raw_tokens = SENSITIVITY_DATASET_TOKENS[args.mixture_dataset]
    else:
        try:
            raw_tokens = NumpyDocumentSourceConfig(
                source_paths=_source_paths(args),
                tokenizer=tokenizer,
                expand_glob=True,
                source_group_size=-1,
            ).get_num_tokens()
        except Exception as exc:
            raise OLMoConfigurationError(
                "Could not determine the sensitivity dataset token count. "
                "If this environment cannot stat the WEKA paths, add the dataset to "
                "SENSITIVITY_DATASET_TOKENS or pass --mixture-dataset-tokens explicitly."
            ) from exc

    usable_tokens = (raw_tokens // args.sequence_length) * args.sequence_length
    if usable_tokens <= 0:
        raise OLMoConfigurationError(
            f"Sensitivity dataset has only {raw_tokens:,d} token(s), which is less than one "
            f"sequence of length {args.sequence_length:,d}."
        )
    if usable_tokens != raw_tokens:
        log.warning(
            "Sensitivity dataset has %d raw tokens; using %d tokens to align to sequence length %d.",
            raw_tokens,
            usable_tokens,
            args.sequence_length,
        )
    return usable_tokens


@dataclass(kw_only=True)
class HybridOlmo3ModelConfigurator(Olmo3ModelConfigurator):
    """Configure OLMo3-sized hybrid models with a repeating GDN/GDN/GDN/attention pattern."""

    def configure_model(
        self,
        *,
        size_spec: str,
        sequence_length: int,
        tokenizer: TokenizerConfig,
        device_type: str,
    ) -> TransformerConfig:
        config = super().configure_model(
            size_spec=size_spec,
            sequence_length=sequence_length,
            tokenizer=tokenizer,
            device_type=device_type,
        )
        if config.n_layers % 4 != 0:
            raise OLMoConfigurationError(
                f"Hybrid model requires a layer count divisible by 4, got {config.n_layers} "
                f"layers for size {size_spec}."
            )
        if not isinstance(config.block, TransformerBlockConfig):
            raise OLMoConfigurationError("Hybrid conversion expected a single transformer block.")
        if not isinstance(config.block.sequence_mixer, AttentionConfig):
            raise OLMoConfigurationError("Hybrid conversion expected an attention sequence mixer.")

        attn_block = config.block
        attn = attn_block.sequence_mixer
        gdn_block = attn_block.replace(
            sequence_mixer=GatedDeltaNetConfig(
                n_heads=attn.n_heads,
                head_dim=max(1, int(0.75 * config.d_model / attn.n_heads)),
                allow_neg_eigval=True,
            )
        )
        config.block = {"gdn": gdn_block, "attn": attn_block}
        config.block_pattern = ["gdn", "gdn", "gdn", "attn"]
        return config


@dataclass(kw_only=True)
class SensitivityLadder(ModelLadder):
    """Ladder recipe for sensitivity-data mixture experiments."""

    model_type: Literal["transformer", "hybrid"]
    mixture_dataset: str
    sensitivity_tokens: int
    training_tokens: int
    chinchilla_multiple: float
    extra_eval_tasks: list[str]

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
            config.callbacks["wandb"].tags = [  # type: ignore[attr-defined]
                f"size:{size_spec}",
                f"model_type:{self.model_type}",
                f"mixture_dataset:{self.mixture_dataset}",
                f"chinchilla_multiple:{_format_chinchilla_multiple(self.chinchilla_multiple)}",
                f"sensitivity_tokens:{self.sensitivity_tokens}",
                f"training_tokens:{self.training_tokens}",
            ]
        if "slack_notifier" in config.callbacks:
            config.callbacks["slack_notifier"].name = run_name  # type: ignore[attr-defined]
        return config

    def _get_in_loop_eval_tasks(self) -> list[str]:
        return sorted(set(super()._get_in_loop_eval_tasks() + self.extra_eval_tasks))


def add_args(cmd: str, parser: argparse.ArgumentParser) -> None:
    if cmd == "launch-all":
        parser.set_defaults(_sensitivity_launch_all=True)
    if "launch" in cmd:
        parser.set_defaults(post_setup=OLMO_EVAL_FORK_POST_SETUP)
    parser.add_argument(
        "--model-type",
        choices=list(SensitivityModelType),
        default=SensitivityModelType.transformer,
        help="Model family for this condition.",
    )
    parser.add_argument(
        "--mixture-dataset",
        choices=SENSITIVITY_DATASETS,
        required=cmd in {"dry-run", "benchmark", "launch-benchmark", "run", "launch", "metrics"},
        help="Sensitivity dataset to mix into OLMo pretraining.",
    )
    parser.add_argument(
        "--mixture-dataset-root",
        type=str,
        default=SENSITIVITY_DATA_ROOT,
        help="Directory containing one subdirectory per sensitivity dataset.",
    )
    parser.add_argument(
        "--mixture-source-path",
        nargs="*",
        default=None,
        help=(
            "Tokenized sensitivity .npy shard path(s) or glob(s). Defaults to "
            "--mixture-dataset-root/--mixture-dataset/*.npy."
        ),
    )
    parser.add_argument(
        "--mixture-dataset-tokens",
        type=int,
        default=None,
        help=(
            "Optional raw token count for the sensitivity dataset. When unset, the script "
            "stats the configured .npy files and aligns the count down to a sequence boundary."
        ),
    )
    parser.add_argument(
        "--mix-base-dir",
        type=str,
        default="gs://ai2-llm/",
        help="Base directory for the standard OLMo data mix.",
    )
    parser.add_argument(
        "--extra-eval-task",
        action="append",
        nargs="+",
        default=[],
        help=(
            "Additional downstream eval task name(s) to run at the default in-loop eval steps. "
            "May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--extra-eval-task-group",
        action="append",
        choices=sorted(task_groups.TASK_GROUPS),
        default=[],
        help=(
            "Additional downstream eval task group(s) to run at the default in-loop eval steps. "
            "May be passed multiple times."
        ),
    )


def _model_configurator(args: argparse.Namespace) -> Olmo3ModelConfigurator:
    kwargs = dict(
        rank_microbatch_size=None if args.rank_mbz is None else args.rank_mbz * args.sequence_length
    )
    if args.model_type == SensitivityModelType.hybrid:
        return HybridOlmo3ModelConfigurator(**kwargs)
    return Olmo3ModelConfigurator(**kwargs)


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    if getattr(args, "_sensitivity_launch_all", False):
        raise OLMoConfigurationError(
            "This ladder computes a size-specific mixture density. Use `launch` with --size for "
            "each size instead of `launch-all`."
        )

    tokenizer = TokenizerConfig.dolma2()
    sizes = get_requested_sizes(args)
    size_for_density = sizes[0]
    if args.mixture_dataset is None:
        args.mixture_dataset = SENSITIVITY_DATASETS[0]

    run_configurator = WSDSChinchillaRunConfigurator(
        chinchilla_multiple=args.chinchilla_multiple,
        lr_multiplier=args.lr_multiplier,
        stepped_schedule=args.stepped_schedule,
    )
    model_configurator = _model_configurator(args)
    model_config = model_configurator.configure_model(
        size_spec=str(size_for_density),
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
        instance_sources=[
            _olmo_source(tokenizer, args.sequence_length, args.mix_base_dir),
        ],
        data_loader=ComposableDataLoaderConfig(
            num_workers=8, instance_filter_config=InstanceFilterConfig()
        ),
    )
    global_batch_size, *_ = draft_ladder._configure_batch_size_and_num_devices(
        str(size_for_density), model_config.num_non_embedding_params
    )
    training_tokens = run_configurator.configure_duration(
        model_config.num_non_embedding_params,
        global_batch_size,
    ).value
    sensitivity_tokens = _get_sensitivity_tokens(args, tokenizer)
    if sensitivity_tokens >= training_tokens:
        raise OLMoConfigurationError(
            f"Sensitivity dataset token count ({sensitivity_tokens:,d}) must be smaller than "
            f"the training token budget ({training_tokens:,d})."
        )

    sensitivity_ratio = sensitivity_tokens / training_tokens
    extra_eval_tasks = list(DEFAULT_EXTRA_EVAL_TASKS)
    extra_eval_tasks.extend(task for group in args.extra_eval_task for task in group)
    for group_name in args.extra_eval_task_group:
        extra_eval_tasks.extend(task_groups.TASK_GROUPS[group_name])

    instance_sources: list[InstanceSourceConfig] = [
        MixingInstanceSourceConfig(
            source_specs=[
                MixingInstanceSourceSpecConfig(
                    source=_olmo_source(tokenizer, args.sequence_length, args.mix_base_dir),
                    ratio=1.0 - sensitivity_ratio,
                    label="olmo-mix-0925",
                ),
                MixingInstanceSourceSpecConfig(
                    source=_sensitivity_source(args, tokenizer),
                    ratio=sensitivity_ratio,
                    label=_source_label(args.mixture_dataset),
                ),
            ],
            num_tokens=training_tokens,
            label=f"olmo-plus-{_source_label(args.mixture_dataset)}",
        )
    ]

    return SensitivityLadder(
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
        instance_sources=instance_sources,
        data_loader=ComposableDataLoaderConfig(
            num_workers=8, instance_filter_config=InstanceFilterConfig()
        ),
        model_type=str(args.model_type),
        mixture_dataset=args.mixture_dataset,
        sensitivity_tokens=sensitivity_tokens,
        training_tokens=training_tokens,
        chinchilla_multiple=args.chinchilla_multiple,
        extra_eval_tasks=extra_eval_tasks,
    )


if __name__ == "__main__":
    main(
        configure_ladder=configure_ladder,
        default_name="sensitivity-ladder",
        add_additional_args=add_args,
    )
