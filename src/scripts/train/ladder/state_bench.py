"""Launch synthetic-only ladder experiments on the StateBench training distribution.

StateBench comprises the ``integer-code--r-trivial``, ``integer-code--aperiodic``,
and ``integer-code--periodic`` training splits. The splits are sampled in
proportion to their tokenized corpus sizes and repeated as necessary to fill the
size-specific Chinchilla budget.

Typical usage:

    uv run src/scripts/train/ladder/state_bench.py launch \\
      --size 60M --model-type transformer --init-seed 0 --max-gpus 8
"""

import argparse
from dataclasses import dataclass

from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    ConcatAndChunkInstanceSourceConfig,
    InstanceFilterConfig,
    NumpyDocumentSourceConfig,
    SamplingInstanceSourceConfig,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.common import get_gpu_type
from olmo_core.internal.ladder import get_requested_sizes, main
from olmo_core.io import join_path
from olmo_core.model_ladder import ModelLadder, WSDSChinchillaRunConfigurator

from sensitivity_ladder import _format_chinchilla_multiple, _wandb_tags
from synthetic_ladder import (
    SYNTHETIC_LADDER_ROOT,
    SensitivityModelType,
    SyntheticModelConfigurator,
    SyntheticSize,
)


STATE_BENCH_DATA_ROOT = (
    "/weka/oe-training-default/jacksonp/state-bench/data/"
    "state-tracking-long-context-v1/rendered-tokenized/tokens"
)
STATE_BENCH_DISTRIBUTION_TOKENS = {
    "integer-code--r-trivial": 3_003_000_000,
    "integer-code--aperiodic": 3_003_000_000,
    "integer-code--periodic": 2_956_800_000,
}
STATE_BENCH_DISTRIBUTIONS = tuple(STATE_BENCH_DISTRIBUTION_TOKENS)
STATE_BENCH_TOTAL_TOKENS = sum(STATE_BENCH_DISTRIBUTION_TOKENS.values())


def _root_dir(cluster: str) -> str:
    if cluster.startswith("ai2/"):
        return SYNTHETIC_LADDER_ROOT
    return "gs://ai2-llm"


def _source_paths(args: argparse.Namespace, distribution: str) -> list[str]:
    """Return the token shard glob for one StateBench training distribution."""
    return [str(join_path(args.state_bench_data_root, distribution, "train", "*.npy"))]


def _state_bench_source(
    args: argparse.Namespace, tokenizer: TokenizerConfig, distribution: str
) -> ConcatAndChunkInstanceSourceConfig:
    """Configure one independently sampled StateBench training distribution."""
    return ConcatAndChunkInstanceSourceConfig(
        sources=[
            NumpyDocumentSourceConfig(
                source_paths=_source_paths(args, distribution),
                tokenizer=tokenizer,
                expand_glob=True,
                source_group_size=-1,
                label=distribution,
            )
        ],
        sequence_length=args.sequence_length,
        label=distribution,
    )


def _model_configurator(args: argparse.Namespace) -> SyntheticModelConfigurator:
    return SyntheticModelConfigurator(
        model_type=str(args.model_type),
        init_seed=args.init_seed,
        rank_microbatch_size=None
        if args.rank_mbz is None
        else args.rank_mbz * args.sequence_length,
    )


@dataclass(kw_only=True)
class StateBenchLadder(ModelLadder):
    """Ladder recipe for synthetic-only StateBench pretraining experiments."""

    model_type: str
    state_bench_tokens: int
    training_tokens: int
    chinchilla_multiple: float
    init_seed: int

    def get_save_folder(self, size_spec: str) -> str:
        return str(
            join_path(
                self.dir,
                size_spec,
                self.model_type,
                "state-bench",
                f"Cx{_format_chinchilla_multiple(self.chinchilla_multiple)}",
                f"init_seed{self.init_seed}",
            )
        )

    def _configure_trainer(self, size_spec: str, for_benchmarking: bool = False):
        config = super()._configure_trainer(size_spec, for_benchmarking=for_benchmarking)
        run_name = (
            f"{size_spec}/{self.model_type}/state-bench/"
            f"Cx{_format_chinchilla_multiple(self.chinchilla_multiple)}/"
            f"init_seed{self.init_seed}"
        )
        if "wandb" in config.callbacks:
            config.callbacks["wandb"].name = run_name  # type: ignore[attr-defined]
            config.callbacks["wandb"].project = self.project or self.name  # type: ignore[attr-defined]
            config.callbacks["wandb"].group = f"{self.name}/{size_spec}/{self.model_type}"  # type: ignore[attr-defined]
            config.callbacks["wandb"].tags = _wandb_tags(  # type: ignore[attr-defined]
                f"size:{size_spec}",
                f"model_type:{self.model_type}",
                "data:state-bench-only",
                "state_bench_distribution:proportional",
                *(f"state_bench_source:{source}" for source in STATE_BENCH_DISTRIBUTIONS),
                *(
                    f"state_bench_source_tokens:{source}:{tokens}"
                    for source, tokens in STATE_BENCH_DISTRIBUTION_TOKENS.items()
                ),
                f"state_bench_total_tokens:{self.state_bench_tokens}",
                f"chinchilla_multiple:{_format_chinchilla_multiple(self.chinchilla_multiple)}",
                f"init_seed:{self.init_seed}",
                f"training_tokens:{self.training_tokens}",
            )
        if "slack_notifier" in config.callbacks:
            config.callbacks["slack_notifier"].name = run_name  # type: ignore[attr-defined]
        return config


def add_args(cmd: str, parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(
        cluster="ai2/jupiter",
        workspace="ai2/beyond-state",
        budget="ai2/oe-other",
        priority="urgent",
        chinchilla_multiple=1.0,
        init_seed=0,
    )
    if cmd == "launch-all":
        parser.set_defaults(_state_bench_launch_all=True)
    parser.add_argument(
        "--model-type",
        choices=list(SensitivityModelType),
        default=SensitivityModelType.transformer,
        help="Model family for this condition.",
    )
    parser.add_argument(
        "--init-seed",
        type=int,
        default=0,
        help="Random seed used for model parameter initialization.",
    )
    parser.add_argument(
        "--state-bench-data-root",
        type=str,
        default=STATE_BENCH_DATA_ROOT,
        help="Directory containing the StateBench distribution directories.",
    )


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    if getattr(args, "_state_bench_launch_all", False):
        raise OLMoConfigurationError(
            "This ladder upsamples StateBench to a size-specific token budget. Use `launch` "
            "with --size for each size instead of `launch-all`."
        )

    tokenizer = TokenizerConfig.dolma2()
    sizes = get_requested_sizes(args)
    size_for_duration = sizes[0]
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
        dir=str(join_path(_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=sizes,
        max_devices=args.max_gpus,
        device_type=get_gpu_type(args.cluster),
        model_configurator=model_configurator,
        run_configurator=run_configurator,
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        instance_sources=[
            _state_bench_source(args, tokenizer, distribution)
            for distribution in STATE_BENCH_DISTRIBUTIONS
        ],
        data_loader=ComposableDataLoaderConfig(
            num_workers=8, instance_filter_config=InstanceFilterConfig()
        ),
    )
    global_batch_size, *_ = draft_ladder._configure_batch_size_and_num_devices(
        str(size_for_duration), model_config.num_non_embedding_params
    )
    training_tokens = run_configurator.configure_duration(
        model_config.num_non_embedding_params, global_batch_size
    ).value

    return StateBenchLadder(
        name=args.name,
        project=args.project,
        dir=str(join_path(_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=sizes,
        max_devices=args.max_gpus,
        device_type=get_gpu_type(args.cluster),
        model_configurator=model_configurator,
        run_configurator=run_configurator,
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        # Sampling preserves the three source sizes' relative proportions, then repeats the
        # distribution as needed to reach the Chinchilla training budget.
        instance_sources=[
            SamplingInstanceSourceConfig(
                sources=[
                    _state_bench_source(args, tokenizer, distribution)
                    for distribution in STATE_BENCH_DISTRIBUTIONS
                ],
                max_tokens=training_tokens,
                label="state-bench",
            )
        ],
        data_loader=ComposableDataLoaderConfig(
            num_workers=8, instance_filter_config=InstanceFilterConfig()
        ),
        model_type=str(args.model_type),
        state_bench_tokens=STATE_BENCH_TOTAL_TOKENS,
        training_tokens=training_tokens,
        chinchilla_multiple=args.chinchilla_multiple,
        init_seed=args.init_seed,
    )


if __name__ == "__main__":
    main(
        configure_ladder=configure_ladder,
        size_enum=SyntheticSize,
        default_name="state-bench",
        add_additional_args=add_args,
    )
