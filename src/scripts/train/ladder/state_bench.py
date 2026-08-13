"""Launch synthetic-only ladder experiments on the StateBench training distribution.

StateBench comprises the ``integer-code--r-trivial``, ``integer-code--aperiodic``,
and ``integer-code--periodic`` training splits. Each run selects exactly one split,
which is repeated as necessary to fill the size-specific Chinchilla budget.

Typical usage:

    uv run src/scripts/train/ladder/state_bench.py launch \\
      --size 60M --model-type transformer --distribution r-trivial \\
      --init-seed 0 --max-gpus 8
"""

import argparse
import sys
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
from olmo_core.internal.ladder import _launch_run, configure_launcher, get_requested_sizes, main
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
    "integer-code--r-trivial": 17_176_160_694,
    "integer-code--aperiodic": 17_176_160_694,
    "integer-code--periodic": 15_431_502_989,
}
STATE_BENCH_DISTRIBUTION_ALIASES = {
    "r-trivial": "integer-code--r-trivial",
    "aperiodic": "integer-code--aperiodic",
    "periodic": "integer-code--periodic",
}


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
    distribution: str
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
                self.distribution,
                f"Cx{_format_chinchilla_multiple(self.chinchilla_multiple)}",
                f"init_seed{self.init_seed}",
            )
        )

    def _configure_trainer(self, size_spec: str, for_benchmarking: bool = False):
        config = super()._configure_trainer(size_spec, for_benchmarking=for_benchmarking)
        run_name = (
            f"{size_spec}/{self.model_type}/{self.distribution}/"
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
                f"state_bench_distribution:{self.distribution}",
                f"state_bench_distribution_tokens:{self.state_bench_tokens}",
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
    if cmd == "launch":
        parser.set_defaults(func=launch_state_bench)
    parser.add_argument(
        "--model-type",
        choices=list(SensitivityModelType),
        default=None,
        help="Model family for this condition. Omit with `launch` to launch both families.",
    )
    parser.add_argument(
        "--init-seed",
        type=int,
        default=0,
        help="Random seed used for model parameter initialization.",
    )
    parser.add_argument(
        "--distribution",
        choices=sorted(STATE_BENCH_DISTRIBUTION_ALIASES),
        default=None,
        help="StateBench training distribution. Omit with `launch` to launch all distributions.",
    )
    parser.add_argument(
        "--state-bench-data-root",
        type=str,
        default=STATE_BENCH_DATA_ROOT,
        help="Directory containing the StateBench distribution directories.",
    )


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    if args.model_type is None or args.distribution is None:
        raise OLMoConfigurationError(
            "Specify both --model-type and --distribution for a single ladder configuration. "
            "The `launch` command expands omitted values into the corresponding suite."
        )

    tokenizer = TokenizerConfig.dolma2()
    distribution = STATE_BENCH_DISTRIBUTION_ALIASES[args.distribution]
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
        instance_sources=[_state_bench_source(args, tokenizer, distribution)],
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
        # Sampling repeats the selected distribution as needed to reach the Chinchilla budget.
        instance_sources=[
            SamplingInstanceSourceConfig(
                sources=[_state_bench_source(args, tokenizer, distribution)],
                max_tokens=training_tokens,
                label="state-bench",
            )
        ],
        data_loader=ComposableDataLoaderConfig(
            num_workers=8, instance_filter_config=InstanceFilterConfig()
        ),
        model_type=str(args.model_type),
        distribution=distribution,
        state_bench_tokens=STATE_BENCH_DISTRIBUTION_TOKENS[distribution],
        training_tokens=training_tokens,
        chinchilla_multiple=args.chinchilla_multiple,
        init_seed=args.init_seed,
    )


def _condition_argv(model_type: str, distribution: str) -> list[str]:
    """Return the current command line with one concrete StateBench condition."""
    condition_flags = {"--model-type", "--distribution"}
    resolved_argv = [sys.argv[0]]
    index = 1
    while index < len(sys.argv):
        argument = sys.argv[index]
        if argument in condition_flags:
            index += 2
        elif any(argument.startswith(f"{flag}=") for flag in condition_flags):
            index += 1
        else:
            resolved_argv.append(argument)
            index += 1
    return [
        *resolved_argv,
        "--model-type",
        model_type,
        "--distribution",
        distribution,
    ]


def launch_state_bench(args: argparse.Namespace) -> None:
    """Launch every StateBench condition selected by the optional suite filters."""
    from olmo_core.utils import prepare_cli_environment

    prepare_cli_environment()
    model_types = [args.model_type] if args.model_type is not None else list(SensitivityModelType)
    distributions = (
        [args.distribution]
        if args.distribution is not None
        else list(STATE_BENCH_DISTRIBUTION_ALIASES)
    )
    suite_size = len(model_types) * len(distributions)

    for model_type in model_types:
        for distribution in distributions:
            args.model_type = model_type
            args.distribution = distribution
            ladder = configure_ladder(args)

            # ``configure_launcher`` builds the command from ``sys.argv``. Add the resolved
            # suite condition so the Beaker job's ``run`` command is always concrete.
            original_argv = sys.argv
            sys.argv = _condition_argv(str(model_type), distribution)
            try:
                launcher = configure_launcher(args, ladder, "run")
            finally:
                sys.argv = original_argv
            if suite_size > 1:
                # The standard launcher enables a log-following soft timeout. A suite must
                # submit every condition without following the first job, so disable that
                # follow-only timeout for its individual submissions.
                launcher.step_timeout = None
                launcher.step_soft_timeout = None

            _launch_run(
                ladder,
                launcher,
                args.size_enum(args.size),
                # Following a multi-condition suite would block before later jobs launch.
                follow=args.follow if suite_size == 1 else False,
                slack_notifications=args.slack_notifications,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main(
        configure_ladder=configure_ladder,
        size_enum=SyntheticSize,
        default_name="state-bench",
        add_additional_args=add_args,
    )
