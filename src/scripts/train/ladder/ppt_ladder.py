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
    NumpyDocumentSourceConfig,
    NumpyDocumentSourceMixConfig,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.common import get_gpu_type, get_root_dir
from olmo_core.internal.ladder import get_requested_sizes, main
from olmo_core.io import join_path
from olmo_core.model_ladder import (
    ModelLadder,
    Olmo3ModelConfigurator,
    RunConfigurator,
    WSDSChinchillaRunConfigurator,
)
from olmo_core.optim import OptimConfig, Scheduler
from olmo_core.train import Checkpointer, Duration, LoadStrategy, TrainerConfig

log = logging.getLogger(__name__)

PPT_DATA_ROOT = "/weka/oe-training-default/jacksonp/datasets/"
PPT_DATASET_NAME = "olmo-ppt/data/processed"


class PPTStage(StrEnum):
    ppt = "ppt"
    train = "train"


@dataclass(kw_only=True)
class FixedStepsRunConfigurator(RunConfigurator):
    """
    Use the base optimizer and scheduler settings, but cap the run at a fixed number of steps.
    """

    base: RunConfigurator
    steps: int
    checkpoint_name: str

    def __post_init__(self):
        if self.steps <= 0:
            raise OLMoConfigurationError("'steps' must be positive")

    def configure_target_batch_size(self, num_params: int) -> int:
        return self.base.configure_target_batch_size(num_params)

    def configure_duration(self, num_params: int, batch_size: int) -> Duration:
        del num_params, batch_size
        return Duration.steps(self.steps)

    def configure_optimizer(self, num_params: int, batch_size: int) -> OptimConfig:
        return self.base.configure_optimizer(num_params, batch_size)

    def configure_lr_scheduler(self, num_params: int, batch_size: int) -> Scheduler:
        return self.base.configure_lr_scheduler(num_params, batch_size)

    def configure_checkpoint_intervals(
        self, num_params: int, batch_size: int
    ) -> list[tuple[Duration, str]]:
        del num_params, batch_size
        return [(Duration.steps(self.steps), self.checkpoint_name)]

    def plot_lr_schedule(
        self,
        num_params: int,
        batch_size: int,
        *,
        show: bool = True,
        save_path: str | None = None,
    ) -> str | None:
        return self.base.plot_lr_schedule(num_params, batch_size, show=show, save_path=save_path)


@dataclass(kw_only=True)
class PPTLadder(ModelLadder):
    """A two-stage ladder recipe for pre-pretraining followed by normal OLMo training."""

    stage: Literal["ppt", "train"]
    ppt_steps: int

    def get_stage_save_folder(self, stage: Literal["ppt", "train"], size_spec: str) -> str:
        return str(join_path(self.dir, f"{stage}-ppt{self.ppt_steps}", size_spec))

    def get_save_folder(self, size_spec: str) -> str:
        return self.get_stage_save_folder(self.stage, size_spec)

    def get_ppt_checkpoint_path(self, size_spec: str) -> str:
        return str(
            join_path(
                self.get_stage_save_folder("ppt", size_spec),
                Checkpointer.checkpoint_dirname(self.ppt_steps),
            )
        )

    def _configure_trainer(
        self,
        size_spec: str,
        for_benchmarking: bool = False,
    ) -> TrainerConfig:
        config = super()._configure_trainer(size_spec, for_benchmarking=for_benchmarking)
        run_name = f"{self.name}-{self.stage}-ppt{self.ppt_steps}-{size_spec}"

        if self.stage == "train" and self.ppt_steps > 0:
            config.load_path = self.get_ppt_checkpoint_path(size_spec)
            config.load_strategy = LoadStrategy.always
            config.load_trainer_state = False
            config.load_optim_state = False

        if "wandb" in config.callbacks:
            config.callbacks["wandb"].name = run_name  # type: ignore[attr-defined]
            config.callbacks["wandb"].group = self.name  # type: ignore[attr-defined]
        if "slack_notifier" in config.callbacks:
            config.callbacks["slack_notifier"].name = run_name  # type: ignore[attr-defined]

        return config


def add_args(cmd: str, parser: argparse.ArgumentParser) -> None:
    del cmd
    parser.add_argument(
        "--stage",
        choices=list(PPTStage),
        default=PPTStage.train,
        help="Which stage to run: PPT warmup or normal OLMo training initialized from PPT.",
    )
    parser.add_argument(
        "--ppt-steps",
        type=int,
        default=0,
        help="Number of pre-pretraining optimizer steps. Set to 0 for the no-PPT baseline.",
    )
    parser.add_argument(
        "--ppt-source-path",
        nargs="*",
        default=None,
        help=(
            "Tokenized PPT .npy shard path(s) or glob(s). Defaults to "
            f"{PPT_DATA_ROOT}/{PPT_DATASET_NAME}/*.npy."
        ),
    )
    parser.add_argument(
        "--ppt-dataset-root",
        type=str,
        default=PPT_DATA_ROOT,
        help="WEKA directory containing tokenized PPT datasets.",
    )
    parser.add_argument(
        "--ppt-dataset-name",
        type=str,
        default=PPT_DATASET_NAME,
        help="Dataset subdirectory under --ppt-dataset-root to use when --ppt-source-path is unset.",
    )


def _base_run_configurator(args: argparse.Namespace) -> WSDSChinchillaRunConfigurator:
    return WSDSChinchillaRunConfigurator(
        chinchilla_multiple=args.chinchilla_multiple,
        lr_multiplier=args.lr_multiplier,
        stepped_schedule=args.stepped_schedule,
    )


def _olmo_instance_sources(
    tokenizer: TokenizerConfig, sequence_length: int
) -> list[InstanceSourceConfig]:
    return [
        ConcatAndChunkInstanceSourceConfig(
            sources=[
                NumpyDocumentSourceMixConfig(
                    tokenizer=tokenizer,
                    mix=DataMix.OLMo_mix_0925,
                    mix_base_dir="gs://ai2-llm/",
                )
            ],
            sequence_length=sequence_length,
        )
    ]


def _ppt_instance_sources(
    args: argparse.Namespace, tokenizer: TokenizerConfig
) -> list[InstanceSourceConfig]:
    source_paths = args.ppt_source_path or [
        str(join_path(args.ppt_dataset_root, args.ppt_dataset_name, "*.npy"))
    ]

    return [
        ConcatAndChunkInstanceSourceConfig(
            sources=[
                NumpyDocumentSourceConfig(
                    source_paths=list(source_paths),
                    tokenizer=tokenizer,
                    expand_glob=True,
                    label=f"ppt:{args.ppt_dataset_name}",
                )
            ],
            sequence_length=args.sequence_length,
            label="ppt",
        )
    ]


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()
    base_run = _base_run_configurator(args)

    if args.stage == PPTStage.ppt:
        if args.ppt_steps <= 0:
            raise OLMoConfigurationError("--ppt-steps must be positive for --stage=ppt")
        run_configurator: RunConfigurator = FixedStepsRunConfigurator(
            base=base_run,
            steps=args.ppt_steps,
            checkpoint_name=f"PPT final ({args.ppt_steps:,d} steps)",
        )
        instance_sources = _ppt_instance_sources(args, tokenizer)
    else:
        run_configurator = base_run
        instance_sources = _olmo_instance_sources(tokenizer, args.sequence_length)

    return PPTLadder(
        name=args.name,
        project=args.project,
        dir=str(join_path(get_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=get_requested_sizes(args),
        max_devices=args.max_gpus,
        device_type=get_gpu_type(args.cluster),
        model_configurator=Olmo3ModelConfigurator(
            rank_microbatch_size=None
            if args.rank_mbz is None
            else args.rank_mbz * args.sequence_length,
        ),
        run_configurator=run_configurator,
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        instance_sources=instance_sources,
        data_loader=ComposableDataLoaderConfig(
            num_workers=8, instance_filter_config=InstanceFilterConfig()
        ),
        stage=str(args.stage),
        ppt_steps=args.ppt_steps,
    )


if __name__ == "__main__":
    main(configure_ladder=configure_ladder, add_additional_args=add_args)
