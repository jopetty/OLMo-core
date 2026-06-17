"""
Launch a two-stage pre-pretraining (PPT) ladder experiment.

Typical usage keeps all variants under one shared ladder root named ``ppt-olmo``:

    uv run src/scripts/train/ladder/ppt_ladder.py launch \
      --size 60M \
      --max-gpus 8 \
      --stage ppt \
      --ppt-steps 500 \
      --cluster ai2/jupiter \
      --workspace ai2/linear-rnns \
      --budget ai2/oe-other \
      --priority urgent

Then launch normal OLMo pretraining initialized from that PPT checkpoint:

    uv run src/scripts/train/ladder/ppt_ladder.py launch \
      --size 60M \
      --max-gpus 8 \
      --stage train \
      --ppt-steps 500 \
      --chinchilla-multiple 8 \
      --cluster ai2/jupiter \
      --workspace ai2/linear-rnns \
      --budget ai2/oe-other \
      --priority urgent

For the 0-PPT baseline, use ``--stage train --ppt-steps 0``. This trains from scratch because
the train stage only loads a PPT checkpoint when ``ppt_steps > 0``.

The default save layout is:

    /weka/oe-training-default/ai2-llm/model-ladders/ppt-olmo/{size}/ppt-{ppt_steps}
    /weka/oe-training-default/ai2-llm/model-ladders/ppt-olmo/{size}/train-{ppt_steps}ppt-Cx{chinchilla_multiple}

W&B runs default to the shared project ``ppt-olmo``. Train runs for a given size share the
group ``ppt-olmo/{size}/train``, so variants such as ``train-0ppt-Cx8`` and
``train-500ppt-Cx8`` are easy to compare.

Use ``--reinit-ppt-embeddings`` on a train stage to load the PPT checkpoint but randomize
the input embedding table before natural-language training starts, e.g.
``--stage train --ppt-steps 500 --reinit-ppt-embeddings``.
Add ``--reinit-ppt-lm-head`` to also randomize the LM head.

The PPT stage reads tokenized numpy shards from
``/weka/oe-training-default/jacksonp/datasets/olmo-ppt/data/processed/*.npy`` by default.
Use ``--ppt-dataset-root`` / ``--ppt-dataset-name`` or ``--ppt-source-path`` for variants.
"""

import argparse
import dataclasses
import logging
from dataclasses import dataclass
from typing import Literal

import torch

from olmo_core.aliases import PathOrStr
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
from olmo_core.io import join_path, normalize_path
from olmo_core.model_ladder import (
    ModelLadder,
    Olmo3ModelConfigurator,
    RunConfigurator,
    WSDSChinchillaRunConfigurator,
)
from olmo_core.optim import CosWithWarmup, OptimConfig, Scheduler
from olmo_core.train import Checkpointer, Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import Callback

log = logging.getLogger(__name__)

PPT_DATA_ROOT = "/weka/oe-training-default/jacksonp/datasets/"
PPT_DATASET_NAME = "olmo-ppt/data/processed"
DEFAULT_CHINCHILLA_MULTIPLE = 8.0


def _format_chinchilla_multiple(chinchilla_multiple: float) -> str:
    return f"{chinchilla_multiple:g}"


def _format_ppt_lr(ppt_lr: float | None, ppt_lr_multiplier: float) -> str:
    if ppt_lr is not None:
        return f"lr{ppt_lr:g}"
    elif ppt_lr_multiplier != 1.0:
        return f"lrx{ppt_lr_multiplier:g}"
    return ""


class PPTStage(StrEnum):
    ppt = "ppt"
    train = "train"


@dataclass
class ReinitEmbeddingsAfterPPTLoadCallback(Callback):
    """Reinitialize input embeddings, and optionally the LM head, after loading PPT weights."""

    ppt_checkpoint_path: str
    seed: int
    reinit_lm_head: bool = False
    enabled: bool = True
    _has_reinitialized: bool = False

    @torch.no_grad()
    def post_checkpoint_loaded(self, path: PathOrStr):
        if not self.enabled or self._has_reinitialized:
            return
        if normalize_path(path) != normalize_path(self.ppt_checkpoint_path):
            return

        model = self.trainer.train_module.model
        if model.embeddings is None:
            raise OLMoConfigurationError("Cannot reinitialize embeddings: model has no embeddings.")

        generator = torch.Generator(device=model.embeddings.weight.device).manual_seed(self.seed)
        model.init_method.init_embeddings(
            model.embeddings,
            d_model=model.d_model,
            embed_scale=model.embed_scale,
            std=model.embedding_init_std
            if model.embedding_init_std is not None
            else model.init_std,
            generator=generator,
        )
        reinitialized_lm_head = False
        if self.reinit_lm_head and model.tie_word_embeddings:
            log.warning(
                "Model has tied word embeddings; reinitializing input embeddings already "
                "reinitialized the LM head weights."
            )
            reinitialized_lm_head = True
        elif self.reinit_lm_head:
            if model.lm_head is None:
                raise OLMoConfigurationError("Cannot reinitialize LM head: model has no LM head.")
            model.init_method.init_final_w_out(
                model.lm_head.w_out,
                d_model=model.d_model,
                std=model.init_std,
                generator=generator,
            )
            reinitialized_lm_head = True
        self._has_reinitialized = True
        log.info(
            "Reinitialized %s after loading PPT checkpoint '%s' with seed %d.",
            "input embeddings and LM head" if reinitialized_lm_head else "input embeddings",
            path,
            self.seed,
        )


@dataclass(kw_only=True)
class FixedStepsRunConfigurator(RunConfigurator):
    """
    Use the base optimizer and scheduler settings, but cap the run at a fixed number of steps.

    Optionally override the Chinchilla-derived batch size and learning rate for the PPT stage.
    The reference ppt2 codebase (michahu/ppt2) uses a much smaller batch (~65K tokens/step vs
    ~524K for a 190M model) and a higher LR (~4e-3 for 190M) than the Chinchilla defaults.
    """

    base: RunConfigurator
    steps: int
    checkpoint_name: str
    ppt_batch_size: int = 65536
    """
    Global batch size in tokens for the PPT stage. Defaults to ``32 * 2048 = 65,536``, matching
    the reference ppt2 codebase and giving 8× more gradient steps per token than the
    Chinchilla-derived batch (~524K for a 190M model).
    """
    ppt_lr: float | None = None
    """
    If set, overrides the peak learning rate for the PPT stage with an absolute value.
    When ``None`` (default), ``ppt_lr_multiplier`` is applied to the Chinchilla-derived LR.
    """
    ppt_lr_multiplier: float = 2.0
    """
    Multiplier applied to the Chinchilla-derived LR for the PPT stage when ``ppt_lr`` is not set.
    Defaults to 2.0 to undo the ``/2`` divisor in the WSDS configurator, matching the reference
    ppt2 codebase (e.g. ~4e-3 for 190M, ~2.3e-3 for 1B).
    """
    warmup_steps: int = 1000
    """Warmup steps for the PPT cosine schedule. Matches the reference ppt2 codebase."""
    checkpoint_every: int = 250
    """Save a checkpoint every this many PPT steps. Matches the reference ppt2 codebase."""

    def __post_init__(self):
        if self.steps <= 0:
            raise OLMoConfigurationError("'steps' must be positive")

    def configure_target_batch_size(self, num_params: int) -> int:
        return self.ppt_batch_size

    def configure_duration(self, num_params: int, batch_size: int) -> Duration:
        del num_params, batch_size
        return Duration.steps(self.steps)

    def configure_optimizer(self, num_params: int, batch_size: int) -> OptimConfig:
        config = self.base.configure_optimizer(num_params, batch_size)
        lr = self.ppt_lr if self.ppt_lr is not None else config.lr * self.ppt_lr_multiplier
        # Match ppt2 phase0: weight_decay=0.033, betas=(0.9, 0.95) regardless of batch size.
        return dataclasses.replace(config, lr=lr, weight_decay=0.033, betas=(0.9, 0.95))

    def configure_lr_scheduler(self, num_params: int, batch_size: int) -> Scheduler:
        del num_params, batch_size
        return CosWithWarmup(warmup=self.warmup_steps, alpha_f=0.1)

    def configure_checkpoint_intervals(
        self, num_params: int, batch_size: int
    ) -> list[tuple[Duration, str]]:
        del num_params, batch_size
        intervals: list[tuple[Duration, str]] = []
        step = self.checkpoint_every
        while step < self.steps:
            intervals.append((Duration.steps(step), f"step {step:,d}"))
            step += self.checkpoint_every
        intervals.append((Duration.steps(self.steps), self.checkpoint_name))
        return intervals

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
    chinchilla_multiple: float
    reinit_ppt_embeddings: bool
    reinit_ppt_lm_head: bool
    embedding_reinit_seed: int
    ppt_batch_size: int
    ppt_lr: float | None
    ppt_lr_multiplier: float
    warmup_steps: int

    def get_stage_dirname(self, stage: Literal["ppt", "train"]) -> str:
        ppt_lr_str = _format_ppt_lr(self.ppt_lr, self.ppt_lr_multiplier)
        warmup_str = f"-wu{self.warmup_steps}" if self.warmup_steps != 1000 else ""
        if stage == "ppt":
            return (
                f"ppt-{self.ppt_steps}"
                f"-bs{self.ppt_batch_size}"
                f"{'-' + ppt_lr_str if ppt_lr_str else ''}"
                f"{warmup_str}"
            )
        else:
            ppt_config = (
                f"-bs{self.ppt_batch_size}"
                f"{'-' + ppt_lr_str if ppt_lr_str else ''}"
                f"{warmup_str}"
            ) if self.ppt_steps > 0 else ""
            return (
                f"train-{self.ppt_steps}ppt"
                f"{ppt_config}"
                f"-Cx{_format_chinchilla_multiple(self.chinchilla_multiple)}"
                f"{'-reinit-emb' if self.reinit_ppt_embeddings else ''}"
                f"{'-lm-head' if self.reinit_ppt_lm_head else ''}"
            )

    def get_stage_save_folder(self, stage: Literal["ppt", "train"], size_spec: str) -> str:
        stage_dir = self.get_stage_dirname(stage)
        return str(join_path(self.dir, size_spec, stage_dir))

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
        stage_dir = self.get_stage_dirname(self.stage)
        run_name = f"{size_spec}/{stage_dir}"
        wandb_group = f"{self.name}/{size_spec}/{self.stage}"

        if self.stage == "train" and self.ppt_steps > 0:
            ppt_checkpoint_path = self.get_ppt_checkpoint_path(size_spec)
            config.load_path = ppt_checkpoint_path
            config.load_strategy = LoadStrategy.always
            config.load_trainer_state = False
            config.load_optim_state = False
            if self.reinit_ppt_embeddings:
                config.callbacks["reinit_embeddings_after_ppt_load"] = (
                    ReinitEmbeddingsAfterPPTLoadCallback(
                        ppt_checkpoint_path=ppt_checkpoint_path,
                        seed=self.embedding_reinit_seed,
                        reinit_lm_head=self.reinit_ppt_lm_head,
                    )
                )

        if "wandb" in config.callbacks:
            config.callbacks["wandb"].name = run_name  # type: ignore[attr-defined]
            config.callbacks["wandb"].project = self.project or self.name  # type: ignore[attr-defined]
            config.callbacks["wandb"].group = wandb_group  # type: ignore[attr-defined]
            config.callbacks["wandb"].tags = [  # type: ignore[attr-defined]
                f"stage:{self.stage}",
                f"size:{size_spec}",
                f"ppt_steps:{self.ppt_steps}",
                f"chinchilla_multiple:{_format_chinchilla_multiple(self.chinchilla_multiple)}",
                f"reinit_ppt_embeddings:{self.reinit_ppt_embeddings}",
                f"reinit_ppt_lm_head:{self.reinit_ppt_lm_head}",
            ]
        if "slack_notifier" in config.callbacks:
            config.callbacks["slack_notifier"].name = run_name  # type: ignore[attr-defined]

        return config


def add_args(cmd: str, parser: argparse.ArgumentParser) -> None:
    del cmd
    parser.set_defaults(chinchilla_multiple=DEFAULT_CHINCHILLA_MULTIPLE)
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
    parser.add_argument(
        "--reinit-ppt-embeddings",
        action="store_true",
        default=False,
        help=(
            "For train stages loaded from a PPT checkpoint, reinitialize the input embedding "
            "table after loading checkpoint weights."
        ),
    )
    parser.add_argument(
        "--reinit-ppt-lm-head",
        action="store_true",
        default=False,
        help=(
            "Only valid with --reinit-ppt-embeddings. Also reinitialize the LM head after "
            "loading PPT weights."
        ),
    )
    parser.add_argument(
        "--embedding-reinit-seed",
        type=int,
        default=12536,
        help="Seed for --reinit-ppt-embeddings.",
    )
    parser.add_argument(
        "--ppt-batch-size",
        type=int,
        default=65536,
        help=(
            "Global batch size in tokens for the PPT stage. "
            "Default: 65536 (32 * 2048), matching the reference ppt2 codebase."
        ),
    )
    parser.add_argument(
        "--ppt-lr",
        type=float,
        default=None,
        help=(
            "Absolute peak learning rate for the PPT stage. When unset, "
            "--ppt-lr-multiplier is applied to the Chinchilla-derived LR instead."
        ),
    )
    parser.add_argument(
        "--ppt-lr-multiplier",
        type=float,
        default=2.0,
        help=(
            "Multiplier applied to the Chinchilla-derived LR for the PPT stage "
            "when --ppt-lr is not set. Default: 2.0, which undoes the /2 divisor "
            "in the WSDS configurator to match the reference ppt2 codebase."
        ),
    )
    parser.add_argument(
        "--ppt-warmup-steps",
        type=int,
        default=1000,
        help="Warmup steps for the PPT cosine LR schedule. Default: 1000, matching ppt2.",
    )
    parser.add_argument(
        "--ppt-checkpoint-every",
        type=int,
        default=250,
        help="Save a checkpoint every this many PPT steps. Default: 250, matching ppt2.",
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

    if args.reinit_ppt_embeddings and (args.stage != PPTStage.train or args.ppt_steps <= 0):
        raise OLMoConfigurationError(
            "--reinit-ppt-embeddings requires --stage=train and --ppt-steps > 0"
        )
    if args.reinit_ppt_lm_head and not args.reinit_ppt_embeddings:
        raise OLMoConfigurationError("--reinit-ppt-lm-head requires --reinit-ppt-embeddings")

    if args.stage == PPTStage.ppt:
        if args.ppt_steps <= 0:
            raise OLMoConfigurationError("--ppt-steps must be positive for --stage=ppt")
        run_configurator: RunConfigurator = FixedStepsRunConfigurator(
            base=base_run,
            steps=args.ppt_steps,
            checkpoint_name=f"PPT final ({args.ppt_steps:,d} steps)",
            ppt_batch_size=args.ppt_batch_size,
            ppt_lr=args.ppt_lr,
            ppt_lr_multiplier=args.ppt_lr_multiplier,
            warmup_steps=args.ppt_warmup_steps,
            checkpoint_every=args.ppt_checkpoint_every,
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
        chinchilla_multiple=args.chinchilla_multiple,
        reinit_ppt_embeddings=args.reinit_ppt_embeddings,
        reinit_ppt_lm_head=args.reinit_ppt_lm_head,
        embedding_reinit_seed=args.embedding_reinit_seed,
        ppt_batch_size=args.ppt_batch_size,
        ppt_lr=args.ppt_lr,
        ppt_lr_multiplier=args.ppt_lr_multiplier,
        warmup_steps=args.ppt_warmup_steps,
    )


if __name__ == "__main__":
    main(
        configure_ladder=configure_ladder,
        default_name="ppt-olmo",
        add_additional_args=add_args,
    )
