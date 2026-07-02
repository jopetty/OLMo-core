"""
Launch ladder experiments trained exclusively on a synthetic sensitivity dataset.

Unlike ``sensitivity_ladder.py``, this ladder does not mix in the standard Dolma/OLMo
pretraining distribution. The selected synthetic dataset is repeated to fill the full
Chinchilla token budget for the requested model size.

Typical usage:

    uv run src/scripts/train/ladder/synthetic_ladder.py launch \
      --size 60M \
      --model-type transformer \
      --dataset-family aperiodic \
      --supervision 100 \
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
import logging
from dataclasses import dataclass
from typing import Literal

from olmo_core.config import StrEnum
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
from olmo_core.model_ladder import (
    DeviceMeshSpec,
    ModelLadder,
    Olmo3ModelConfigurator,
    WSDSChinchillaRunConfigurator,
)
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerConfig

from sensitivity_ladder import (
    SensitivityModelType,
    _format_chinchilla_multiple,
    _wandb_tags,
)

log = logging.getLogger(__name__)

SYNTHETIC_DATA_ROOT = "/weka/oe-training-default/jacksonp/sensitivity-data"
SYNTHETIC_LADDER_ROOT = "/weka/oe-training-default/ai2-llm"

SYNTHETIC_DATASETS: dict[str, tuple[str, int]] = {
    # "aperiodic_0supervision_n200000000_v26_a50_m64_z1p2": (
    #     "data/processed/aperiodic_0supervision_n200000000_v26_a50_m64_z1p2",
    #     12_986_323_520,
    # ),
    # "aperiodic_0supervision_n200000000_v26_a50_m64_z1p2_assignments_gte10": (
    #     "data/processed/aperiodic_0supervision_n200000000_v26_a50_m64_z1p2_assignments_gte10",
    #     9_079_059_147,
    # ),
    "aperiodic_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10": (
        "data/processed/aperiodic_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10",
        3_907_264_373,
    ),
    # "aperiodic_50supervision_n200000000_v26_a50_m64_z1p2": (
    #     "data/processed/aperiodic_50supervision_n200000000_v26_a50_m64_z1p2",
    #     35_693_177_705,
    # ),
    # "aperiodic_50supervision_n200000000_v26_a50_m64_z1p2_assignments_gte10": (
    #     "data/processed/aperiodic_50supervision_n200000000_v26_a50_m64_z1p2_assignments_gte10",
    #     30_449_347_284,
    # ),
    "aperiodic_50supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10": (
        "data/processed/aperiodic_50supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10",
        5_243_830_421,
    ),
    # "aperiodic_100supervision_n200000000_v26_a50_m64_z1p2": (
    #     "data/processed/aperiodic_100supervision_n200000000_v26_a50_m64_z1p2",
    #     39_952_828_315,
    # ),
    # "aperiodic_100supervision_n200000000_v26_a50_m64_z1p2_assignments_gte10": (
    #     "data/processed/aperiodic_100supervision_n200000000_v26_a50_m64_z1p2_assignments_gte10",
    #     33_789_706_236,
    # ),
    "aperiodic_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10": (
        "data/processed/aperiodic_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10",
        6_163_122_079,
    ),
    # "periodic_0supervision_n200000000_v26_a50_m64_z1p2": (
    #     "data/processed/periodic_0supervision_n200000000_v26_a50_m64_z1p2",
    #     12_280_257_160,
    # ),
    # "periodic_0supervision_n200000000_v26_a50_m64_z1p2_assignments_gte10": (
    #     "data/processed/periodic_0supervision_n200000000_v26_a50_m64_z1p2_assignments_gte10",
    #     8_495_963_227,
    # ),
    "periodic_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10": (
        "data/processed/periodic_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10",
        3_784_293_933,
    ),
    # "periodic_50supervision_n200000000_v26_a50_m64_z1p2": (
    #     "data/processed/periodic_50supervision_n200000000_v26_a50_m64_z1p2",
    #     34_992_390_055,
    # ),
    # "periodic_50supervision_n200000000_v26_a50_m64_z1p2_assignments_gte10": (
    #     "data/processed/periodic_50supervision_n200000000_v26_a50_m64_z1p2_assignments_gte10",
    #     29_871_525_899,
    # ),
    "periodic_50supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10": (
        "data/processed/periodic_50supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10",
        5_120_864_156,
    ),
    # "periodic_100supervision_n200000000_v26_a50_m64_z1p2": (
    #     "data/processed/periodic_100supervision_n200000000_v26_a50_m64_z1p2",
    #     39_254_458_432,
    # ),
    # "periodic_100supervision_n200000000_v26_a50_m64_z1p2_assignments_gte10": (
    #     "data/processed/periodic_100supervision_n200000000_v26_a50_m64_z1p2_assignments_gte10",
    #     33_214_964_138,
    # ),
    "periodic_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10": (
        "data/processed/periodic_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10",
        6_039_494_294,
    ),
    # "r-trivial_0supervision_n200000000_v26_a50_m64_z1p2": (
    #     "data/processed/r-trivial_0supervision_n200000000_v26_a50_m64_z1p2",
    #     7_220_383_790,
    # ),
    # "r-trivial_0supervision_n200000000_v26_a50_m64_z1p2_assignments_gte5": (
    #     "data/processed/r-trivial_0supervision_n200000000_v26_a50_m64_z1p2_assignments_gte5",
    #     4_416_670_621,
    # ),
    "r-trivial_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5": (
        "data/processed/r-trivial_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5",
        2_803_713_169,
    ),
    # "r-trivial_0supervision_n200000000_v26_a50_m64_z1p2_s0": (
    #     "data/processed/r-trivial_0supervision_n200000000_v26_a50_m64_z1p2_s0",
    #     7_220_383_790,
    # ),
    # "r-trivial_0supervision_n200000000_v26_a50_m64_z1p2_s0_assignments_gte5": (
    #     "data/processed/r-trivial_0supervision_n200000000_v26_a50_m64_z1p2_s0_assignments_gte5",
    #     4_416_670_621,
    # ),
    # "r-trivial_0supervision_n200000000_v26_a50_m64_z1p2_s0_assignments_lt5": (
    #     "data/processed/r-trivial_0supervision_n200000000_v26_a50_m64_z1p2_s0_assignments_lt5",
    #     2_803_713_169,
    # ),
    # "r-trivial_50supervision_n200000000_v26_a50_m64_z1p2": (
    #     "data/processed/r-trivial_50supervision_n200000000_v26_a50_m64_z1p2",
    #     13_012_519_045,
    # ),
    # "r-trivial_50supervision_n200000000_v26_a50_m64_z1p2_assignments_gte5": (
    #     "data/processed/r-trivial_50supervision_n200000000_v26_a50_m64_z1p2_assignments_gte5",
    #     9_948_565_788,
    # ),
    "r-trivial_50supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5": (
        "data/processed/r-trivial_50supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5",
        3_063_953_257,
    ),
    # "r-trivial_50supervision_n200000000_v26_a50_m64_z1p2_s1": (
    #     "data/processed/r-trivial_50supervision_n200000000_v26_a50_m64_z1p2_s1",
    #     13_012_519_045,
    # ),
    # "r-trivial_50supervision_n200000000_v26_a50_m64_z1p2_s1_assignments_gte5": (
    #     "data/processed/r-trivial_50supervision_n200000000_v26_a50_m64_z1p2_s1_assignments_gte5",
    #     9_948_565_788,
    # ),
    # "r-trivial_50supervision_n200000000_v26_a50_m64_z1p2_s1_assignments_lt5": (
    #     "data/processed/r-trivial_50supervision_n200000000_v26_a50_m64_z1p2_s1_assignments_lt5",
    #     3_063_953_257,
    # ),
    # "r-trivial_100supervision_n200000000_v26_a50_m64_z1p2": (
    #     "data/processed/r-trivial_100supervision_n200000000_v26_a50_m64_z1p2",
    #     14_552_656_130,
    # ),
    # "r-trivial_100supervision_n200000000_v26_a50_m64_z1p2_assignments_gte5": (
    #     "data/processed/r-trivial_100supervision_n200000000_v26_a50_m64_z1p2_assignments_gte5",
    #     11_041_219_279,
    # ),
    "r-trivial_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5": (
        "data/processed/r-trivial_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5",
        3_511_436_851,
    ),
    # "r-trivial_100supervision_n200000000_v26_a50_m64_z1p2_s2": (
    #     "data/processed/r-trivial_100supervision_n200000000_v26_a50_m64_z1p2_s2",
    #     7_302_818_917,
    # ),
    # "r-trivial_100supervision_n200000000_v26_a50_m64_z1p2_s2_assignments_gte5": (
    #     "data/processed/r-trivial_100supervision_n200000000_v26_a50_m64_z1p2_s2_assignments_gte5",
    #     5_540_471_093,
    # ),
    # "r-trivial_100supervision_n200000000_v26_a50_m64_z1p2_s2_assignments_lt5": (
    #     "data/processed/r-trivial_100supervision_n200000000_v26_a50_m64_z1p2_s2_assignments_lt5",
    #     1_762_347_824,
    # ),
}

SYNTHETIC_DATASET_ALIASES: dict[tuple[str, str, str], str] = {
    ("aperiodic", "0sup", "lt10"): (
        "aperiodic_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10"
    ),
    ("aperiodic", "50sup", "lt10"): (
        "aperiodic_50supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10"
    ),
    ("aperiodic", "100sup", "lt10"): (
        "aperiodic_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10"
    ),
    ("periodic", "0sup", "lt10"): (
        "periodic_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10"
    ),
    ("periodic", "50sup", "lt10"): (
        "periodic_50supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10"
    ),
    ("periodic", "100sup", "lt10"): (
        "periodic_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10"
    ),
    ("r-trivial", "0sup", "lt5"): (
        "r-trivial_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5"
    ),
    ("r-trivial", "50sup", "lt5"): (
        "r-trivial_50supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5"
    ),
    ("r-trivial", "100sup", "lt5"): (
        "r-trivial_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5"
    ),
}

SYNTHETIC_DATASET_CHINCHILLA_MULTIPLES: dict[str, float] = {
    # Largest power-of-two multiple that fits within the dataset for both the transformer
    # and hybrid 60M configs. The hybrid has the slightly larger 1xC budget, so it determines
    # these conservative choices.
    "aperiodic_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10": 2,
    "aperiodic_50supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10": 4,
    "aperiodic_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10": 4,
    "periodic_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10": 2,
    "periodic_50supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10": 4,
    "periodic_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10": 4,
    "r-trivial_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5": 2,
    "r-trivial_50supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5": 2,
    "r-trivial_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5": 2,
}


class SyntheticSize(StrEnum):
    size_60M = "60M"


def _source_label(dataset: str) -> str:
    return dataset.replace("_n200000000_v26_a50_m64_z1p2", "")


def _root_dir(cluster: str) -> str:
    if cluster.startswith("ai2/"):
        return SYNTHETIC_LADDER_ROOT
    return "gs://ai2-llm"


@dataclass(kw_only=True)
class SyntheticModelConfigurator(Olmo3ModelConfigurator):
    """Configure transformer or hybrid synthetic-ladder models from the OLMo3 60M preset."""

    model_type: Literal["transformer", "hybrid"]

    def configure_rank_microbatch_size(
        self,
        *,
        size_spec: str,
        sequence_length: int,
        device_type: str,
    ) -> int:
        return super().configure_rank_microbatch_size(
            size_spec=size_spec,
            sequence_length=sequence_length,
            device_type=device_type,
        )

    def configure_minimal_device_mesh_spec(
        self,
        *,
        size_spec: str,
        sequence_length: int,
        device_type: str,
    ) -> DeviceMeshSpec:
        return super().configure_minimal_device_mesh_spec(
            size_spec=size_spec,
            sequence_length=sequence_length,
            device_type=device_type,
        )

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
        if self.model_type == SensitivityModelType.transformer:
            return config

        assert isinstance(config.block, TransformerBlockConfig)
        assert isinstance(config.block.sequence_mixer, AttentionConfig)
        if config.n_layers % 4 != 0:
            raise OLMoConfigurationError(
                f"Synthetic hybrid model requires n_layers to be divisible by 4; got "
                f"{config.n_layers}."
            )

        attn_block = config.block
        num_heads = attn_block.sequence_mixer.n_heads
        gdn_block = attn_block.replace(
            sequence_mixer=GatedDeltaNetConfig(
                n_heads=num_heads,
                head_dim=int(0.75 * config.d_model / num_heads),
                allow_neg_eigval=True,
            ),
        )
        config.block = {"gdn": gdn_block, "attn": attn_block}
        config.block_pattern = ["gdn", "gdn", "gdn", "attn"]
        return config


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


def _get_synthetic_tokens(args: argparse.Namespace) -> int:
    if args.mixture_dataset_tokens is not None:
        return args.mixture_dataset_tokens

    _, configured_tokens = SYNTHETIC_DATASETS[args.mixture_dataset]
    if configured_tokens > 0:
        return configured_tokens

    log.warning(
        "No token count is configured for synthetic dataset '%s'. Recording 0 in run tags; "
        "pass --mixture-dataset-tokens to override.",
        args.mixture_dataset,
    )
    return 0


def _normalize_dataset_family(family: str) -> str:
    family = family.lower().replace("_", "-")
    if family in {"rtrivial", "trivial"}:
        family = "r-trivial"
    return family


def _normalize_supervision(supervision: str) -> str:
    supervision = supervision.lower()
    if supervision.endswith("supervision"):
        supervision = supervision.removesuffix("supervision") + "sup"
    elif supervision.isdigit():
        supervision = f"{supervision}sup"
    return supervision


def _infer_split(family: str) -> str:
    if family in {"aperiodic", "periodic"}:
        return "lt10"
    if family == "r-trivial":
        return "lt5"
    raise OLMoConfigurationError(
        f"Unknown synthetic dataset family '{family}'. Expected one of: "
        "aperiodic, periodic, r-trivial."
    )


def _normalize_dataset_spec(dataset_spec: list[str]) -> tuple[str, str, str] | None:
    if not dataset_spec:
        return None
    if len(dataset_spec) not in {2, 3}:
        raise OLMoConfigurationError(
            "Synthetic dataset shorthand must have 2 parts, e.g. `aperiodic 0` or "
            "`r-trivial 50`. The legacy 3-part form with an explicit split is also accepted."
        )

    family = _normalize_dataset_family(dataset_spec[0])
    supervision = _normalize_supervision(dataset_spec[1])
    split = _infer_split(family)
    if len(dataset_spec) == 3:
        explicit_split = dataset_spec[2].lower()
        if explicit_split != split:
            raise OLMoConfigurationError(
                f"Split '{explicit_split}' does not match family '{family}'. Expected '{split}'."
            )

    return family, supervision, split


def _resolve_mixture_dataset(args: argparse.Namespace) -> str:
    named_spec = None
    dataset_family = getattr(args, "dataset_family", None)
    supervision = getattr(args, "supervision", None)
    if dataset_family is not None or supervision is not None:
        if dataset_family is None or supervision is None:
            raise OLMoConfigurationError(
                "Specify both --dataset-family and --supervision, or pass --mixture-dataset."
            )
        family = _normalize_dataset_family(dataset_family)
        named_spec = (
            family,
            _normalize_supervision(supervision),
            _infer_split(family),
        )

    positional_spec = _normalize_dataset_spec(getattr(args, "dataset_spec", []))
    shorthand = named_spec or positional_spec
    if shorthand is not None:
        if args.mixture_dataset is not None:
            raise OLMoConfigurationError(
                "Specify either --dataset-family/--supervision or --mixture-dataset, not both."
            )
        if named_spec is not None and positional_spec is not None:
            raise OLMoConfigurationError(
                "Specify the synthetic dataset condition with named flags, not positional args."
            )
        try:
            return SYNTHETIC_DATASET_ALIASES[shorthand]
        except KeyError as exc:
            available = ", ".join(" ".join(alias) for alias in sorted(SYNTHETIC_DATASET_ALIASES))
            raise OLMoConfigurationError(
                f"Unknown synthetic dataset shorthand '{' '.join(shorthand)}'. "
                f"Available conditions: {available}."
            ) from exc

    if args.mixture_dataset is not None:
        return args.mixture_dataset

    raise OLMoConfigurationError(
        "Specify a synthetic dataset condition, e.g. "
        "`--dataset-family aperiodic --supervision 0`, or pass --mixture-dataset."
    )


def _resolve_chinchilla_multiple(args: argparse.Namespace) -> float:
    if args.chinchilla_multiple is not None:
        return args.chinchilla_multiple

    try:
        return SYNTHETIC_DATASET_CHINCHILLA_MULTIPLES[args.mixture_dataset]
    except KeyError as exc:
        raise OLMoConfigurationError(
            f"No automatic Chinchilla multiple is configured for dataset "
            f"'{args.mixture_dataset}'. Pass --chinchilla-multiple explicitly."
        ) from exc


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
        workspace="ai2/beyond-state",
        # workspace="ai2/linear-rnns",
        budget="ai2/oe-other",
        priority="urgent",
        chinchilla_multiple=None,
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
        "--dataset-family",
        "--family",
        choices=["aperiodic", "periodic", "r-trivial"],
        default=None,
        help=(
            "Synthetic dataset family. The split is inferred: lt10 for "
            "aperiodic/periodic and lt5 for r-trivial."
        ),
    )
    parser.add_argument(
        "--supervision",
        choices=["0", "50", "100"],
        default=None,
        help="Synthetic dataset supervision level.",
    )
    parser.add_argument(
        "--mixture-dataset",
        choices=sorted(SYNTHETIC_DATASETS),
        default=None,
        help=(
            "Full synthetic dataset directory name to use for all pretraining tokens. "
            "Usually --dataset-family and --supervision are easier."
        ),
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
    parser.add_argument(
        "--mixture-dataset-tokens",
        type=int,
        default=None,
        help=(
            "Optional raw token count for the synthetic dataset, used only for run metadata. "
            "The synthetic-only training duration is determined by the model size and "
            "Chinchilla multiple."
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
    args.mixture_dataset = _resolve_mixture_dataset(args)
    args.chinchilla_multiple = _resolve_chinchilla_multiple(args)

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
    mixture_dataset_tokens = _get_synthetic_tokens(args)
    return SyntheticLadder(
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
