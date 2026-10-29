"""Backend-independent, strict version-one public contracts."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal[1] = 1


Kind = Literal[
    "trainable_torch",
    "onnx_fp",
    "onnx_qdq",
    "torch_export",
    "tensorrt_engine",
    "training_state",
    "dataset",
    "evaluation",
    "pruning_plan",
]


class ArtifactRef(Schema):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    kind: Kind
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class InitializationSpec(Schema):
    kind: Literal["random", "pretrained", "weights", "resume_ref"] = "random"
    weights: str | None = None
    source: ArtifactRef | None = None
    strict: bool = True
    head_policy: Literal["preserve", "replace_head"] = "preserve"

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.kind == "random" and (self.weights is not None or self.source is not None):
            raise ValueError("random initialization cannot have weights/source")
        if self.kind == "pretrained" and (not self.weights or self.source is not None):
            raise ValueError("pretrained requires weights only")
        if self.kind in {"weights", "resume_ref"} and (self.source is None or self.weights):
            raise ValueError("weights/resume_ref requires source only")
        if self.source and self.source.kind != (
            "training_state" if self.kind == "resume_ref" else "trainable_torch"
        ):
            raise ValueError("initialization source kind mismatch")
        return self


class ModelSpec(Schema):
    provider: str
    architecture: str
    task: Literal["classification", "detection"]
    revision: str | None = None
    num_classes: int = Field(gt=0)
    initialization: InitializationSpec = Field(default_factory=InitializationSpec)
    input_shape: list[int] = Field(default_factory=lambda: [1, 3, 32, 32])
    serializer: str = "state_dict"
    seed: int = 42

    @model_validator(mode="after")
    def shape(self) -> Self:
        if len(self.input_shape) != 4 or any(n <= 0 for n in self.input_shape):
            raise ValueError("input_shape must be positive NCHW")
        return self


class CalibrationSpec(Schema):
    source_split: Literal["train"] = "train"
    count: int = Field(gt=0)
    seed: int = 42


class TransformNames(Schema):
    training: str
    evaluation: str
    calibration: str


class DatasetSpec(Schema):
    provider: str
    name: str
    task: Literal["classification", "detection"] = "classification"
    revision: str | None = None
    source: str | None = None
    split_policy: str
    seed: int = 42
    download: bool = False
    cache: str | None = None
    calibration: CalibrationSpec | None = None
    transforms: TransformNames | None = None
    final_test_available: bool = True
    smoke_only: bool = False


class OptimizerSpec(Schema):
    name: str = "adamw"
    lr: float = Field(default=0.001, gt=0)


class ObjectiveSpec(Schema):
    plugin: str = "classification.cross_entropy"
    temperature: float = Field(default=1.0, gt=0)
    hard_weight: float = Field(default=1.0, ge=0)
    soft_weight: float = Field(default=0.0, ge=0)


class TrainSpec(Schema):
    engine: str
    mode: Literal["scratch", "finetune", "resume"]
    epochs: int = Field(gt=0)
    batch_size: int = Field(default=32, gt=0)
    optimizer: OptimizerSpec = Field(default_factory=OptimizerSpec)
    scheduler: str | None = None
    amp: bool = False
    seed: int = 42
    objective: ObjectiveSpec = Field(default_factory=ObjectiveSpec)
    resume_scope: Literal["epoch_boundary"] = "epoch_boundary"


class ArchitectureRecipe(Schema):
    factory: str
    revision: str
    kwargs: dict[str, JsonValue] = Field(default_factory=dict)
    structural_edits: list[dict[str, JsonValue]] = Field(default_factory=list)
    expected_shapes: dict[str, list[int]] = Field(default_factory=dict)


class CapabilityReport(Schema):
    status: Literal[
        "planned", "experimental", "supported", "unsupported", "unverified_environment"
    ] = "planned"
    reasons: list[str] = Field(default_factory=list)
    extras: list[str] = Field(default_factory=list)
    environment_profile: str | None = None
    tested_versions: dict[str, str] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def evidence_required(self) -> Self:
        if self.status == "supported" and not self.evidence_refs:
            raise ValueError("supported requires evidence")
        return self


class ErrorReport(Schema):
    code: str
    message: str
    details: dict[str, JsonValue] = Field(default_factory=dict)


class ModelOctopusError(Exception):
    def __init__(self, code: str, message: str):
        self.report = ErrorReport(code=code, message=message)
        super().__init__(message)
