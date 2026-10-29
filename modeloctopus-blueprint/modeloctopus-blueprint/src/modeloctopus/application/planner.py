"""Static planning only; never executes recipe operations or Python imports."""

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from modeloctopus.core.schema import (
    CapabilityReport,
    DatasetSpec,
    ModelOctopusError,
    ModelSpec,
    Schema,
    TrainSpec,
)


class ExportParams(Schema):
    format: Literal["onnx", "torch_export"]
    target: str
    shape_policy: Literal["fixed", "dynamic"] = "fixed"
    exporter_profile: str | None = None
    output_semantics: str | None = None


class QuantizationParams(Schema):
    plugin: str
    split: Literal["calibration"] = "calibration"
    target: str
    weight_bits: int = Field(default=8, gt=0)
    activation_bits: int = Field(default=8, gt=0)
    estimator: str = "minmax"


class PruningParams(Schema):
    plugin: str
    channel_ratio: float = Field(gt=0, lt=1)
    preserve_task_head: bool = True


class BuildParams(Schema):
    backend: Literal["tensorrt"]
    profile: str


class EvaluationParams(Schema):
    split: Literal["val", "test"]
    runtime: str
    metrics: list[str]
    max_drop_pp: float | None = Field(default=None, ge=0)
    evaluator: str | None = None
    smoke_only: bool = False


class GraphParams(Schema):
    include_qdq: bool = True
    format: Literal["html", "json"] = "json"


class ProjectSpec(Schema):
    name: str
    root: str


class Step(Schema):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    kind: str
    inputs: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    profile: str | None = None


class Recipe(Schema):
    schema_version: Literal[1]
    project: ProjectSpec
    environment_profiles: dict[str, str] = Field(default_factory=dict)
    steps: list[Step] = Field(min_length=1)


# Inputs map name to (accepted kinds, required). Outputs have concrete artifact kinds.
MODEL = {"trainable_torch"}
EXPORTED = {"onnx_fp", "onnx_qdq", "torch_export", "tensorrt_engine"}
RULES: dict[str, tuple[type[Schema], dict, dict]] = {
    "model.create": (ModelSpec, {}, {"model": "trainable_torch"}),
    "dataset.prepare": (DatasetSpec, {}, {"dataset": "dataset"}),
    "train.fit": (
        TrainSpec,
        {
            "model": (MODEL, True),
            "dataset": ({"dataset"}, True),
            "teacher": (MODEL, False),
            "training_state": ({"training_state"}, False),
        },
        {"model": "trainable_torch", "training_state": "training_state"},
    ),
    "export.create": (ExportParams, {"model": (MODEL, True)}, {"artifact": "onnx_fp"}),
    "compression.quantize": (
        QuantizationParams,
        {"source": ({"onnx_fp"}, True), "dataset": ({"dataset"}, True)},
        {"artifact": "onnx_qdq"},
    ),
    "compression.prune": (
        PruningParams,
        {"source": (MODEL, True), "dataset": ({"dataset"}, True)},
        {"model": "trainable_torch", "plan": "pruning_plan"},
    ),
    "deployment.build": (
        BuildParams,
        {"source": ({"onnx_fp", "onnx_qdq"}, True)},
        {"artifact": "tensorrt_engine"},
    ),
    "evaluation.compare": (
        EvaluationParams,
        {
            "reference": (MODEL | EXPORTED, True),
            "candidate": (MODEL | EXPORTED, True),
            "dataset": ({"dataset"}, True),
        },
        {"report": "evaluation"},
    ),
    "graph.inspect": (
        GraphParams,
        {"artifact": ({"onnx_fp", "onnx_qdq"}, True)},
        {"report": "evaluation"},
    ),
}


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ModelOctopusError("DUPLICATE_YAML_KEY", str(key))
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def load_recipe(path: str | Path) -> Recipe:
    return Recipe.model_validate(yaml.load(Path(path).read_text(), Loader=UniqueLoader))


def plan(path: str | Path) -> dict:
    recipe = load_recipe(path)
    steps = {step.id: step for step in recipe.steps}
    if len(steps) != len(recipe.steps):
        raise ModelOctopusError("DUPLICATE_STEP", "step ids must be unique")
    dependencies: dict[str, set[str]] = {}
    outputs: dict[str, dict] = {}
    resolved = {}
    base = Path(path).resolve().parent
    for step in recipe.steps:
        if step.kind not in RULES:
            raise ModelOctopusError("UNKNOWN_OPERATION", step.kind)
        schema, inputs, result = RULES[step.kind]
        params = schema.model_validate(step.params).model_dump(mode="json")
        if step.profile and step.profile not in recipe.environment_profiles:
            raise ModelOctopusError("UNKNOWN_PROFILE", step.profile)
        if set(step.inputs) - set(inputs):
            raise ModelOctopusError("UNKNOWN_INPUT", step.id)
        if any(required and name not in step.inputs for name, (_, required) in inputs.items()):
            raise ModelOctopusError("MISSING_INPUT", step.id)
        if step.kind == "train.fit":
            if (params["mode"] == "resume") != ("training_state" in step.inputs):
                raise ModelOctopusError("TRAINING_STATE_INCOMPATIBLE", step.id)
        if step.kind == "dataset.prepare":
            for key in ("source", "cache"):
                if params[key] is not None:
                    params[key] = str((base / params[key]).resolve())
        outputs[step.id] = dict(result)
        if step.kind == "export.create" and params["format"] == "torch_export":
            outputs[step.id]["artifact"] = "torch_export"
        resolved[step.id] = params
        dependencies[step.id] = set()
    for step in recipe.steps:
        for name, value in step.inputs.items():
            match = re.fullmatch(r"@([A-Za-z][A-Za-z0-9_-]*)\.([a-z_]+)", value)
            if not match:
                raise ModelOctopusError("INVALID_REFERENCE", value)
            parent, output = match.groups()
            if parent not in steps or output not in outputs[parent]:
                raise ModelOctopusError("UNKNOWN_REFERENCE", value)
            if outputs[parent][output] not in RULES[step.kind][1][name][0]:
                raise ModelOctopusError("ARTIFACT_KIND_MISMATCH", value)
            dependencies[step.id].add(parent)
    ordered: list[str] = []
    pending = dict(dependencies)
    while pending:
        ready = [name for name, parents in pending.items() if parents <= set(ordered)]
        if not ready:
            raise ModelOctopusError("RECIPE_CYCLE", ", ".join(pending))
        ordered.extend(ready)
        for name in ready:
            del pending[name]
    tasks: dict[str, str] = {}
    classes: dict[str, int] = {}
    for name in ordered:
        step = steps[name]
        params = resolved[name]
        parents = [value[1:].split(".")[0] for value in step.inputs.values()]
        known_tasks = {tasks[parent] for parent in parents if parent in tasks}
        if "task" in params:
            known_tasks.add(params["task"])
        if len(known_tasks) > 1:
            raise ModelOctopusError("MODEL_TASK_MISMATCH", name)
        if known_tasks:
            tasks[name] = next(iter(known_tasks))
        model_parents = [parent for parent in parents if parent in classes]
        if step.kind == "train.fit" and len({classes[p] for p in model_parents}) > 1:
            raise ModelOctopusError("HEAD_CLASS_MAPPING_MISMATCH", name)
        if "num_classes" in params:
            classes[name] = params["num_classes"]
        elif model_parents:
            classes[name] = classes[model_parents[0]]
    return {
        "schema_version": 1,
        "project_root": str((base / recipe.project.root).resolve()),
        "steps": [
            {
                "id": name,
                "kind": steps[name].kind,
                "params": resolved[name],
                "inputs": steps[name].inputs,
                "outputs": outputs[name],
                "profile": recipe.environment_profiles.get(steps[name].profile or ""),
                "capability": CapabilityReport(
                    reasons=["M1_PLUS_OPERATION_NOT_IMPLEMENTED"],
                ).model_dump(),
                "deferred": [
                    "class mapping/actual model structure",
                    "backend/hardware compatibility",
                ],
            }
            for name in ordered
        ],
        "executable": False,
    }
