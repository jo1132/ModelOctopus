import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from modeloctopus import (
    ArtifactRef,
    CapabilityReport,
    InitializationSpec,
    LocalArtifactStore,
    ModelSpec,
)
from modeloctopus.core.registry import PluginDescriptor, Registry
from modeloctopus.core.schema import ModelOctopusError


def publish_worker(root):
    return LocalArtifactStore(root).publish("dataset", {"index.json": b"{}"}).model_dump()


def test_core_import_without_backend():
    code = """
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'onnx', 'ultralytics', 'aimet_onnx', 'tensorrt'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
import modeloctopus
from modeloctopus.application.planner import plan
from modeloctopus.application.doctor import doctor
assert doctor()['validation'] == 'passed'
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    "changes",
    [{"num_classes": "3"}, {"typo": 1}, {"input_shape": [0, 3, 4, 4]}, {"num_classes": True}],
)
def test_strict_schema(changes):
    data = dict(provider="custom", architecture="tinycnn", task="classification", num_classes=3)
    with pytest.raises(ValidationError):
        ModelSpec(**(data | changes))


def test_initialization_and_capability():
    for config in ({"kind": "random", "weights": "a"}, {"kind": "weights"}, {"kind": "pretrained"}):
        with pytest.raises(ValidationError):
            InitializationSpec(**config)
    with pytest.raises(ValidationError):
        CapabilityReport(status="supported")
    with pytest.raises(ValidationError):
        ArtifactRef(id="../escape", kind="dataset", digest="0" * 64)


def test_registry():
    registry = Registry()
    calls = []
    descriptor = PluginDescriptor(
        id="test", config_schema=ModelSpec, capabilities=CapabilityReport()
    )
    registry.register(descriptor, lambda: calls.append(1) or object())
    assert calls == []
    with pytest.raises(ModelOctopusError, match="test"):
        registry.register(descriptor, lambda: None)
    with pytest.raises(ModelOctopusError):
        registry.register(
            descriptor.model_copy(update={"id": "old", "api_version": 2}), lambda: None
        )
    with pytest.raises(ValidationError):
        registry.load("test", {"typo": 1})
    assert calls == []
    config = dict(provider="custom", architecture="tinycnn", task="classification", num_classes=3)
    assert registry.load("test", config) is registry.load("test", config)
    assert calls == [1]

    def missing():
        raise ModuleNotFoundError("optional backend")

    registry.register(descriptor.model_copy(update={"id": "missing"}), missing)
    with pytest.raises(ModelOctopusError) as error:
        registry.load("missing", config)
    assert error.value.report.code == "DEPENDENCY_NOT_INSTALLED"


def test_store_hash_corruption_and_paths(tmp_path):
    store = LocalArtifactStore(tmp_path)
    ref = store.publish("dataset", {"sub/data": b"123"}, {"revision": "1"})
    assert ref == store.publish("dataset", {"sub/data": b"123"}, {"revision": "1"})
    assert ref.digest != store.publish("dataset", {"sub/data": b"124"}).digest
    assert store.read(ref, "sub/data") == b"123"
    for name in ("../escape", "/tmp/escape", "manifest.json", "a/../b", "a//b"):
        with pytest.raises(ModelOctopusError):
            store.publish("dataset", {name: b"x"})
    (tmp_path / "artifacts" / ref.id / "sub/data").write_bytes(b"broken")
    with pytest.raises(ModelOctopusError) as error:
        store.verify(ref)
    assert error.value.report.code == "ARTIFACT_CORRUPTED"


def test_partial_write_and_manifest_tamper(tmp_path, monkeypatch):
    store = LocalArtifactStore(tmp_path)
    original = Path.open

    def failing_open(path, *args, **kwargs):
        if path.name == "crash" and args and args[0] == "wb":
            raise OSError("simulated interrupted write")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError):
        store.publish("dataset", {"first": b"ok", "crash": b"x"})
    assert not list((tmp_path / "artifacts").iterdir())
    assert list((tmp_path / "diagnostics").iterdir())
    monkeypatch.setattr(Path, "open", original)
    ref = store.publish("dataset", {"index": b"x"})
    manifest = tmp_path / "artifacts" / ref.id / "manifest.json"
    data = json.loads(manifest.read_text())
    data["metadata"]["changed"] = True
    manifest.write_text(json.dumps(data))
    with pytest.raises(ModelOctopusError):
        store.verify(ref)


def test_duplicate_process_writers(tmp_path):
    with ProcessPoolExecutor(max_workers=3) as pool:
        refs = list(pool.map(publish_worker, [str(tmp_path)] * 6))
    assert all(ref == refs[0] for ref in refs)
    assert len(list((tmp_path / "artifacts").iterdir())) == 1
    assert not list((tmp_path / "staging").iterdir())
    LocalArtifactStore(tmp_path).verify(ArtifactRef(**refs[0]))


def test_stale_staging_is_not_published(tmp_path):
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "staging" / "interrupted").mkdir()
    ref = store.publish("dataset", {"x": b"x"})
    assert store.verify(ref)
    assert (tmp_path / "staging" / "interrupted").exists()


def test_extra_file_and_symlink_rejected(tmp_path):
    store = LocalArtifactStore(tmp_path)
    ref = store.publish("dataset", {"x": b"x"})
    directory = tmp_path / "artifacts" / ref.id
    (directory / "extra").write_bytes(b"x")
    with pytest.raises(ModelOctopusError):
        store.verify(ref)
    (directory / "extra").unlink()
    (directory / "x").unlink()
    (directory / "x").symlink_to(tmp_path / "outside")
    with pytest.raises(ModelOctopusError):
        store.verify(ref)


def write_recipe(tmp_path, steps):
    path = tmp_path / "recipe.yaml"
    path.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "project": {"name": "test", "root": "work"}, "steps": steps}
        )
    )
    return path


def model_step(name="model"):
    return {
        "id": name,
        "kind": "model.create",
        "params": {
            "provider": "custom",
            "architecture": "tinycnn",
            "task": "classification",
            "num_classes": 3,
        },
    }


def test_plan_examples():
    from modeloctopus.application.planner import plan

    for path in Path("examples").glob("*.yaml"):
        result = plan(path)
        assert result["steps"]
        assert result["executable"] is False
        assert all(step["capability"]["status"] == "planned" for step in result["steps"])


def test_planner_invalid_references_cycles_and_kinds(tmp_path):
    from modeloctopus.application.planner import plan

    export = {
        "id": "export",
        "kind": "export.create",
        "inputs": {"model": "@model.model"},
        "params": {"format": "onnx", "target": "ort"},
    }
    recipe = [export, model_step()]
    assert [s["id"] for s in plan(write_recipe(tmp_path, recipe))["steps"]] == ["model", "export"]
    for ref, code in (
        ("@missing.model", "UNKNOWN_REFERENCE"),
        ("@model.nope", "UNKNOWN_REFERENCE"),
        ("python:evil", "INVALID_REFERENCE"),
        ("@export.artifact", "ARTIFACT_KIND_MISMATCH"),
    ):
        export["inputs"]["model"] = ref
        with pytest.raises(ModelOctopusError) as error:
            plan(write_recipe(tmp_path, recipe))
        assert error.value.report.code == code

    def prune(name, source):
        return {
            "id": name,
            "kind": "compression.prune",
            "inputs": {"source": f"@{source}.model", "dataset": "@data.dataset"},
            "params": {"plugin": "test", "channel_ratio": 0.2},
        }

    dataset = {
        "id": "data",
        "kind": "dataset.prepare",
        "params": {"provider": "synthetic", "name": "small", "split_policy": "fixed"},
    }
    with pytest.raises(ModelOctopusError) as error:
        plan(write_recipe(tmp_path, [dataset, prune("a", "b"), prune("b", "a")]))
    assert error.value.report.code == "RECIPE_CYCLE"


def test_planner_unknown_fields_profiles_and_resume(tmp_path):
    from modeloctopus.application.planner import plan

    for step in (
        model_step() | {"profile": "missing"},
        model_step() | {"kind": "unknown"},
        model_step() | {"inputs": {"evil": "@x.y"}},
    ):
        with pytest.raises(ModelOctopusError):
            plan(write_recipe(tmp_path, [step]))
    step = model_step()
    step["params"]["typo"] = 1
    with pytest.raises(ValidationError):
        plan(write_recipe(tmp_path, [step]))
    with pytest.raises(ModelOctopusError):
        plan(write_recipe(tmp_path, [model_step(), model_step()]))
    path = tmp_path / "duplicate.yaml"
    path.write_text("schema_version: 1\nschema_version: 2\n")
    with pytest.raises(ModelOctopusError):
        plan(path)


def test_paths_relative_to_recipe(tmp_path, monkeypatch):
    from modeloctopus.application.planner import plan

    path = write_recipe(
        tmp_path,
        [
            {
                "id": "data",
                "kind": "dataset.prepare",
                "params": {
                    "provider": "local",
                    "name": "x",
                    "source": "data",
                    "cache": "cache",
                    "split_policy": "fixed",
                },
            }
        ],
    )
    monkeypatch.chdir("/")
    result = plan(path)
    assert result["project_root"] == str(tmp_path / "work")
    assert result["steps"][0]["params"]["source"] == str(tmp_path / "data")


def test_cli_error_and_plan():
    process = subprocess.run(
        [sys.executable, "-m", "modeloctopus", "doctor", "--profile", "bad"],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 2
    assert json.loads(process.stderr)["code"] == "UNKNOWN_PROFILE"
    process = subprocess.run(
        [sys.executable, "-m", "modeloctopus", "plan", "examples/classification_kd_ptq.yaml"],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0
    assert json.loads(process.stdout)["executable"] is False


def test_task_and_teacher_class_mismatch(tmp_path):
    from modeloctopus.application.planner import plan

    data = {
        "id": "data",
        "kind": "dataset.prepare",
        "params": {
            "provider": "synthetic",
            "name": "x",
            "task": "detection",
            "split_policy": "fixed",
        },
    }
    train = {
        "id": "train",
        "kind": "train.fit",
        "inputs": {"model": "@model.model", "dataset": "@data.dataset"},
        "params": {"engine": "torch", "mode": "scratch", "epochs": 1},
    }
    with pytest.raises(ModelOctopusError) as error:
        plan(write_recipe(tmp_path, [model_step(), data, train]))
    assert error.value.report.code == "MODEL_TASK_MISMATCH"
    data["params"]["task"] = "classification"
    teacher = model_step("teacher")
    teacher["params"]["num_classes"] = 4
    train["inputs"]["teacher"] = "@teacher.model"
    with pytest.raises(ModelOctopusError) as error:
        plan(write_recipe(tmp_path, [model_step(), data, teacher, train]))
    assert error.value.report.code == "HEAD_CLASS_MAPPING_MISMATCH"
    train["params"]["mode"] = "resume"
    with pytest.raises(ModelOctopusError) as error:
        plan(write_recipe(tmp_path, [model_step(), data, train]))
    assert error.value.report.code == "TRAINING_STATE_INCOMPATIBLE"


def test_registry_entry_point_discovery_is_lazy(monkeypatch):
    from modeloctopus.core import registry as module

    calls = []

    class Entry:
        def load(self):
            return lambda: (
                PluginDescriptor(
                    id="external", config_schema=ModelSpec, capabilities=CapabilityReport()
                ),
                lambda: calls.append("backend"),
            )

    monkeypatch.setattr(module, "entry_points", lambda group: [Entry()])
    registry = Registry()
    registry.discover("models")
    assert registry.describe("external").api_version == 1
    assert calls == []


def test_doctor_actual_cpu_and_cuda():
    from modeloctopus.application.doctor import doctor

    pytest.importorskip("torch")
    cpu = doctor("train-cpu")
    assert cpu["validation"] == "passed"
    assert cpu["device"] == "cpu"
    assert cpu["status"] == "experimental"
    cuda = doctor("train-cuda")
    assert cuda["validation"] in {"passed", "not_run"}
    if cuda["validation"] == "not_run":
        assert cuda["status"] == "unverified_environment"
