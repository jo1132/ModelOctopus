"""Environment inventory; package presence is never execution evidence."""

import importlib.metadata
import platform
import shutil
import subprocess
import sys
from typing import Any

from modeloctopus.core.schema import ModelOctopusError

PROFILES = {
    "core-ci": ["pydantic", "PyYAML"],
    "train-cpu": ["torch"],
    "train-cuda": ["torch"],
    "quant-aimet": ["aimet-onnx", "onnx", "onnxruntime"],
    "deploy-tensorrt": ["tensorrt"],
}


def doctor(profile: str = "core-ci") -> dict[str, Any]:
    if profile not in PROFILES:
        raise ModelOctopusError("UNKNOWN_PROFILE", profile)
    packages: dict[str, str | None] = {}
    for package in PROFILES[profile]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    report: dict[str, Any] = {
        "schema_version": 1,
        "profile": profile,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "docker_cli": shutil.which("docker"),
        "nvidia_smi": shutil.which("nvidia-smi"),
        "validation": "not_run",
        "status": "unverified_environment",
    }
    if profile == "core-ci":
        report.update(validation="passed", status="supported")
    elif profile in {"train-cpu", "train-cuda"} and packages["torch"]:
        code = """import json, torch
from modeloctopus.models.tinycnn import create_tinycnn
model = create_tinycnn().eval()
device = "cuda" if PROFILE == "train-cuda" else "cpu"
if device == "cuda" and not torch.cuda.is_available():
    print(json.dumps({"validation": "not_run", "reason": "CUDA_UNAVAILABLE"}))
else:
    model.to(device)
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 8, 8, device=device))
    assert output.shape == (2, 3) and torch.isfinite(output).all()
    print(json.dumps({"validation": "passed", "shape": list(output.shape), "device": device}))
""".replace("PROFILE", repr(profile))
        process = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        import json

        if process.returncode:
            report.update(validation="failed", diagnostic=process.stderr)
        else:
            report.update(json.loads(process.stdout))
            if report["validation"] == "passed":
                report["status"] = "experimental"  # fixture only, no training support
    if any(version is None for version in packages.values()):
        report["reason"] = "DEPENDENCY_NOT_INSTALLED"
    return report
