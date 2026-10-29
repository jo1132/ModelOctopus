"""Content-addressed local bundles with process locks and atomic publication."""

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .schema import ArtifactRef, Kind, ModelOctopusError


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_name(name: str) -> None:
    if (
        not name
        or Path(name).is_absolute()
        or ".." in Path(name).parts
        or "\\" in name
        or "\x00" in name
    ):
        raise ModelOctopusError("INVALID_PATH", name)
    if (
        name == "."
        or name == "manifest.json"
        or name.startswith("manifest.json/")
        or str(Path(name)) != name
    ):
        raise ModelOctopusError("INVALID_PATH", name)


class LocalArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        for name in ("artifacts", "staging", "diagnostics", "locks"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _lock(self, artifact_id: str):
        with (self.root / "locks" / artifact_id).open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def publish(
        self, kind: Kind, files: dict[str, bytes], metadata: dict | None = None
    ) -> ArtifactRef:
        for name, content in files.items():
            safe_name(name)
            if not isinstance(content, bytes):
                raise TypeError("artifact files must be bytes")
        manifest = {
            "schema_version": 1,
            "kind": kind,
            "metadata": metadata or {},
            "files": {name: digest(content) for name, content in sorted(files.items())},
        }
        manifest_bytes = canonical(manifest)
        checksum = digest(manifest_bytes)
        ref = ArtifactRef(id=checksum, kind=kind, digest=checksum)
        with self._lock(ref.id):
            target = self.root / "artifacts" / ref.id
            if target.exists():
                self.verify(ref)
                return ref
            stage = Path(tempfile.mkdtemp(dir=self.root / "staging"))
            try:
                for name, content in {**files, "manifest.json": manifest_bytes}.items():
                    path = stage / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("wb") as handle:
                        handle.write(content)
                        handle.flush()
                        os.fsync(handle.fileno())
                self._verify_directory(stage, ref)
                # Persist directory entries before publishing the complete tree.
                directories = [p for p in stage.rglob("*") if p.is_dir()] + [stage]
                for path in directories:
                    descriptor = os.open(path, os.O_RDONLY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                os.rename(stage, target)
                directory = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            except BaseException:
                if stage.exists():
                    os.rename(stage, self.root / "diagnostics" / stage.name)
                raise
        return ref

    def _verify_directory(self, directory: Path, ref: ArtifactRef) -> dict:
        try:
            if directory.is_symlink() or (directory / "manifest.json").is_symlink():
                raise ValueError("symlink in bundle")
            raw = (directory / "manifest.json").read_bytes()
            manifest = json.loads(raw)
            if digest(raw) != ref.digest or manifest["kind"] != ref.kind:
                raise ValueError("manifest digest/kind mismatch")
            if manifest["schema_version"] != 1:
                raise ValueError("manifest schema mismatch")
            expected = set(manifest["files"]) | {"manifest.json"}
            actual = set()
            for path in directory.rglob("*"):
                if path.is_symlink():
                    raise ValueError("symlink in bundle")
                if path.is_file():
                    actual.add(path.relative_to(directory).as_posix())
            if actual != expected:
                raise ValueError("bundle file inventory mismatch")
            for name, checksum in manifest["files"].items():
                safe_name(name)
                if digest((directory / name).read_bytes()) != checksum:
                    raise ValueError(f"file digest mismatch: {name}")
            return manifest
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ModelOctopusError("ARTIFACT_CORRUPTED", str(exc)) from exc

    def verify(self, ref: ArtifactRef) -> dict:
        return self._verify_directory(self.root / "artifacts" / ref.id, ref)

    def read(self, ref: ArtifactRef, name: str) -> bytes:
        safe_name(name)
        manifest = self.verify(ref)
        if name not in manifest["files"]:
            raise ModelOctopusError("UNKNOWN_ARTIFACT_FILE", name)
        return (self.root / "artifacts" / ref.id / name).read_bytes()
