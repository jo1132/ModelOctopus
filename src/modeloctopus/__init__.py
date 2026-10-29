"""ModelOctopus M0 public core. Backend imports are opt-in."""

from .core.schema import (
    ArchitectureRecipe,
    ArtifactRef,
    CapabilityReport,
    DatasetSpec,
    InitializationSpec,
    ModelOctopusError,
    ModelSpec,
    TrainSpec,
)
from .core.store import LocalArtifactStore

__all__ = [
    "ArchitectureRecipe",
    "ArtifactRef",
    "CapabilityReport",
    "DatasetSpec",
    "InitializationSpec",
    "LocalArtifactStore",
    "ModelOctopusError",
    "ModelSpec",
    "TrainSpec",
]
