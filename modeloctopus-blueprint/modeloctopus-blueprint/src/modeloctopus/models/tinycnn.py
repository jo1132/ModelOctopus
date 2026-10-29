"""Offline CPU fixture, not an M1 model service."""

import io

import torch
from torch import nn

from modeloctopus.core.schema import ArchitectureRecipe, ArtifactRef, ModelOctopusError
from modeloctopus.core.store import LocalArtifactStore


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 3, channels: int = 4):
        super().__init__()
        if (
            type(num_classes) is not int
            or type(channels) is not int
            or min(num_classes, channels) < 1
        ):
            raise ValueError("positive integer dimensions required")
        self.features = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, images):
        return self.classifier(self.features(images).flatten(1))


def create_tinycnn(num_classes: int = 3, channels: int = 4) -> TinyCNN:
    return TinyCNN(num_classes=num_classes, channels=channels)


def describe_architecture(model: TinyCNN) -> ArchitectureRecipe:
    return ArchitectureRecipe(
        factory="modeloctopus.tinycnn",
        revision="1",
        kwargs={
            "num_classes": model.classifier.out_features,
            "channels": model.classifier.in_features,
        },
        expected_shapes={name: list(value.shape) for name, value in model.state_dict().items()},
    )


def rebuild(recipe: ArchitectureRecipe) -> TinyCNN:
    if (
        recipe.factory != "modeloctopus.tinycnn"
        or recipe.revision != "1"
        or recipe.structural_edits
    ):
        raise ModelOctopusError("UNSUPPORTED_ARCHITECTURE", recipe.factory)
    if set(recipe.kwargs) != {"num_classes", "channels"}:
        raise ModelOctopusError("INVALID_ARCHITECTURE", "unexpected architecture kwargs")
    model = create_tinycnn(**recipe.kwargs)  # type: ignore[arg-type]
    if describe_architecture(model).expected_shapes != recipe.expected_shapes:
        raise ModelOctopusError("ARCHITECTURE_SHAPE_MISMATCH", recipe.factory)
    return model


def save_fixture(store: LocalArtifactStore, model: TinyCNN) -> ArtifactRef:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    recipe = describe_architecture(model)
    return store.publish(
        "trainable_torch",
        {"weights.pt": buffer.getvalue(), "architecture.json": recipe.model_dump_json().encode()},
        {"fixture": True, "torch": torch.__version__, "serializer": "state_dict"},
    )


def load_fixture(store: LocalArtifactStore, ref: ArtifactRef) -> TinyCNN:
    if ref.kind != "trainable_torch":
        raise ModelOctopusError("ARTIFACT_KIND_MISMATCH", ref.kind)
    recipe = ArchitectureRecipe.model_validate_json(store.read(ref, "architecture.json"))
    model = rebuild(recipe)
    state = torch.load(
        io.BytesIO(store.read(ref, "weights.pt")), map_location="cpu", weights_only=True
    )
    model.load_state_dict(state, strict=True)
    return model
