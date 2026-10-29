import json
import subprocess
import sys

import pytest

from modeloctopus import LocalArtifactStore


def test_fixture_forward_backward_and_fresh_process(tmp_path):
    torch = pytest.importorskip("torch")
    from modeloctopus.models.tinycnn import create_tinycnn, save_fixture

    torch.manual_seed(42)
    model = create_tinycnn().eval()
    images = torch.linspace(-1, 1, 2 * 3 * 8 * 8).reshape(2, 3, 8, 8)
    expected = model(images).detach()
    model(images).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    store = LocalArtifactStore(tmp_path)
    ref = save_fixture(store, model)
    code = """
import json, sys, torch
from modeloctopus import ArtifactRef, LocalArtifactStore
from modeloctopus.models.tinycnn import load_fixture
model = load_fixture(LocalArtifactStore(sys.argv[1]), ArtifactRef.model_validate_json(sys.argv[2])).eval()
images = torch.linspace(-1, 1, 2 * 3 * 8 * 8).reshape(2, 3, 8, 8)
with torch.no_grad():
    print(json.dumps(model(images).tolist()))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), ref.model_dump_json()],
        capture_output=True,
        text=True,
        check=True,
    )
    torch.testing.assert_close(torch.tensor(json.loads(result.stdout)), expected, rtol=0, atol=0)


def test_bad_recipe_and_strict_state_load(tmp_path):
    torch = pytest.importorskip("torch")
    import io
    from modeloctopus.core.schema import ModelOctopusError
    from modeloctopus.models.tinycnn import (
        create_tinycnn,
        describe_architecture,
        load_fixture,
        rebuild,
    )

    recipe = describe_architecture(create_tinycnn())
    with pytest.raises(ModelOctopusError):
        rebuild(recipe.model_copy(update={"revision": "unknown"}))
    with pytest.raises(ModelOctopusError):
        rebuild(recipe.model_copy(update={"expected_shapes": {}}))
    state = create_tinycnn().state_dict()
    state.pop("classifier.bias")
    buffer = io.BytesIO()
    torch.save(state, buffer)
    store = LocalArtifactStore(tmp_path)
    ref = store.publish(
        "trainable_torch",
        {"weights.pt": buffer.getvalue(), "architecture.json": recipe.model_dump_json().encode()},
    )
    with pytest.raises(RuntimeError, match="Missing key"):
        load_fixture(store, ref)
