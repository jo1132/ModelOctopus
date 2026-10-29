"""Run from the repository: python examples/m0_roundtrip.py (requires torch extra)."""

from tempfile import TemporaryDirectory

import torch

from modeloctopus import LocalArtifactStore
from modeloctopus.models.tinycnn import create_tinycnn, load_fixture, save_fixture

with TemporaryDirectory() as root:
    store = LocalArtifactStore(root)
    torch.manual_seed(42)
    model = create_tinycnn().eval()
    images = torch.zeros(2, 3, 8, 8)
    ref = save_fixture(store, model)
    restored = load_fixture(store, ref).eval()
    with torch.no_grad():
        torch.testing.assert_close(model(images), restored(images), rtol=0, atol=0)
    print(f"TinyCNN CPU round-trip passed: {ref.id}")
