import sys
import torch

sys.path.append("src")  # noqa: E402
from training.model import build_model  # noqa: E402


def test_smoke():
    """Basic smoke test: model builds and runs a forward pass."""
    model = build_model(num_classes=7, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 7)
