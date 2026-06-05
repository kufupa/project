import math
import torch
from smolvla_grpo.dgpo import gaussian_hellinger_sq, dgpo_redistribution_weights


def test_hellinger_zero_when_identical():
    mu = torch.randn(3, 5, 4)
    log_std = torch.zeros(3, 5, 4)
    d = gaussian_hellinger_sq(mu, mu, log_std, log_std)
    assert d.shape == (3, 5)
    assert torch.allclose(d, torch.zeros(3, 5), atol=1e-6)


def test_hellinger_bounded_and_monotonic():
    log_std = torch.zeros(1, 1, 1)
    mu_a = torch.zeros(1, 1, 1)
    d_small = gaussian_hellinger_sq(mu_a, mu_a + 0.5, log_std, log_std)
    d_big = gaussian_hellinger_sq(mu_a, mu_a + 5.0, log_std, log_std)
    assert 0.0 <= d_small.item() < d_big.item() <= 1.0


def test_weights_unit_mean_and_uniform_when_no_deviation():
    # all deviations equal -> softmax uniform -> all weights == 1.0
    dev = torch.zeros(2, 6)
    mask = torch.ones(2, 6, dtype=torch.bool)
    w = dgpo_redistribution_weights(dev, mask, tau=0.5, kappa=0.0)
    assert torch.allclose(w, torch.ones(2, 6), atol=1e-5)
    # unit mean over valid chunks
    assert torch.allclose(w.mean(dim=1), torch.ones(2), atol=1e-5)


def test_weights_concentrate_on_high_deviation_chunk():
    dev = torch.tensor([[0.0, 0.0, 0.9, 0.0]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    w = dgpo_redistribution_weights(dev, mask, tau=0.5, kappa=0.0)
    assert w[0, 2] > w[0, 0]
    assert torch.allclose(w.mean(dim=1), torch.ones(1), atol=1e-5)


def test_weights_respect_mask_unit_mean_over_valid_only():
    dev = torch.tensor([[0.1, 0.2, 0.3, 0.0]])
    mask = torch.tensor([[True, True, True, False]])
    w = dgpo_redistribution_weights(dev, mask, tau=0.5, kappa=0.0)
    # masked chunk weight is 0; mean over the 3 valid == 1
    assert w[0, 3].item() == 0.0
    assert abs(w[0, :3].mean().item() - 1.0) < 1e-5
