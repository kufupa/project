import torch
from scripts.grpo.train_phase11_env_on_policy_grpo import _compute_dgpo_chunk_weights


class _FakeChunk:
    def __init__(self, chunk_idx, *, chunk_len=5, adim=4, valid=True):
        # current/old-policy mean shifts with chunk index -> deviation grows
        self.distr_mean = torch.full((chunk_len, adim), float(chunk_idx))
        self.distr_log_std = torch.zeros(chunk_len, adim)
        self.valid_action_mask = torch.ones(chunk_len, dtype=torch.bool) if valid \
            else torch.zeros(chunk_len, dtype=torch.bool)
        self.proc_snapshot = chunk_idx
        self.flow_sde_trace = {"A_next": torch.zeros(1, chunk_len, adim)}


class _FakeTraj:
    def __init__(self, n, invalid_idx=()):
        self.chunks = [_FakeChunk(i, valid=(i not in invalid_idx)) for i in range(n)]


class _FakeRef:
    """Frozen ref returns mu=0 always -> deviation grows with chunk index."""
    def get_flow_sde_log_probs_for_chunk_from_proc_list(self, procs, traces, *, chunk_len):
        b = len(procs)
        adim = 4
        return (torch.zeros(b, chunk_len), torch.zeros(b, chunk_len, adim),
                torch.zeros(b, chunk_len, adim))


def test_weights_unit_mean_and_concentrate_on_high_deviation():
    rollouts = [_FakeTraj(4)]
    out = _compute_dgpo_chunk_weights(
        _FakeRef(), rollouts, chunk_len=5, tau=0.5, kappa=0.0, device=torch.device("cpu")
    )
    assert len(out) == 1
    w = out[0]                       # weights for the 4 valid chunks, in order
    assert len(w) == 4
    assert abs(sum(w) / len(w) - 1.0) < 1e-4   # unit mean over valid chunks
    assert w[3] > w[0]                          # bigger deviation -> bigger weight


def test_weights_align_to_valid_chunks_only():
    rollouts = [_FakeTraj(4, invalid_idx=(1,))]  # chunk 1 invalid -> skipped
    out = _compute_dgpo_chunk_weights(
        _FakeRef(), rollouts, chunk_len=5, tau=0.5, kappa=0.0, device=torch.device("cpu")
    )
    assert len(out[0]) == 3                      # only 3 valid chunks
    assert abs(sum(out[0]) / 3 - 1.0) < 1e-4


def test_uniform_weights_equal_grpo_advantage():
    # all-equal deviation -> weights all 1.0 -> A*w == A (DGPO reduces to GRPO)
    from smolvla_grpo.dgpo import dgpo_redistribution_weights
    dev = torch.full((1, 8), 0.3)
    mask = torch.ones(1, 8, dtype=torch.bool)
    w = dgpo_redistribution_weights(dev, mask, tau=0.5, kappa=0.0)
    assert torch.allclose(w, torch.ones(1, 8), atol=1e-5)
