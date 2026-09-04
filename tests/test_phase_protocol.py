import torch

from src.training.runner import partition, phase_count, total_length


def test_all_nonoverlap_origins_recover_the_same_observations_once():
    context, patch, stride, channels = 512, 12, 12, 2
    values = torch.arange(context * channels, dtype=torch.float32).reshape(1, context, channels)
    for origin in range(phase_count(patch, stride)):
        padded, observed = partition(values, origin, context, patch, stride)
        recovered = padded[0][observed[0]]
        assert recovered.shape == (context, channels)
        assert torch.equal(recovered, values[0])
        assert int(observed.sum()) == context
        assert padded.shape[1] == total_length(context, patch, stride)


def test_partition_keeps_origin_specific_boundary_layout_without_changing_values():
    values = torch.randn(1, 512, 1)
    layouts = []
    for origin in range(12):
        padded, observed = partition(values, origin, 512, 12, 12)
        layouts.append(observed[0].tolist())
        assert torch.equal(padded[0][observed[0]], values[0])
    assert len({tuple(layout) for layout in layouts}) == 12

