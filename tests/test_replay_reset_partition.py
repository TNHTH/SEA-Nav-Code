import torch
import pytest
from rsl_rl.replay import partition_reset_env_ids


def test_disjoint_complete_three_way_partition():
    normal, replay, fallback = partition_reset_env_ids(torch.tensor([2,4,7,8]), torch.tensor([0,1,1,0],dtype=torch.bool), torch.tensor([0,1,0,0],dtype=torch.bool))
    assert normal.tolist() == [2,8]
    assert replay.tolist() == [4]
    assert fallback.tolist() == [7]


def test_invalid_partition_is_rejected():
    with pytest.raises(ValueError):
        partition_reset_env_ids(torch.tensor([2,2]), torch.tensor([1,0],dtype=torch.bool), torch.tensor([1,0],dtype=torch.bool))
