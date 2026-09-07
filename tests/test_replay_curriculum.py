import torch
import pytest
from rsl_rl.replay import CurriculumConfig, reset_probability, update_goal_level, select_collision_reset, select_terminal_replay


def test_literal_equation_one_and_upstream_scaling():
    levels = torch.tensor([-1., 0., .5, 1., 1.5, 2.])
    assert torch.allclose(reset_probability(levels, CurriculumConfig(mode="paper_v1")), torch.tensor([.1,.1,.3,.5,.5,.5]))
    assert torch.allclose(reset_probability(levels, CurriculumConfig()), torch.tensor([.1,.1,.1+.4/3,.1+.8/3,.5,.5]))


def test_strict_events_mask_and_stored_level_clamp():
    cfg = CurriculumConfig(max_level=2)
    value, up, down = update_goal_level(torch.zeros(6), torch.tensor([.49,.5,1.,2.,2.01,.1]), torch.tensor([1,1,1,1,1,0], dtype=torch.bool), cfg)
    assert value.tolist() == [1,0,0,0,0,0]
    assert up.tolist() == [True,False,False,False,False,False]
    assert down.tolist() == [False,False,False,False,True,False]
    value, _, _ = update_goal_level(value, torch.zeros(6), torch.ones(6, dtype=torch.bool), cfg)
    assert value.tolist() == [2,1,1,1,1,1]


def test_two_independent_gates_success_timeout_and_paper_carry():
    yes = torch.ones(4, dtype=torch.bool)
    no = ~yes
    first = select_collision_reset(yes, yes, torch.full((4,), .5), torch.tensor([.49,.5,.1,.9]))
    assert first.tolist() == [True,False,True,False]
    second = select_terminal_replay(yes, torch.tensor([0,0,1,0],dtype=torch.bool), torch.tensor([0,0,0,1],dtype=torch.bool), first, torch.tensor([.8,.79,0.,0.]), CurriculumConfig())
    assert second.tolist() == [False,True,False,False]
    paper = select_terminal_replay(yes, no, no, first, None, CurriculumConfig(mode="paper_v1"))
    assert torch.equal(paper, first)


def test_resolved_replay_consumer_requires_declared_delta_and_uses_acsi_values():
    from pathlib import Path
    from rsl_rl.experiment_config import resolve_run_config
    from rsl_rl.replay import replay_configs_from_resolved
    path=Path(__file__).resolve().parents[1]/"configs/parity_registry.yaml"
    base=dict(registry_path=path,algorithm_profile="upstream_fbce672c",runtime_stack="isaac_gym_preview4")
    missing=resolve_run_config(**base,implementation_delta=["ppo_state_identity_repair"])
    with pytest.raises(ValueError,match="replay_reset_reconstruction_v1"):
        replay_configs_from_resolved(missing,max_level=7)
    resolved=resolve_run_config(**base,implementation_delta=["ppo_state_identity_repair","replay_reset_reconstruction_v1"])
    replay,acsi=replay_configs_from_resolved(resolved,max_level=7)
    assert replay.reconstruction_policy=="new_replay_episode_v1"
    assert acsi.max_level==7 and acsi.terminal_replay_probability==.8
    assert reset_probability(torch.tensor([1.]),acsi).item()==pytest.approx(.1+.4/1.5)
