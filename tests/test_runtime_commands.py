from pathlib import Path
import pytest
from rsl_rl.runtime_preflight import RuntimePaths, validate_runtime_paths, validate_command_update, blocked_result

def test_paths_no_write_and_containment(tmp_path):
    launch=tmp_path/"launch"; launch.write_text("#!/bin/sh\n"); launch.chmod(0o700)
    asset=tmp_path/"assets"; asset.mkdir(); root=tmp_path/"runs"/"fresh"
    p=validate_runtime_paths(RuntimePaths(launch,root,asset,None))
    assert p.run_root==root and not root.exists()
    with pytest.raises(ValueError,match="executable"):
        validate_runtime_paths(RuntimePaths(asset,root,asset,None))
    manifest=tmp_path/"escape.json"; manifest.write_text("{}")
    with pytest.raises(ValueError,match="inside asset"):
        validate_runtime_paths(RuntimePaths(launch,root,asset,manifest))
    link=tmp_path/"linked"; link.symlink_to(asset,target_is_directory=True)
    with pytest.raises(ValueError,match="canonical"):
        validate_runtime_paths(RuntimePaths(launch,root,link,None))

@pytest.mark.parametrize("update",[{"shell":"rm"},{"goal_x":float("nan")},{"reset":"yes"},{"goal_y":True}])
def test_commands_reject_unknown_nonfinite_and_wrong_types(update):
    with pytest.raises(ValueError):
        validate_command_update(update)

def test_command_and_blocked_schema():
    assert validate_command_update({"goal_x":1,"goal_y":2,"reset":True})=={"goal_x":1.,"goal_y":2.,"reset":True}
    result=blocked_result("isaaclab_adapter","missing launcher","supply executable")
    assert result["status"]=="blocked" and result["validation_rung"]==0

def test_command_filter_saturation_keeps_upstream_recurrence_and_masked_neighbors():
    import torch
    from sea_nav_current_isaaclab_full_method.adapters.command_delay import CommandDelayFilter,CommandDelayConfig
    filt=CommandDelayFilter(CommandDelayConfig(alpha=.5),num_envs=3)
    command=torch.tensor([[9.,9.,9.],[.4,.4,.4],[-9.,-9.,-9.]])
    out1,_=filt.step(command)
    assert torch.allclose(out1,torch.tensor([[1.5,1.,1.],[.2,.2,.2],[-.5,-1.,-1.]]))
    out2,_=filt.step(command)
    assert torch.allclose(filt.filtered,torch.tensor([[2.25,2.25,2.25],[.3,.3,.3],[-2.25,-2.25,-2.25]]))
    assert torch.allclose(out2,torch.tensor([[2.,1.,1.],[.3,.3,.3],[-.5,-1.,-1.]]))
    filt.reset_state(torch.tensor([1]))
    assert filt.filtered.tolist()==[[2.25,2.25,2.25],[0,0,0],[-2.25,-2.25,-2.25]]
    out3,debug=filt.step(torch.zeros(3,3))
    assert torch.allclose(out3,torch.tensor([[1.125,1.,1.],[0,0,0],[-.5,-1.,-1.]]))
    assert torch.allclose(filt.filtered,torch.tensor([[1.125,1.125,1.125],[0,0,0],[-1.125,-1.125,-1.125]]))
    assert torch.equal(debug['executed_command'],out3) and debug['delay_steps']==0
