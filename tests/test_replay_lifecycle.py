import pytest
import torch
from rsl_rl.replay import execute_reset_transaction, build_reset_frames, bootstrap_history, advance_history, require_runtime_contract
from test_collision_replay_cpu import ring, push


@pytest.mark.parametrize("failure", [None, "validate", "write", "reconstruct", "epilogue"])
def test_transaction_ack_is_last_and_failed_rows_finish_normal_before_cancel(failure):
    b = ring(n=3, capacity=5, undo=(1,4))
    push(b,[1],[0]); push(b,[1],[1],True)
    s = b.reserve_pre_collision(torch.tensor([1]), undo_steps=1)
    state = torch.tensor([10.,20.,30.])
    observations = state.clone()
    events = []
    failed = [False]
    def stage(name, ids):
        events.append(name)
        if name == failure and not failed[0]:
            failed[0] = True
            raise RuntimeError(name)
    def validate(sel):
        stage("validate",sel.env_ids); b.validate_selection(sel)
    def write(sel):
        state[sel.env_ids] = 100
        stage("write",sel.env_ids)
    def normal(ids):
        state[ids] = 200
        events.append("normal")
    def rebuild(ids):
        observations[ids] = state[ids] + 1
        stage("reconstruct",ids)
    def epilogue(ids):
        assert b.pending_token[1] == s.tokens[0]
        stage("epilogue",ids)
    result = execute_reset_transaction(torch.tensor([0,1]), torch.tensor([False,True]), s, b,
                                      normal, write, rebuild, epilogue, validate=validate)
    assert state[2] == observations[2] == 30
    assert result.normal_ids.tolist() == [0]
    assert result.replay_ids.tolist() == ([1] if failure is None else [])
    assert result.fallback_ids.tolist() == ([] if failure is None else [1])
    assert observations.tolist() == [201,101 if failure is None else 201,30]
    assert b.pending_token[1] == -1
    assert (1,1) in b.cancel_reasons if failure else b.collision_onset[1] == -1


def test_fallback_failure_propagates_without_ack_or_new_episode():
    b = ring(capacity=5,undo=(1,4))
    push(b,[0],[0]); push(b,[0],[1],True)
    s = b.reserve_pre_collision(torch.tensor([0]),undo_steps=1)
    def fail(*_): raise RuntimeError("no physics")
    with pytest.raises(RuntimeError,match="no physics"):
        execute_reset_transaction(torch.tensor([0]),torch.tensor([True]),s,b,fail,fail,fail,fail)
    assert b.pending_token[0] == s.tokens[0]
    assert b.episode_id[0] == 0
    assert not b.cancel_reasons


@pytest.mark.parametrize('failing_hook', ['prepare_rows', 'refresh_rows'])
def test_failed_normal_fallback_hook_never_finishes_or_cancels(failing_hook):
    # Pure lifecycle callbacks model failure timing, not simulator behavior.
    b=ring(capacity=2,undo=(1,1))
    push(b,[0],[0]); push(b,[0],[1],True)
    selected=b.reserve_pre_collision(torch.tensor([0]),undo_steps=1)
    events=[]
    def validate(_):
        events.append('validate')
        raise RuntimeError('invalid scene before replay preparation')
    def normal(ids):
        assert ids.tolist()==[0]
        for stage in ('prepare_rows','physical_write','refresh_rows'):
            events.append(stage)
            if stage==failing_hook:
                raise RuntimeError(stage)
    def forbidden(*_):
        raise AssertionError('no replay write/reconstruction/finish after failed fallback hook')
    with pytest.raises(RuntimeError,match=failing_hook):
        execute_reset_transaction(torch.tensor([0]),torch.tensor([True]),selected,b,
                                  normal,forbidden,forbidden,forbidden,validate=validate)
    assert events==(['validate','prepare_rows'] if failing_hook=='prepare_rows' else
                    ['validate','prepare_rows','physical_write','refresh_rows'])
    assert b.pending_token[0]==selected.tokens[0]
    assert b.collision_onset[0]==1
    assert not b.cancel_reasons


def test_transaction_rejects_reservation_outside_requested_rows_before_effects():
    b=ring(capacity=5,undo=(1,4))
    push(b,[1],[0]); push(b,[1],[1],True)
    selection=b.reserve_pre_collision(torch.tensor([1]),undo_steps=1)
    def forbidden(*_): raise AssertionError("unexpected physical effect")
    with pytest.raises(ValueError,match="requested"):
        execute_reset_transaction(torch.tensor([0]),torch.tensor([False]),selection,b,
                                  forbidden,forbidden,forbidden,forbidden)
    assert b.pending_token[1]==selection.tokens[0]


def test_missing_or_partial_runtime_capability_is_explicitly_blocked():
    with pytest.raises(RuntimeError,match="blocked"):
        require_runtime_contract(None,"isaaclab_adapter")
    with pytest.raises(RuntimeError,match="blocked"):
        require_runtime_contract(object(),"isaac_gym_preview4")


def test_uncovered_stateful_reward_is_rejected_at_configuration_boundary():
    from rsl_rl.replay import validate_reset_reward_terms
    validate_reset_reward_terms(["termination","collision","action_rate","dof_acc"])
    with pytest.raises(ValueError,match="unsupported reset consumer"):
        validate_reset_reward_terms(["unrestored_hidden_controller_state"])


def test_restored_frames_and_histories_ignore_poisoned_terminal_memory():
    root = torch.tensor([[2.,3.,.4,0.,0.,0.,1.,1.,2.,3.,4.,5.,6.]])
    frames = build_reset_frames(root, torch.ones(1,12), torch.full((1,12),2.), torch.zeros(1,12),
                                torch.full((1,41),4.), torch.tensor([[5.,6.]]), "xyzw")
    assert frames["body_linear"].tolist() == [[1,2,3]]
    assert frames["gravity"].tolist() == [[0,0,-1]]
    assert frames["navigation"][0,:12].tolist() == [0,0,-1,0,0,0,1,2,3,4,5,6]
    assert frames["slr"][0,:9].tolist() == [1,1.25,1.5,0,0,-1,0,0,0]
    hist = torch.full((3,10,55),float("nan"))
    hist[0] = 7; hist[2] = 9
    bootstrap_history(hist,torch.tensor([1]),frames["navigation"])
    assert torch.equal(hist[1,0], hist[1,-1])
    assert torch.isfinite(hist).all()
    assert (hist[0] == 7).all() and (hist[2] == 9).all()
    frame = torch.full((3,55),11.)
    advance_history(hist,frame,torch.tensor([False,True,False]))
    assert (hist[1] == 11).all()
    assert (hist[0,:-1] == 7).all() and (hist[0,-1] == 11).all()


def test_bootstrap_rebuilds_all_history_hold_and_baseline_groups_masked():
    from rsl_rl.replay import bootstrap_episode_buffers
    root=torch.tensor([[2.,3.,.4,0.,0.,0.,1.,1.,2.,3.,4.,5.,6.]])
    rays=torch.full((1,41),4.); goal=torch.tensor([[5.,6.]])
    velocities=torch.full((1,12),2.)
    frames=build_reset_frames(root,torch.ones(1,12),velocities,torch.zeros(1,12),rays,goal,"xyzw")
    shapes={"navigation_history":(10,55),"slr_history":(10,45),"ray_history":(10,41),
            "goal_history":(10,2),"position_history":(10,2),"held_rays":(41,),"held_goal":(2,),
            "last_dof_velocity":(12,),"last_root_velocity":(6,),"last_body_twist":(6,),
            "command":(3,),"action":(12,),"filter":(3,),"episode_length":(),"goal_timer":(),"stay_timer":(),
            "collision":(),"previous_collision":(),"initial":(),"static":()}
    state={key:torch.full((3,)+shape,99.) for key,shape in shapes.items()}
    for value in state.values(): value[1]=float("nan")
    bootstrap_episode_buffers(state,torch.tensor([1]),frames,root,velocities,rays,goal,root[:,:2])
    for value in state.values():
        assert torch.isfinite(value).all()
        assert (value[0]==99).all() and (value[2]==99).all()
    assert state["episode_length"][1]==state["goal_timer"][1]==state["stay_timer"][1]==0
    assert state["collision"][1]==state["previous_collision"][1]==0 and state["initial"][1]==1
    assert state["last_root_velocity"][1].tolist()==[1,2,3,4,5,6]
    assert (state["last_dof_velocity"][1]==2).all()
    assert state["position_history"][1].tolist()==[[2,3]]*10
    assert state["goal_history"][1].tolist()==[[5,6]]*10
    assert (state["held_rays"][1]==4).all()
    assert torch.equal(state["slr_history"][1,0],frames["slr"][0])


def test_quaternion_conventions_rotate_world_velocity_and_goal_consistently():
    from rsl_rl.replay import local_goal
    root=torch.tensor([[2.,3.,0.,0.,0.,2**-.5,2**-.5,1.,0.,0.,0.,0.,0.]])
    frames=build_reset_frames(root,torch.zeros(1,12),torch.zeros(1,12),torch.zeros(1,12),torch.ones(1,41),torch.zeros(1,2),"xyzw")
    torch.testing.assert_close(frames["body_linear"],torch.tensor([[0.,-1.,0.]]),atol=1e-6,rtol=0)
    torch.testing.assert_close(local_goal(root,torch.tensor([[3.,3.,0.]])),torch.tensor([[0.,-1.]]),atol=1e-6,rtol=0)
    root[:,3:7]=root[:,[6,3,4,5]]
    wxyz=build_reset_frames(root,torch.zeros(1,12),torch.zeros(1,12),torch.zeros(1,12),torch.ones(1,41),torch.zeros(1,2),"wxyz")
    torch.testing.assert_close(wxyz["navigation"],frames["navigation"])
