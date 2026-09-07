import pytest
import torch

from rsl_rl.replay import CollisionReplayBuffer, CollisionReplayConfig, ReplayTensorSpec


def ring(n=2, capacity=151, undo=(100, 150)):
    return CollisionReplayBuffer(CollisionReplayConfig(ring_buffer_steps=capacity, undo_steps_range=undo),
                                 n, spec=ReplayTensorSpec(dof_count=2, geometry_fields=(("target", 3),)))


def push(buf, ids, steps, collision=False):
    ids = torch.tensor(ids)
    steps = torch.tensor(steps)
    root = torch.zeros(len(ids), 13)
    root[:, 0] = steps + ids * 1000
    root[:, 6] = 1
    buf.push(env_ids=ids, root_state=root, dof_pos=root[:, :2], dof_vel=root[:, :2] + 1,
             task_state={"target": root[:, :3] + 3}, collision=torch.full((len(ids),), collision),
             record_step_ids=steps)
    return root


def test_capacity_covers_inclusive_maximum():
    with pytest.raises(ValueError, match="ring_buffer_steps"):
        ring(capacity=150)


def test_onset_anchor_inclusive_endpoints_and_alias_isolation():
    b = ring()
    for step in range(151):
        value = push(b, [0, 1], [step, step], step == 150)
    value.zero_()
    s = b.reserve_pre_collision(torch.tensor([0, 1]), undo_steps=torch.tensor([100, 150]))
    assert s.batch.step_ids.tolist() == [50, 0]
    assert s.batch.fields["root_state"][:, 0].tolist() == [50, 1000]
    assert s.requested_undo.tolist() == s.effective_undo.tolist() == [100, 150]
    assert not s.short_history.any()
    b.cancel_restore(s, "retry")
    push(b, [0], [151], True)
    again = b.reserve_pre_collision(torch.tensor([0]), undo_steps=100)
    assert again.collision_steps.tolist() == [150]
    assert again.batch.step_ids.tolist() == [50]


def test_independent_wrap_short_history_cancel_and_consumption():
    b = ring(capacity=5, undo=(1, 4))
    for step in range(13):
        push(b, [0], [step], step == 12)
    for step in range(3):
        push(b, [1], [step], step == 2)
    s = b.reserve_pre_collision(torch.tensor([0, 1]), undo_steps=4)
    assert s.batch.step_ids.tolist() == [8, 0]
    assert s.effective_undo.tolist() == [4, 2]
    assert s.short_history.tolist() == [False, True]
    b.acknowledge_restore(s.subset(torch.tensor([True, False])))
    b.cancel_restore(s.subset(torch.tensor([False, True])), "failed write")
    assert b.reserve_pre_collision(torch.tensor([0]), undo_steps=1).env_ids.numel() == 0
    retry = b.reserve_pre_collision(torch.tensor([1]), undo_steps=1)
    assert retry.batch.step_ids.tolist() == [1]
    b.begin_episode(torch.tensor([1]))
    with pytest.raises(ValueError, match="stale"):
        b.acknowledge_restore(retry)
    assert b.reserve_pre_collision(torch.tensor([1]), undo_steps=1).env_ids.numel() == 0


def test_pending_overwrite_rejected_even_with_immutable_staging():
    b = ring(capacity=5, undo=(1, 4))
    for step in range(5):
        push(b, [0], [step], step == 4)
    s = b.reserve_pre_collision(torch.tensor([0]), undo_steps=4)
    push(b, [0], [5])
    assert s.batch.fields["root_state"][0, 0] == 0
    with pytest.raises(ValueError, match="stale"):
        b.acknowledge_restore(s)
    b.cancel_restore(s, "overwritten")


def test_context_and_physical_validation_before_commit():
    b = ring(capacity=5, undo=(1, 4))
    push(b, [0], [0])
    push(b, [0], [1], True)
    s = b.reserve_pre_collision(torch.tensor([0]), undo_steps=1)
    b.validate_selection(s)
    s.batch.fields["root_state"][0, 6] = 0
    with pytest.raises(ValueError, match="quaternion"):
        b.validate_selection(s)
    b.task_generation[0] += 1
    with pytest.raises(ValueError, match="stale"):
        b.validate_selection(s)
    b.cancel_restore(s,"scene generation changed; normal fallback completed")
    assert b.pending_token[0] == -1


def test_random_undo_draw_includes_both_endpoints():
    b=ring(n=256)
    ids=list(range(256))
    for step in range(151):
        push(b,ids,[step]*256,step==150)
    selected=b.reserve_pre_collision(torch.arange(256),generator=torch.Generator().manual_seed(3))
    assert selected.requested_undo.min()==100
    assert selected.requested_undo.max()==150


def test_reservation_cannot_be_committed_in_another_ring_with_matching_counters():
    first=ring(capacity=5,undo=(1,4)); second=ring(capacity=5,undo=(1,4))
    for b in (first,second):
        push(b,[0],[0]); push(b,[0],[1],True)
    selected=first.reserve_pre_collision(torch.tensor([0]),undo_steps=1)
    second.reserve_pre_collision(torch.tensor([0]),undo_steps=1)
    with pytest.raises(ValueError,match="stale"):
        second.acknowledge_restore(selected)


def test_overwrite_is_stale_even_if_bad_producer_reuses_step_id():
    b=ring(capacity=2,undo=(1,1))
    push(b,[0],[0]); push(b,[0],[1],True)
    selected=b.reserve_pre_collision(torch.tensor([0]),undo_steps=1)
    push(b,[0],[0])
    with pytest.raises(ValueError,match="stale"):
        b.acknowledge_restore(selected)


def test_cancellation_history_is_bounded_per_row_across_retries_and_episodes():
    b=ring(n=2,capacity=2,undo=(1,1))
    for env in (0,1):
        push(b,[env],[0]); push(b,[env],[1],True)
    neighbor=b.reserve_pre_collision(torch.tensor([1]),undo_steps=1)
    b.cancel_restore(neighbor,'neighbor cancellation')
    for attempt in range(1000):
        selected=b.reserve_pre_collision(torch.tensor([0]),undo_steps=1)
        assert selected.env_ids.tolist()==[0]
        b.cancel_restore(selected,'retry %d' % attempt)
    assert len(b.cancel_reasons)==2
    assert b.cancel_reasons[(1,1)]=='neighbor cancellation'
    assert b.cancel_reasons[(0,1000)]=='retry 999'
    assert b.last_cancellations[0].episode_id==0
    assert b.last_cancellations[0].token==1000
    b.begin_episode(torch.tensor([0]),task_generations=torch.tensor([8]))
    push(b,[0],[0]); push(b,[0],[1],True)
    selected=b.reserve_pre_collision(torch.tensor([0]),undo_steps=1)
    b.cancel_restore(selected,'next episode')
    assert len(b.cancel_reasons)==len(b.last_cancellations)==2
    assert (0,1000) not in b.cancel_reasons
    assert b.last_cancellations[0].episode_id==1
    assert b.last_cancellations[0].task_generation==8
    assert b.last_cancellations[0].token==1001
    retry=b.reserve_pre_collision(torch.tensor([0]),undo_steps=1)
    with pytest.raises(ValueError,match='stale'):
        b.cancel_restore(selected,'old token must not clear current retry')
    assert b.pending_token[0]==retry.tokens[0]
    b.acknowledge_restore(retry)
    assert b.pending_token[0]==-1
