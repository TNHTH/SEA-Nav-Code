import pytest
import torch
from rsl_rl.perception_delay import PerceptionDelayConfig, TimestampedPerception

def test_upstream_discrete_indices_and_hold_age():
    p=TimestampedPerception(PerceptionDelayConfig(),2,1,device="cpu")
    ids=torch.arange(2); gen=torch.Generator().manual_seed(9)
    p.reset(ids,0.,torch.ones(2,1),torch.zeros(2,2))
    seen=set()
    for tick in range(1,101):
        now=tick*.02
        p.push(ids,now,torch.full((2,1),float(tick)),torch.full((2,2),float(tick)),generator=gen)
        out=p.observe(ids,now,generator=gen)
        assert torch.all(out.sample_timestamp<=now+1e-8)
        assert torch.allclose(out.actual_age,now-out.sample_timestamp)
        if tick%5==0:
            ages=out.actual_age.tolist(); seen.update(round(x,2) for x in ages)
            assert all(round(x,2) in (.04,.06) for x in ages)
        if tick%5==4 and tick>5:
            assert all(round(x,2) in (.12,.14) for x in out.actual_age.tolist())
    assert seen=={.04,.06}

def test_reset_is_masked_synthetic_current_sample_until_history():
    p=TimestampedPerception(PerceptionDelayConfig(),2,1)
    ids=torch.arange(2); p.reset(ids,1.,torch.ones(2,1),torch.zeros(2,2))
    p.push(ids,1.02,torch.full((2,1),2.),torch.ones(2,2))
    before=p.observe(torch.tensor([1]),1.02)
    p.reset(torch.tensor([0]),1.03,torch.tensor([[9.]]),torch.tensor([[8.,7.]]))
    out=p.observe(ids,1.03)
    assert out.rays[:,0].tolist()==[9.,2.]
    assert out.synthetic_bootstrap.tolist()==[True,False]
    assert out.sample_timestamp[1]==before.sample_timestamp[0]
    assert out.goals[0].tolist()==[8.,7.]

@pytest.mark.parametrize("dt",[.02,.01])
def test_paper_diagnostic_latency_is_seeded_and_age_is_not_latency(dt):
    cfg=PerceptionDelayConfig(mode="paper_diagnostic",policy_dt_s=dt,acquisition_period_s=.1,
                              refresh_period_s=dt,latency_min_s=.04,latency_max_s=.08)
    def run():
        p=TimestampedPerception(cfg,2,1)
        ids=torch.arange(2); g=torch.Generator().manual_seed(12)
        p.reset(ids,0.,torch.ones(2,1),torch.zeros(2,2))
        result=[]
        for tick in range(1,41):
            now=tick*dt
            p.push(ids,now,torch.full((2,1),float(tick)),torch.zeros(2,2),generator=g)
            out=p.observe(ids,now,generator=g)
            assert torch.all(out.sample_timestamp<=now+1e-8)
            result.append(torch.stack((out.sample_timestamp,out.sampled_latency,out.actual_age)))
        return torch.stack(result)
    a=run(); assert torch.equal(a,run())
    actual=a[:,2]; latency=a[:,1]
    assert torch.any(actual>.08)
    assert torch.all((latency==0)|((latency>=.04)&(latency<=.08)))
    assert not torch.equal(actual,latency)
