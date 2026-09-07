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

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('dt',[.02,.01])
@pytest.mark.parametrize('mode',['discrete_history_sample_and_hold','paper_diagnostic'])
def test_long_tensor_clock_cadence_and_nonzero_masked_epoch(dtype,dt,mode):
    acquisition=dt if mode=='discrete_history_sample_and_hold' else .1
    cfg=PerceptionDelayConfig(mode=mode,policy_dt_s=dt,acquisition_period_s=acquisition,
                              refresh_period_s=.1 if mode.startswith('discrete') else dt)
    p=TimestampedPerception(cfg,2,1)
    ids=torch.arange(2); p.reset(ids,0.,torch.zeros(2,1),torch.zeros(2,2))
    g=torch.Generator().manual_seed(9)
    counts=torch.zeros(2,dtype=torch.long)
    reset_tick=337
    for tick in range(1,1001):
        now=torch.full((2,),tick,dtype=dtype)*dt
        if tick==reset_tick:
            before=p.times[1].clone()
            p.reset(ids[:1],now[:1],torch.full((1,1),float(tick)),torch.full((1,2),float(tick)))
            assert torch.equal(p.times[1],before)
            counts[0]=0
        old=p.write.clone()
        p.push(ids,now,torch.full((2,1),float(tick)),torch.full((2,2),float(tick)),generator=g)
        counts+=(p.write!=old).long()
        out=p.observe(ids,now,generator=g)
        for row in range(2):
            age_tick=tick-(reset_tick if row==0 and tick>=reset_tick else 0)
            assert counts[row].item()==age_tick//round(acquisition/dt)
            if mode.startswith('discrete') and age_tick>=round(.1/dt):
                since_refresh=age_tick%round(.1/dt)
                allowed=((2+since_refresh)*dt,(3+since_refresh)*dt)
                assert min(abs(out.actual_age[row].item()-a) for a in allowed)<1e-5
                sample_tick=int(out.rays[row,0].item())
                assert tick-sample_tick in (2+since_refresh,3+since_refresh)
                assert out.goals[row,0].item()==sample_tick

def test_clock_no_host_scalar_extraction_and_nonfinite_rejection():
    from torch.utils._python_dispatch import TorchDispatchMode
    class Ops(TorchDispatchMode):
        def __init__(self): self.names=[]
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            self.names.append(str(func)); return func(*args,**(kwargs or {}))
    p=TimestampedPerception(PerceptionDelayConfig(),2,1); ids=torch.arange(2)
    p.reset(ids,0.,torch.ones(2,1),torch.zeros(2,2))
    with Ops() as ops:
        p.push(ids,torch.tensor([.02,.02]),torch.ones(2,1),torch.zeros(2,2))
        p.observe(ids,torch.tensor([.02,.02]))
    assert not any('_local_scalar_dense' in name for name in ops.names)
    with pytest.raises((ValueError,RuntimeError),match='finite'):
        p.push(ids,float('nan'),torch.ones(2,1),torch.zeros(2,2))
