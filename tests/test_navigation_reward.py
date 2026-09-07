import pytest
import torch
from rsl_rl.navigation_reward import compute_navigation_reward_terms, weight_reward_terms

NAMES = ("termination", "reach", "velocity", "clearance", "stuck", "collision", "angular")
PW = dict(zip(NAMES, [-100,10,15,15,-5,-4,-.05]))
UW = dict(zip(NAMES, [-100,10,4,5,-5,-4,-.05]))
CASES = [
    ({}, [0,0,2/3,2/3,0,0,0], [0,0,2/3,2/3,0,0,0]),
    (dict(distance=1,goal_x=0,cos_theta=0,vx=.2,min_ray=.5,dead=True,history=[0,.05],current=.05),
     [0,0,1/3,1/3,0,0,0], [0,0,1/3,.2,1,0,0]),
    (dict(distance=2,goal_x=2,vx=2,min_ray=.1), [0,0,19/9,2,0,0,0], [0,0,11/18,0,0,0,0]),
    (dict(distance=2,goal_x=2,vx=-1,min_ray=.5), [0,0,-8/9,-1,0,0,0], [0,0,1/9,0,0,0,0]),
    (dict(distance=2,goal_x=-2,cos_theta=-1,vx=1,cos_phi=0,min_ray=.5),
     [0,0,-8/9,0,0,0,0], [0,0,1/9,0,0,0,0]),
    (dict(distance=2,goal_x=0,cos_theta=0,vx=.2,min_ray=.5,dead=True,history=[0,.05,.09],current=.15),
     [0,0,1/9,.2,1,0,0], [0,0,1/9,.2,0,0,0]),
    (dict(wx=3,wy=4), [0,0,2/3,2/3,0,0,5], [0,0,2/3,2/3,0,0,25]),
    (dict(distance=0,goal_x=0,min_ray=.5), [0,1,1,1,0,0,0], [0,1,1,1,0,0,0]),
]

def inputs(**overrides):
    x=dict(distance=.5,goal_x=.5,cos_theta=1,vx=0,vy=0,wx=0,wy=0,wz=0,
           cos_phi=1,min_ray=2,dead=False,history=[0,.2],current=.2,
           terminated=False,initial=False,not_just_reset=True,generic=0,head_base=0,leg=0)
    x.update(overrides)
    h=x.pop("history"); p=x.pop("current")
    out={k:torch.tensor([v], dtype=torch.bool if isinstance(v,bool) else torch.float64) for k,v in x.items()}
    out["position_history"]=torch.tensor([[[v,0] for v in h]],dtype=torch.float64)
    out["position"]=torch.tensor([[p,0]],dtype=torch.float64)
    return out

@pytest.mark.parametrize("overrides,paper,upstream",CASES)
def test_independent_scalar_oracles(overrides,paper,upstream):
    for mode,expected in [("paper_diagnostic",paper),("upstream_fbce672c",upstream)]:
        terms=compute_navigation_reward_terms(inputs(**overrides),mode)
        assert list(terms)==list(NAMES)
        assert [t.item() for t in terms.values()]==pytest.approx(expected,abs=1e-10)

@pytest.mark.parametrize("mode,weights,raw",[("paper_diagnostic",PW,40),("upstream_fbce672c",UW,19)])
@pytest.mark.parametrize("dt",[.02,.01])
def test_one_time_dt_matches_gym_prescaled_path(mode,weights,raw,dt):
    terms=compute_navigation_reward_terms(inputs(distance=0,goal_x=0,min_ray=.5),mode)
    weighted=weight_reward_terms(terms,weights,dt)
    assert sum(weighted.values()).item()==pytest.approx(raw*dt)
    # Gym owns the once-only coefficient preparation, not the pure raw helper.
    assert sum(terms[k]*(weights[k]*dt) for k in terms).item()==pytest.approx(raw*dt)

def test_contacts_initial_mask_and_unknown_formula():
    x=inputs(vx=1,vy=2,wz=3,generic=1,head_base=2,leg=3)
    assert compute_navigation_reward_terms(x,"upstream_fbce672c")["collision"].item()==57*51
    assert compute_navigation_reward_terms(inputs(initial=True,generic=1),"upstream_fbce672c")["collision"].item()==0
    with pytest.raises(ValueError,match="formula"):
        compute_navigation_reward_terms(x,"paper_v1")
