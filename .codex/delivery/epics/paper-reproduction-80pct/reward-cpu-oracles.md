# Task 6 CPU reward scalar oracles

Date: 2026-09-07. Task-provided coordination HEAD: `172e14c`. This is a
read-only-source test-preparation note derived from the reward section of
`scientific-wiring-audit.md`; no simulator or production module was imported.

## Identity and scope

The paper-formula values below are **diagnostic-not-accepted-paper** oracles.
They may test a separately labelled paper-formula diagnostic profile, but must
not make an accepted `paper_v1` identity pass: the literal Table V action-bound
identity remains blocked, and the reward `dt` integration is a
`resolved_upstream_fallback` because the paper is silent about it.

The named-term order is
`[termination, reach, velocity, clearance, stuck, collision, angular]`.
Paper raw weights are `[-100, 10, 15, 15, -5, -4, -0.05]`; upstream raw
weights are `[-100, 10, 4, 5, -5, -4, -0.05]`. “Weighted raw” means the dot
product before any time integration.

All eight cases explicitly set `terminated=False`, `initial_step=False`, and
`not_just_reset=True`. Contacts are disabled without assuming simulator body
indices: the paper weighted contact-indicator sum is `0`, and upstream
`(generic, head/base, leg)` contact counts are `(0,0,0)`. Thus collision is
exactly zero even though `initial_step=False`. `r_min` is the minimum clipped
ray range used by the upstream clearance expression; `phi` is the already
selected smoothed most-open-ray angle. `H` is the ordered position history and
`p` is current position. All coordinates are metres and angles radians.

## Fully fixed inputs

| case | `d, goal_x, theta, cos(theta)` | `vx, vy, wx, wy, wz` | `phi, cos(phi), r_min` | `dead, H, p` |
|---|---|---|---|---|
| `d_eq_0.5` | `.5, .5, 0, 1` | `0,0,0,0,0` | `0,1,2` | `False, [(0,0),(.2,0)], (.2,0)` |
| `d_eq_1` | `1, 0, pi/2, 0` | `.2,0,0,0,0` | `0,1,.5` | `True, [(0,0),(.05,0)], (.05,0)` |
| `fast_near_wall` | `2, 2, 0, 1` | `2,0,0,0,0` | `0,1,.1` | `False, [(0,0),(.2,0)], (.2,0)` |
| `backward` | `2, 2, 0, 1` | `-1,0,0,0,0` | `0,1,.5` | `False, [(0,0),(.2,0)], (.2,0)` |
| `goal_behind` | `2, -2, pi, -1` | `1,0,0,0,0` | `pi/2,0,.5` | `False, [(0,0),(.2,0)], (.2,0)` |
| `history_anchor` | `2, 0, pi/2, 0` | `.2,0,0,0,0` | `0,1,.5` | `True, [(0,0),(.05,0),(.09,0)], (.15,0)` |
| `angular_3_4` | `.5, .5, 0, 1` | `0,0,3,4,0` | `0,1,2` | `False, [(0,0),(.2,0)], (.2,0)` |
| `dt_once` | `0, 0, computational cos=1` | `0,0,0,0,0` | `0,1,.5` | `False, [(0,0),(.2,0)], (.2,0)` |

For `dt_once`, goal direction at zero distance is immaterial because `vx=0`;
the pure-function scalar input `cos(theta)` is nevertheless fixed to `1`.

## Expected raw terms and totals

Values below are independently calculated, not produced by either reward
implementation. Decimal comparison should use the eventual helper's dtype-
appropriate tolerance; the displayed values are rounded to 12 decimal places.

| case | paper named terms | paper weighted raw | upstream named terms | upstream weighted raw | diagnostic purpose |
|---|---|---:|---|---:|---|
| `d_eq_0.5` | `[0,0,.666666666667,.666666666667,0,0,0]` | `20` | `[0,0,.666666666667,.666666666667,0,0,0]` | `6` | Strict reach gate is off at equality; both clearance expressions take their near branch. |
| `d_eq_1` | `[0,0,.333333333333,.333333333333,0,0,0]` | `10` | `[0,0,.333333333333,.2,1,0,0]` | `-2.666666666667` | Paper `d>1` clearance/stuck gates are off; upstream `d>.5` gates are on. |
| `fast_near_wall` | `[0,0,2.111111111111,2,0,0,0]` | `61.666666666667` | `[0,0,.611111111111,0,0,0]` | `2.444444444444` | Paper velocity/clearance remain uncapped; upstream velocity progress caps at `.5` and wall penalty clips clearance to zero. |
| `backward` | `[0,0,-.888888888889,-1,0,0,0]` | `-28.333333333333` | `[0,0,.111111111111,0,0,0]` | `.444444444444` | Paper preserves negative backward progress and clearance; upstream clamps forward speed to zero. |
| `goal_behind` | `[0,0,-.888888888889,0,0,0,0]` | `-13.333333333333` | `[0,0,.111111111111,0,0,0]` | `.444444444444` | Paper uses `cos(theta) vx`; upstream clamps negative goal-x alignment to zero. |
| `history_anchor` | `[0,0,.111111111111,.2,1,0,0]` | `-.333333333333` | `[0,0,.111111111111,.2,0,0,0]` | `1.444444444444` | Paper anchor displacement is `.09<.1`; upstream current-anchor movement is `.15>=.1`. |
| `angular_3_4` | `[0,0,.666666666667,.666666666667,0,0,5]` | `19.75` | `[0,0,.666666666667,.666666666667,0,0,25]` | `4.75` | Paper `hypot(3,4)=5`; upstream `3^2+4^2=25`. |
| `dt_once` | `[0,1,1,1,0,0,0]` | `40` | `[0,1,1,1,0,0,0]` | `19` | Exact one-time temporal-integration sentinel. |

The `history_anchor` metrics are explicitly:
`delta_p_max=max(||p_t-H[0]||)=.09`, while
`move_max=max(||p_t-p||)=.15`. The `d_eq_1` upstream stuck value is `1`, not
`2`: on Torch 2.6 boolean addition has boolean-OR semantics.

For `dt_once`, applying `raw_weight * policy_dt` exactly once gives:

| profile | weighted raw | `dt=.02` effective | `dt=.01` effective |
|---|---:|---:|---:|
| paper diagnostic | `40` | `.8` | `.4` |
| upstream | `19` | `.38` | `.19` |

The Gym pre-scaled-weight path must produce those same effective values without
another multiplication. Double scaling would incorrectly produce paper
`.016/.004` and upstream `.0076/.0019` at `dt=.02/.01`, respectively; raw
`40/19` is likewise wrong for an integrating consumer.

## Exact executed probes

Torch 2.6 boolean semantics were checked only in the dedicated CPU environment:

```console
$ test -x ../sea-nav-cpu-venv/bin/python && ../sea-nav-cpu-venv/bin/python - <<'PY'
import torch
print('torch', torch.__version__)
a = torch.tensor([False, False, True, True], dtype=torch.bool)
b = torch.tensor([False, True, False, True], dtype=torch.bool)
for expr, value in [
    ('a + b', a + b),
    ('a | b', a | b),
    ('(a + b).dtype', (a + b).dtype),
]:
    print(expr, value if not isinstance(value, torch.dtype) else value)
print('equal', torch.equal(a + b, a | b))
PY
torch 2.6.0+cpu
a + b tensor([False,  True,  True,  True])
a | b tensor([False,  True,  True,  True])
(a + b).dtype torch.bool
equal True
```

The independent standard-library calculation command was:

```console
$ python3 - <<'PY'
import json
from math import hypot
PW=(-100,10,15,15,-5,-4,-.05)
UW=(-100,10,4,5,-5,-4,-.05)
N=('termination','reach','velocity','clearance','stuck','collision','angular')
def b(d): return 1/(1+2*d*d)
def metrics(x):
    h=x['history']; p=x['current']
    delta=max(hypot(q[0]-h[0][0],q[1]-h[0][1]) for q in h)
    move=max(hypot(q[0]-p[0],q[1]-p[1]) for q in h)
    return delta,move
def terms(x, paper):
    d=x['d']; vx=x['vx']; vy=x['vy']; wz=x['wz']; bd=b(d); delta,move=metrics(x)
    reach=bd*(d<.5)
    if paper:
        velocity=x['cos_theta']*vx+bd
        clearance=(x['cos_phi']*vx if d>1 else bd)
        stuck=float(d>1 and delta<.1 and vx>0 and abs(wz)<1)
        angular=hypot(x['wx'],x['wy'])
    else:
        velocity=min(max(x['goal_x']/(d+1e-4),0)*max(vx,0),min(d,.5))+bd
        vplus=max(vx,0); vlim=min(x['min_ray'],.5); aphi=max(x['cos_phi'],0)
        clearance=(max(aphi*min(vplus,vlim)-.2*max(vplus-vlim,0),0) if d>.5 else bd)
        dead=x['dead']; escape=dead and (vx>0 or abs(wz)<1)
        stuck=float(x['not_just_reset'] and d>.5 and (escape or dead) and move<.1)
        stuck+=(abs(wz)+abs(min(vy,.5))+abs(min(vx,.5)))*(d<=.5)
        angular=x['wx']**2+x['wy']**2
    collision=0.0
    return [float(x['terminated']),reach,velocity,clearance,stuck,collision,angular]
def add(name,**kw):
    base=dict(terminated=False,d=.5,goal_x=.5,cos_theta=1,vx=0.,vy=0.,wx=0.,wy=0.,wz=0.,cos_phi=1,min_ray=2.,dead=False,not_just_reset=True,history=[[0.,0.],[.2,0.]],current=[.2,0.],initial_step=False,paper_contact_sum=0,upstream_contact_counts=[0,0,0])
    base.update(kw); cases.append((name,base))
cases=[]
add('d_eq_0.5')
add('d_eq_1',d=1.,goal_x=0.,cos_theta=0.,vx=.2,min_ray=.5,dead=True,history=[[0.,0.],[.05,0.]],current=[.05,0.])
add('fast_near_wall',d=2.,goal_x=2.,cos_theta=1.,vx=2.,min_ray=.1)
add('backward',d=2.,goal_x=2.,cos_theta=1.,vx=-1.,min_ray=.5)
add('goal_behind',d=2.,goal_x=-2.,cos_theta=-1.,vx=1.,cos_phi=0.,min_ray=.5)
add('history_anchor',d=2.,goal_x=0.,cos_theta=0.,vx=.2,min_ray=.5,dead=True,history=[[0.,0.],[.05,0.],[.09,0.]],current=[.15,0.])
add('angular_3_4',wx=3.,wy=4.)
add('dt_once',d=0.,goal_x=0.,cos_theta=1.,min_ray=.5)
for name,x in cases:
    out={'case':name,'inputs':x}
    for label,w,paper in [('paper',PW,True),('upstream',UW,False)]:
        t=terms(x,paper); raw=sum(a*c for a,c in zip(w,t))
        out[label]={'terms':dict(zip(N,[round(v,12) for v in t])),'weighted_raw':round(raw,12)}
        if name=='dt_once': out[label]['effective']={'.02':round(raw*.02,12),'.01':round(raw*.01,12)}
    print(json.dumps(out,separators=(',',':')))
PY
```

Exact output (one JSON object per case):

```json
{"case":"d_eq_0.5","inputs":{"terminated":false,"d":0.5,"goal_x":0.5,"cos_theta":1,"vx":0.0,"vy":0.0,"wx":0.0,"wy":0.0,"wz":0.0,"cos_phi":1,"min_ray":2.0,"dead":false,"not_just_reset":true,"history":[[0.0,0.0],[0.2,0.0]],"current":[0.2,0.0],"initial_step":false,"paper_contact_sum":0,"upstream_contact_counts":[0,0,0]},"paper":{"terms":{"termination":0.0,"reach":0.0,"velocity":0.666666666667,"clearance":0.666666666667,"stuck":0.0,"collision":0.0,"angular":0.0},"weighted_raw":20.0},"upstream":{"terms":{"termination":0.0,"reach":0.0,"velocity":0.666666666667,"clearance":0.666666666667,"stuck":0.0,"collision":0.0,"angular":0.0},"weighted_raw":6.0}}
{"case":"d_eq_1","inputs":{"terminated":false,"d":1.0,"goal_x":0.0,"cos_theta":0.0,"vx":0.2,"vy":0.0,"wx":0.0,"wy":0.0,"wz":0.0,"cos_phi":1,"min_ray":0.5,"dead":true,"not_just_reset":true,"history":[[0.0,0.0],[0.05,0.0]],"current":[0.05,0.0],"initial_step":false,"paper_contact_sum":0,"upstream_contact_counts":[0,0,0]},"paper":{"terms":{"termination":0.0,"reach":0.0,"velocity":0.333333333333,"clearance":0.333333333333,"stuck":0.0,"collision":0.0,"angular":0.0},"weighted_raw":10.0},"upstream":{"terms":{"termination":0.0,"reach":0.0,"velocity":0.333333333333,"clearance":0.2,"stuck":1.0,"collision":0.0,"angular":0.0},"weighted_raw":-2.666666666667}}
{"case":"fast_near_wall","inputs":{"terminated":false,"d":2.0,"goal_x":2.0,"cos_theta":1.0,"vx":2.0,"vy":0.0,"wx":0.0,"wy":0.0,"wz":0.0,"cos_phi":1,"min_ray":0.1,"dead":false,"not_just_reset":true,"history":[[0.0,0.0],[0.2,0.0]],"current":[0.2,0.0],"initial_step":false,"paper_contact_sum":0,"upstream_contact_counts":[0,0,0]},"paper":{"terms":{"termination":0.0,"reach":0.0,"velocity":2.111111111111,"clearance":2.0,"stuck":0.0,"collision":0.0,"angular":0.0},"weighted_raw":61.666666666667},"upstream":{"terms":{"termination":0.0,"reach":0.0,"velocity":0.611111111111,"clearance":0,"stuck":0.0,"collision":0.0,"angular":0.0},"weighted_raw":2.444444444444}}
{"case":"backward","inputs":{"terminated":false,"d":2.0,"goal_x":2.0,"cos_theta":1.0,"vx":-1.0,"vy":0.0,"wx":0.0,"wy":0.0,"wz":0.0,"cos_phi":1,"min_ray":0.5,"dead":false,"not_just_reset":true,"history":[[0.0,0.0],[0.2,0.0]],"current":[0.2,0.0],"initial_step":false,"paper_contact_sum":0,"upstream_contact_counts":[0,0,0]},"paper":{"terms":{"termination":0.0,"reach":0.0,"velocity":-0.888888888889,"clearance":-1.0,"stuck":0.0,"collision":0.0,"angular":0.0},"weighted_raw":-28.333333333333},"upstream":{"terms":{"termination":0.0,"reach":0.0,"velocity":0.111111111111,"clearance":0.0,"stuck":0.0,"collision":0.0,"angular":0.0},"weighted_raw":0.444444444444}}
{"case":"goal_behind","inputs":{"terminated":false,"d":2.0,"goal_x":-2.0,"cos_theta":-1.0,"vx":1.0,"vy":0.0,"wx":0.0,"wy":0.0,"wz":0.0,"cos_phi":0.0,"min_ray":0.5,"dead":false,"not_just_reset":true,"history":[[0.0,0.0],[0.2,0.0]],"current":[0.2,0.0],"initial_step":false,"paper_contact_sum":0,"upstream_contact_counts":[0,0,0]},"paper":{"terms":{"termination":0.0,"reach":0.0,"velocity":-0.888888888889,"clearance":0.0,"stuck":0.0,"collision":0.0,"angular":0.0},"weighted_raw":-13.333333333333},"upstream":{"terms":{"termination":0.0,"reach":0.0,"velocity":0.111111111111,"clearance":0,"stuck":0.0,"collision":0.0,"angular":0.0},"weighted_raw":0.444444444444}}
{"case":"history_anchor","inputs":{"terminated":false,"d":2.0,"goal_x":0.0,"cos_theta":0.0,"vx":0.2,"vy":0.0,"wx":0.0,"wy":0.0,"wz":0.0,"cos_phi":1,"min_ray":0.5,"dead":true,"not_just_reset":true,"history":[[0.0,0.0],[0.05,0.0],[0.09,0.0]],"current":[0.15,0.0],"initial_step":false,"paper_contact_sum":0,"upstream_contact_counts":[0,0,0]},"paper":{"terms":{"termination":0.0,"reach":0.0,"velocity":0.111111111111,"clearance":0.2,"stuck":1.0,"collision":0.0,"angular":0.0},"weighted_raw":-0.333333333333},"upstream":{"terms":{"termination":0.0,"reach":0.0,"velocity":0.111111111111,"clearance":0.2,"stuck":0.0,"collision":0.0,"angular":0.0},"weighted_raw":1.444444444444}}
{"case":"angular_3_4","inputs":{"terminated":false,"d":0.5,"goal_x":0.5,"cos_theta":1,"vx":0.0,"vy":0.0,"wx":3.0,"wy":4.0,"wz":0.0,"cos_phi":1,"min_ray":2.0,"dead":false,"not_just_reset":true,"history":[[0.0,0.0],[0.2,0.0]],"current":[0.2,0.0],"initial_step":false,"paper_contact_sum":0,"upstream_contact_counts":[0,0,0]},"paper":{"terms":{"termination":0.0,"reach":0.0,"velocity":0.666666666667,"clearance":0.666666666667,"stuck":0.0,"collision":0.0,"angular":5.0},"weighted_raw":19.75},"upstream":{"terms":{"termination":0.0,"reach":0.0,"velocity":0.666666666667,"clearance":0.666666666667,"stuck":0.0,"collision":0.0,"angular":25.0},"weighted_raw":4.75}}
{"case":"dt_once","inputs":{"terminated":false,"d":0.0,"goal_x":0.0,"cos_theta":1.0,"vx":0.0,"vy":0.0,"wx":0.0,"wy":0.0,"wz":0.0,"cos_phi":1,"min_ray":0.5,"dead":false,"not_just_reset":true,"history":[[0.0,0.0],[0.2,0.0]],"current":[0.2,0.0],"initial_step":false,"paper_contact_sum":0,"upstream_contact_counts":[0,0,0]},"paper":{"terms":{"termination":0.0,"reach":1.0,"velocity":1.0,"clearance":1.0,"stuck":0.0,"collision":0.0,"angular":0.0},"weighted_raw":40.0,"effective":{".02":0.8,".01":0.4}},"upstream":{"terms":{"termination":0.0,"reach":1.0,"velocity":1.0,"clearance":1.0,"stuck":0.0,"collision":0.0,"angular":0.0},"weighted_raw":19.0,"effective":{".02":0.38,".01":0.19}}}
```
