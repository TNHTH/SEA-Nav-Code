import torch
from torch.utils._python_dispatch import TorchDispatchMode
from test_collision_replay_cpu import ring


class Probe(TorchDispatchMode):
    def __init__(self):
        super().__init__(); self.ops=[]; self.materialized=0; self.writes=0; self.scalars=[]
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        result=func(*args,**(kwargs or {})); name=str(func); self.ops.append(name)
        materializers={"aten.index.Tensor","aten.clone.default","aten.roll.default","aten.cat.default","aten.stack.default","aten.where.self"}
        if name in materializers: self.materialized += result.numel()
        if name == "aten.index_put_.default": self.writes += args[2].numel()
        if name == "aten.index_copy_.default": self.writes += args[3].numel()
        if name == "aten._local_scalar_dense.default": self.scalars.append(name)
        return result


def test_active_row_push_cost_independent_of_capacity_and_inactive_population():
    probes=[]
    for n,capacity in [(4,8),(4,4096),(2048,8)]:
        b=ring(n=n,capacity=capacity,undo=(1,4))
        ids=torch.tensor([0,2]); root=torch.zeros(2,13); root[:,6]=1
        probe=Probe()
        with probe:
            b.push(env_ids=ids,root_state=root,dof_pos=root[:,:2],dof_vel=root[:,:2],task_state={"target":root[:,:3]},collision=torch.tensor([False,True]))
        probes.append(probe)
        # 20 compact scalars + 9 identity/metadata writes per active row.
        assert probe.writes <= 2*(20+9)
        assert probe.materialized <= 2*20
        assert not probe.scalars
        assert "aten.cat.default" not in probe.ops and "aten.stack.default" not in probe.ops
    assert len({(p.writes,p.materialized) for p in probes})==1


def test_probe_positive_controls_catch_selected_history_copy_and_scalar():
    for kind in ("clone","roll","cat","stack","where","selected"):
        volumes=[]
        for capacity in (8,4096):
            storage=torch.zeros(4,capacity,5); ids=torch.tensor([0,2]); probe=Probe()
            with probe:
                if kind=="clone": storage.clone()
                elif kind=="roll": torch.roll(storage,1,1)
                elif kind=="cat": torch.cat((storage[:,1:],storage[:,:1]),1)
                elif kind=="stack": torch.stack((storage[0],storage[2]))
                elif kind=="where": torch.where(storage>0,storage,storage)
                else: storage[ids].clone()
            volumes.append(probe.materialized)
        assert volumes[1]>volumes[0]>40
    probe=Probe()
    with probe: storage.view(-1,5); storage[:2]
    assert probe.materialized==0
    with probe: storage[0,0,0].item()
    assert probe.scalars==["aten._local_scalar_dense.default"]
