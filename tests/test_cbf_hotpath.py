"""Checked CBF operator budgets and actual actor/export boundary regressions."""
from collections import Counter
import ast
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.cbf_actor_critic import DifferentiableSafeActorCritic
from rsl_rl.modules.cbf_lse_layer import ExactLSECBFLayer
from sea_nav_current_isaaclab_full_method.adapters.cbf_shield import FootprintAwareLSECBFLayer


VARIANTS = ("core", "adapter", "footprint_ablation")
ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests/fixtures/cbf_paper_damped_v1.json").read_text())


def make_layer(kind, **kwargs):
    if kind == "core":
        return ExactLSECBFLayer(**kwargs)
    if kind == "footprint_ablation":
        kwargs.update(footprint_radius_m=.55, algorithm_profile="ablation/hotpath")
    return FootprintAwareLSECBFLayer(**kwargs)


def valid_inputs(batch=2):
    return [torch.zeros(batch, 3), torch.ones(batch, 41), torch.ones(batch, 1)]


class LayerOperators(TorchDispatchMode):
    def __init__(self, active=True):
        super().__init__()
        self.active = active
        self.counts = Counter()
        self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if self.active:
            self.counts[str(func)] += 1
        return func(*args, **(kwargs or {}))

    def start(self, module, inputs):
        self.active = True
        self.calls += 1

    def stop(self, module, inputs, output):
        self.active = False

    def budget(self):
        return {name: self.counts[op] for name, op in (
            ("scalar", "aten._local_scalar_dense.default"),
            ("sum", "aten.sum.dim_IntList"),
            ("norm", "aten.linalg_vector_norm.default"),
            ("min", "aten.min.dim"))}


@pytest.mark.parametrize("kind", VARIANTS)
@pytest.mark.parametrize("batch", (2, 2048))
def test_ordinary_operator_budget(kind, batch):
    layer = make_layer(kind)
    args = valid_inputs(batch)
    probe = LayerOperators()
    with probe:
        output = layer(*args)
    diagnostic_probe = LayerOperators()
    with diagnostic_probe:
        diagnostic_output, _ = layer.forward_with_diagnostics(*args)
    torch.testing.assert_close(output, diagnostic_output, atol=0, rtol=0)
    assert probe.budget() == {"scalar": 1, "sum": 3, "norm": 0, "min": 0}
    assert diagnostic_probe.budget() == {"scalar": 1, "sum": 4, "norm": 1,
                                         "min": 0 if kind == "core" else 2}


@pytest.mark.parametrize("kind", VARIANTS)
def test_real_actor_ppo_layer_call_budget(kind):
    torch.manual_seed(1729)
    actor = DifferentiableSafeActorCritic(
        3, actor_hidden_dims=[16], critic_hidden_dims=[16], encoder_hidden_dims=[16])
    actor.cbf_layer = make_layer(kind)
    ppo = PPO(actor, num_learning_epochs=2, num_mini_batches=2,
              schedule="fixed", desired_kl=None)
    ppo.init_storage(2, 2, (550,), (3,))
    probe = LayerOperators(active=False)
    before = actor.cbf_layer.register_forward_pre_hook(probe.start)
    after = actor.cbf_layer.register_forward_hook(probe.stop)
    try:
        obs = torch.zeros(2, 550)
        with probe, torch.no_grad():
            for _ in range(2):
                ppo.act(obs, obs)
                ppo.process_env_step(obs, torch.tensor([1., -1.]),
                                     torch.zeros(2, dtype=torch.bool), {})
            ppo.compute_returns(obs)
        collection = (probe.calls, probe.budget())
        probe.calls = 0
        probe.counts.clear()
        with probe:
            metrics = ppo.update()
        assert all(torch.isfinite(torch.tensor(metrics)))
        assert collection == (2, {"scalar": 2, "sum": 6, "norm": 0, "min": 0})
        assert probe.calls == 8
        assert probe.budget() == {"scalar": 8, "sum": 24, "norm": 0, "min": 0}
    finally:
        before.remove()
        after.remove()


def legacy_validate(layer, u_bar, lidar_dists, alpha):
    """Frozen a453b15 short-circuit boundary, independent of new validators."""
    if u_bar.dim() != 2 or u_bar.size(1) != 3 or u_bar.size(0) == 0:
        raise ValueError("u_bar must be nonempty [B,3]")
    if lidar_dists.dim() != 2 or lidar_dists.size(1) != layer.num_rays or lidar_dists.size(0) != u_bar.size(0):
        raise ValueError("lidar_dists must be [B,num_rays] with matching batch")
    if alpha.dim() != 2 or alpha.size(1) != 1 or alpha.size(0) != u_bar.size(0):
        raise ValueError("alpha must be [B,1] with matching batch")
    if not torch.isfinite(u_bar).all() or not torch.isfinite(lidar_dists).all() or not torch.isfinite(alpha).all():
        raise ValueError("CBF inputs must be finite")
    if (lidar_dists <= 0).any():
        raise ValueError("lidar_dists must be positive before preprocessing")
    if (alpha <= 0).any():
        raise ValueError("alpha must be already positive; transform raw logits once")


def legacy_call(layer, args, api):
    # The numeric oracle is the existing golden fixture; this oracle isolates
    # legacy validation order and preprocessing, including downstream errors.
    legacy_validate(layer, *args)
    u, rays, alpha = args
    effective = rays
    if getattr(layer, "footprint_radius_m", 0.) > 0:
        effective = (rays - layer.footprint_radius_m).clamp_min(layer.min_effective_clearance_m)
    output, diag = layer._compute(u, effective, alpha)
    if isinstance(layer, FootprintAwareLSECBFLayer):
        diag["ray_min_raw"] = rays.min(dim=1, keepdim=True).values
        diag["ray_min_effective"] = effective.min(dim=1, keepdim=True).values
    return output if api == "forward" else (output, diag)


def outcome(call):
    try:
        return None, call()
    except Exception as exc:
        return type(exc), str(exc)


def replaced(index, value):
    args = valid_inputs()
    args[index] = value
    return args


def invalid_cases():
    cases = []
    for index, shape in enumerate(((2, 3), (2, 41), (2, 1))):
        for value in (float("nan"), float("inf")):
            cases.append(("nonfinite_{}_{}".format(index, value), replaced(index, torch.full(shape, value))))
        if index:
            for value in (0., -1.):
                cases.append(("nonpositive_{}_{}".format(index, value), replaced(index, torch.full(shape, value))))
    cases.extend([
        ("u_width", replaced(0, torch.zeros(2, 2))),
        ("u_rank", replaced(0, torch.zeros(3))),
        ("empty", [torch.zeros(0, 3), torch.ones(0, 41), torch.ones(0, 1)]),
        ("ray_width", replaced(1, torch.ones(2, 40))),
        ("ray_batch", replaced(1, torch.ones(1, 41))),
        ("ray_rank", replaced(1, torch.ones(41))),
        ("alpha_rank", replaced(2, torch.ones(2))),
        ("alpha_batch", replaced(2, torch.ones(1, 1))),
        ("finite_before_positive", [torch.zeros(2, 3), torch.zeros(2, 41), torch.full((2, 1), float("nan"))]),
        ("rays_before_alpha", [torch.zeros(2, 3), torch.zeros(2, 41), torch.zeros(2, 1)]),
        ("shape_before_nonfinite", [torch.full((2, 3), float("nan")), torch.ones(2, 40), torch.ones(2, 1)]),
    ])
    return cases


def unusual_cases():
    cases = []
    for index, value in enumerate(valid_inputs()):
        for kind, replacement in (
            ("complex", value.to(torch.complex64)),
            ("sparse", value.to_sparse()),
            ("quantized", torch.quantize_per_tensor(value, .1, 0, torch.quint8)),
            ("meta", value.to("meta")),
        ):
            args = replaced(index, replacement)
            cases.append(("{}_{}".format(kind, index), args))
            if index:
                earlier_nan = list(args)
                earlier_nan[0] = torch.full((2, 3), float("nan"))
                cases.append(("nan_u_{}_{}".format(kind, index), earlier_nan))
    cases.append(("all_meta", [value.to("meta") for value in valid_inputs()]))
    for dtype in (torch.bool, torch.int64, torch.float64):
        cases.append((str(dtype), replaced(1, torch.ones(2, 41, dtype=dtype))))
    # Optional newer dtypes are not assumed to support isfinite/comparisons.
    # A closed acceleration set must fall back, not reject these inputs itself.
    for dtype_name in ("float8_e4m3fn", "float8_e5m2", "uint16", "uint32", "uint64"):
        dtype = getattr(torch, dtype_name, None)
        if dtype is not None:
            args = replaced(1, torch.ones(2, 41).to(dtype))
            cases.append((dtype_name, args))
            args = list(args)
            args[0] = torch.full((2, 3), float("nan"))
            cases.append(("nan_u_" + dtype_name, args))
    return cases


@pytest.mark.parametrize("kind", VARIANTS)
@pytest.mark.parametrize("api", ("forward", "forward_with_diagnostics"))
def test_eager_exact_legacy_errors_and_dtype_acceptance(kind, api):
    layer = make_layer(kind)
    for name, args in invalid_cases() + unusual_cases():
        expected = outcome(lambda: legacy_call(layer, args, api))
        actual = outcome(lambda: getattr(layer, api)(*args))
        assert actual[0] == expected[0], name
        if expected[0] is not None:
            assert actual[1] == expected[1], name
        else:
            torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0, msg=name)
    # A real native meta buffer mismatch is not CUDA hardware evidence.
    layer = make_layer(kind).to("meta")
    args = valid_inputs()
    assert outcome(lambda: getattr(layer, api)(*args)) == outcome(lambda: legacy_call(layer, args, api))


@pytest.mark.parametrize("kind", VARIANTS)
def test_fallback_and_metadata_operator_costs(kind):
    layer = make_layer(kind)
    # Failed aggregation adds one decision before the original ordered checks.
    for index, value, expected_scalars in (
        (0, torch.full((2, 3), float("nan")), 2),
        (1, torch.zeros(2, 41), 5),
        (2, torch.zeros(2, 1), 6),
        (1, torch.zeros(2, 40), 0),
    ):
        probe = LayerOperators()
        with probe, pytest.raises(ValueError):
            layer(*replaced(index, value))
        assert probe.budget()["scalar"] == expected_scalars
        assert probe.budget()["sum"] == 0
    for name, args in unusual_cases():
        if not name.startswith("nan_u_"):
            continue
        probe = LayerOperators()
        with probe, pytest.raises(ValueError, match="^CBF inputs must be finite$"):
            layer(*args)
        # Direct legacy fallback: only u's predicate, not aggregation first.
        assert probe.budget()["scalar"] == 1, name
        assert probe.counts["aten.bitwise_and.Tensor"] == 0, name


def script_roundtrip(layer):
    stream = io.BytesIO()
    torch.jit.save(torch.jit.script(layer), stream)
    stream.seek(0)
    return torch.jit.load(stream)


@pytest.mark.parametrize("kind", VARIANTS)
def test_actual_script_save_load_both_checked_apis(kind):
    layer = make_layer(kind)
    loaded = script_roundtrip(layer)
    assert list(loaded.state_dict()) == list(layer.state_dict()) == ["ray_unit_vectors"]
    fresh = make_layer(kind)
    fresh.load_state_dict(loaded.state_dict(), strict=True)
    for api in ("forward", "forward_with_diagnostics"):
        torch.testing.assert_close(getattr(loaded, api)(*valid_inputs()),
                                   getattr(layer, api)(*valid_inputs()), atol=0, rtol=0)
        for name, args in invalid_cases():
            error_type, message = outcome(lambda: legacy_call(layer, args, api))
            assert error_type is ValueError, name
            with pytest.raises(torch.jit.Error) as error:
                getattr(loaded, api)(*args)
            assert str(error.value).splitlines()[-1] == "builtins.ValueError: " + message, name
    graph = str(loaded._c._get_method("_validate_inputs").graph)
    assert graph.count("aten::Bool") == 1
    assert "_assert_async" not in str(loaded.inlined_graph)
    ordinary_graph = str(loaded.inlined_graph)
    assert "aten::norm" not in ordinary_graph
    assert "aten::min(" not in ordinary_graph


@pytest.mark.parametrize("kind", VARIANTS)
@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["id"])
def test_golden_ordinary_diagnostic_gradient_equivalence(kind, case):
    layer = make_layer(kind, fov_deg=case["fov_deg"], **FIXTURE["parameters"])
    rows = []
    for api in ("forward", "forward_with_diagnostics"):
        args = [torch.tensor([value], dtype=torch.float32, requires_grad=True)
                for value in (case["u_bar"], case["rays_m"], [case["alpha"]])]
        result = getattr(layer, api)(*args)
        output = result if api == "forward" else result[0]
        grads = torch.autograd.grad(output.square().sum(), args, retain_graph=True)
        rows.append((output, grads))
        if api == "forward_with_diagnostics":
            diag = result[1]
            assert set(diag) == {"h_comp", "Lg_h", "Lg_norm_sq", "r", "eta_raw", "eta",
                                 "correction_norm", "residual_before", "residual_after"} | (
                                     set() if kind == "core" else {"ray_min_raw", "ray_min_effective"})
            if kind != "footprint_ablation":
                for key, want in case["expected"].items():
                    torch.testing.assert_close(output if key == "u_s" else diag[key],
                                               torch.tensor([want], dtype=torch.float32),
                                               atol=FIXTURE["atol"], rtol=FIXTURE["rtol"])
            assert all(value.requires_grad for value in diag.values())
            diag_grads = torch.autograd.grad(sum(value.square().sum() for value in diag.values()), args)
            assert all(torch.isfinite(value).all() for value in diag_grads)
            expected_residual = torch.where(diag["r"] < 0, -diag["eta"] * layer.damping_factor, diag["r"])
            torch.testing.assert_close(diag["residual_after"], expected_residual)
        assert torch.equal(output[:, 2], args[0][:, 2])
    torch.testing.assert_close(rows[0], rows[1], atol=0, rtol=0)


@pytest.mark.parametrize("kind", ("adapter", "footprint_ablation"))
def test_footprint_preprocessing_both_apis_and_gradients(kind):
    layer = make_layer(kind, min_effective_clearance_m=.01)
    core = ExactLSECBFLayer()
    rows = []
    for variant in ("explicit_core", "forward", "forward_with_diagnostics"):
        u = torch.tensor([[.8, -.2, .3]], requires_grad=True)
        rays = torch.linspace(.001, 1.0, 41).unsqueeze(0).requires_grad_()
        alpha = torch.tensor([[.7]], requires_grad=True)
        if variant == "explicit_core":
            effective = (rays - .55).clamp_min(.01) if kind == "footprint_ablation" else rays
            output = core(u, effective, alpha)
        else:
            result = getattr(layer, variant)(u, rays, alpha)
            output = result if variant == "forward" else result[0]
            if variant == "forward_with_diagnostics":
                expected_min = .01 if kind == "footprint_ablation" else .001
                torch.testing.assert_close(result[1]["ray_min_effective"], torch.tensor([[expected_min]]))
        rows.append((output, torch.autograd.grad(output.square().sum(), (u, rays, alpha))))
    torch.testing.assert_close(rows[0], rows[1], atol=0, rtol=0)
    torch.testing.assert_close(rows[0], rows[2], atol=0, rtol=0)


@pytest.mark.parametrize("kind", VARIANTS)
def test_transforms_do_not_bypass_dynamic_checks(kind):
    layer = make_layer(kind)
    for checked in (layer, script_roundtrip(layer)):
        for api in ("forward", "forward_with_diagnostics"):
            for args, message in (
                (replaced(1, torch.exp2(torch.full((2, 41), 1000.))), "CBF inputs must be finite"),
                (replaced(1, torch.exp2(torch.full((2, 41), -1000.))), "lidar_dists must be positive before preprocessing"),
                (replaced(2, torch.nn.functional.softplus(torch.full((2, 1), -1000.))), "alpha must be already positive"),
            ):
                with pytest.raises((ValueError, torch.jit.Error), match=message):
                    getattr(checked, api)(*args)


@pytest.mark.parametrize("kind", VARIANTS)
def test_actual_gym_exporter_ast_script_save_load(kind, monkeypatch):
    # Execute the real exporter definition at its real filename/line offsets;
    # no copied wrapper, tracing substitution, fake Gym or simulator import.
    source_path = ROOT / "training/legged_gym/legged_gym/utils/helpers.py"
    parsed = ast.parse(source_path.read_text(), filename=str(source_path))
    definition = next(node for node in parsed.body if isinstance(node, ast.FunctionDef)
                      and node.name == "export_policy_as_jit")
    module = types.ModuleType("cbf_hotpath_actual_exporter")
    module.__file__ = str(source_path)
    module.__dict__.update(torch=torch, copy=copy, os=os)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    isolated = ast.Module(body=[definition], type_ignores=[])
    exec(compile(isolated, str(source_path), "exec"), module.__dict__)
    torch.manual_seed(421)
    actor = DifferentiableSafeActorCritic(
        3, actor_hidden_dims=[8], critic_hidden_dims=[8], encoder_hidden_dims=[8])
    actor.cbf_layer = make_layer(kind)
    obs = torch.zeros(2, 550)
    prohibited_before = {key for key in sys.modules if key.startswith(("isaacgym", "legged_gym"))}
    with tempfile.TemporaryDirectory(prefix="sea-nav-cbf-export-") as directory:
        module.export_policy_as_jit(actor, directory, "policy.pt")
        loaded = torch.jit.load(str(Path(directory) / "policy.pt"))
        torch.testing.assert_close(loaded(obs), actor.act_inference(obs), atol=0, rtol=0)
        invalid = obs.clone()
        invalid[:, -actor.num_obs_one_step + actor.num_props] = -1000.
        with pytest.raises(torch.jit.Error, match="positive before preprocessing"):
            loaded(invalid)
        graph = str(loaded.cbf_layer.inlined_graph)
        assert "aten::norm" not in graph and "aten::min(" not in graph
    assert {key for key in sys.modules if key.startswith(("isaacgym", "legged_gym"))} == prohibited_before
