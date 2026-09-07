import pytest

from rsl_rl.runners.on_policy_runner import resolve_algorithm_class, resolve_policy_class


@pytest.mark.parametrize("resolver, name", [(resolve_policy_class, "ActorCritic"),
                                           (resolve_policy_class, "DifferentiableSafeActorCritic"),
                                           (resolve_algorithm_class, "PPO")])
def test_exact_registry(resolver, name):
    assert resolver(name).__name__ == name


@pytest.mark.parametrize("resolver", [resolve_policy_class, resolve_algorithm_class])
def test_registry_rejects_expression_without_execution(tmp_path, resolver):
    marker = tmp_path / "executed"
    with pytest.raises(ValueError, match="unknown .* class"):
        resolver("__import__('pathlib').Path(%r).touch()" % str(marker))
    assert not marker.exists()
    for bad in (None, [], "ppo", " ActorCritic"):
        with pytest.raises(ValueError):
            resolver(bad)
