import pytest
import torch

from sea_nav_core import paper_v1_lreg_loss, paper_v1_shield_loss


def test_shield_golden_both_terms_have_point_one_coefficient():
    nominal = torch.zeros((2, 2), dtype=torch.float64)
    biased = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    alpha = torch.tensor([[0.05], [0.2]], dtype=torch.float64)
    # Mean per-row squared norm is 15; mean squared alpha deficit is .00125.
    assert paper_v1_shield_loss(nominal, biased, alpha).item() == pytest.approx(1.500125)


def lreg_inputs():
    mean = torch.tensor([[2.0, -2.0], [0.0, 0.0]], dtype=torch.float64)
    lower = torch.tensor([-1.0, -1.0], dtype=torch.float64)
    upper = -lower
    perturbed_mean = mean + 2
    value = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    return mean, lower, upper, perturbed_mean, value, value + 3


def test_lreg_golden_range_sum_and_mse_reductions_and_no_double_weight():
    # range=1, actor_MSE=4, critic_MSE=9.
    assert paper_v1_lreg_loss(*lreg_inputs()).item() == pytest.approx(1.245)


def test_equal_output_zero_losses_without_parameter_side_effects():
    mean = torch.zeros((2, 2), dtype=torch.float64)
    value = torch.zeros((2, 1), dtype=torch.float64)
    assert paper_v1_shield_loss(mean, mean.clone(), torch.ones_like(value)).item() == 0
    assert paper_v1_lreg_loss(mean, -torch.ones(2, dtype=torch.float64),
                            torch.ones(2, dtype=torch.float64), mean.clone(), value, value.clone()).item() == 0


def test_both_loss_helpers_have_gradients_and_match_script():
    nominal = torch.zeros((2, 2), dtype=torch.float64, requires_grad=True)
    biased = torch.ones((2, 2), dtype=torch.float64, requires_grad=True)
    alpha = torch.full((2, 1), 0.05, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(paper_v1_shield_loss, (nominal, biased, alpha))
    shield = paper_v1_shield_loss(nominal, biased, alpha)
    shield.backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (nominal, biased, alpha))
    torch.testing.assert_close(torch.jit.script(paper_v1_shield_loss)(nominal, biased, alpha), shield)
    args = lreg_inputs()
    for index in (0, 3, 4, 5):
        args[index].requires_grad_()
    assert torch.autograd.gradcheck(paper_v1_lreg_loss, args)
    loss = paper_v1_lreg_loss(*args)
    loss.backward()
    assert all(args[i].grad is not None and torch.isfinite(args[i].grad).all() for i in (0, 3, 4, 5))
    torch.testing.assert_close(torch.jit.script(paper_v1_lreg_loss)(*args), loss)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_alpha_is_not_silently_transformed(bad):
    with pytest.raises(ValueError):
        paper_v1_shield_loss(torch.ones(2, 2), torch.ones(2, 2), torch.full((2, 1), bad))


@pytest.mark.parametrize("index", range(6))
def test_lreg_nonfinite_rejection(index):
    args = list(lreg_inputs())
    args[index].view(-1)[0] = float("nan")
    with pytest.raises(ValueError):
        paper_v1_lreg_loss(*args)


def test_loss_shape_batch_dtype_bounds_rejection():
    for biased in (torch.ones(3, 2), torch.ones(2, 3), torch.ones(2, 2, dtype=torch.float64)):
        with pytest.raises(ValueError):
            paper_v1_shield_loss(torch.ones(2, 2), biased, torch.ones(2, 1))
    for index, replacement in ((0, torch.empty(0, 2)), (1, torch.ones(2, 1)),
                                (2, torch.tensor([-2.0, -2.0], dtype=torch.float64)),
                                (4, torch.ones(3, 1, dtype=torch.float64))):
        args = list(lreg_inputs())
        args[index] = replacement
        with pytest.raises(ValueError):
            paper_v1_lreg_loss(*args)


def test_loss_overflow_does_not_return_infinite_objective():
    with pytest.raises(ValueError, match="overflow"):
        paper_v1_shield_loss(torch.zeros(2, 2), torch.full((2, 2), 1e30), torch.ones(2, 1))
    args = list(lreg_inputs())
    args[0].fill_(1e300)
    with pytest.raises(ValueError, match="overflow"):
        paper_v1_lreg_loss(*args)
