import numpy as np

from mechanopharm_minimal.models import TwoStateModel, ThreeStateProtectionModel


def test_two_state_probability_bounds():
    model = TwoStateModel()
    c = np.linspace(0.0, 2.0, 25)
    m = np.linspace(-1.0, 1.0, 25)
    C, M = np.meshgrid(c, m)
    p = model.occupancy(C, M)
    assert np.all(p >= 0.0)
    assert np.all(p <= 1.0)


def test_three_state_peak_outputs_are_nonnegative():
    model = ThreeStateProtectionModel()
    peak, tpeak = model.peak_metrics(c=1.0, m=0.8)
    assert peak >= 0.0
    assert tpeak >= 0.0
