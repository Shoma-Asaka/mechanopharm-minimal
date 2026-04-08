import numpy as np

from mechanopharm_minimal.models import TwoStateModel, ThreeStateProtectionModel


def test_two_state_range():
    model = TwoStateModel()
    c = np.linspace(0, 2, 20)
    m = np.linspace(-1, 1, 20)
    C, M = np.meshgrid(c, m)
    p = model.occupancy(C, M)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_three_state_peak_metrics():
    model = ThreeStateProtectionModel()
    peak, tpeak = model.peak_metrics(c=1.0, m=0.8)
    assert peak > 0.0
    assert tpeak >= 0.0
