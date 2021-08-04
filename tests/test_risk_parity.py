import numpy as np
import pandas as pd

import pytest
from pypfopt_test import risk_parity
from pypfopt_test import risk_models
from tests.utilities_for_tests import get_data, resource


def test_risk_parity_errors():
    with pytest.raises(TypeError):
        rp = risk_parity()

    df = get_data()
    returns = df.pct_change().dropna(how="all")
    S = risk_models.sample_cov(df)

    S_np = S.to_numpy()
    with pytest.raises(TypeError):
        rp = risk_parity(S_np)

    returns_np = returns.to_numpy()
    with pytest.raises(TypeError):
        rp = risk_parity(S,returns=returns_np)


def test_risk_parity_portfolio():
    df = get_data()
    returns = df.pct_change().dropna(how="all")
    S = risk_models.sample_cov(df)

    rp = risk_parity(cov_matrix=S)
    with pytest.raises(ValueError):
        rp.portfolio_performance()
    w = rp.optimize()
    p = rp.portfolio_performance()
    assert p[0] == p[2] == None
    np.testing.assert_almost_equal(p[1],0.19285771548743316)

    rp = risk_parity(returns=returns,cov_matrix=S)
    with pytest.raises(ValueError):
        rp.portfolio_performance()
    w = rp.optimize()
    np.testing.assert_allclose(
        rp.portfolio_performance(),
        (0.22932220285615676, 0.19285771548743316, 1.0853711625024225),
    )

def test_weights():
    df = get_data()
    returns = df.pct_change().dropna(how="all")
    S = risk_models.sample_cov(df)
    rp = risk_parity(cov_matrix=S)
    w = rp.optimize()
    # uncomment this line if you want generating a new file
    #pd.Series(w).to_csv(resource("weights_rp.csv"))
    x = pd.read_csv(resource("weights_rp.csv"), squeeze=True, index_col=0)
    pd.testing.assert_series_equal(x, pd.Series(w), check_names=False, rtol=1e-2)

    assert isinstance(w, dict)
    assert set(w.keys()) == set(df.columns)
    np.testing.assert_almost_equal(sum(w.values()), 1)
    assert all([i >= 0 for i in w.values()])

def test_risk_parity_functions():
    df = get_data()
    S = np.asmatrix(risk_models.sample_cov(df[['GOOG','AAPL','FB','AMZN']]))
    w = np.asmatrix(np.ones(4) / 4)
    w_t = np.asmatrix([0.4,0.1,0.4,0.1])

    np.testing.assert_almost_equal(risk_parity._calculate_portfolio_var(w,S),0.08443887625739224)
    np.testing.assert_array_almost_equal(risk_parity._calculate_risk_contribution(w,S),np.array([[0.04833427],
    [0.07617729],
    [0.0486168 ],
    [0.11745532]]))
    np.testing.assert_almost_equal(risk_parity._risk_budget_objective(w,[S,w_t]),0.0192165287933482)