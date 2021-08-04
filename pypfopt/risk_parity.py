"""
The ``risk_parity`` houses a class of the same name which generates porfolios by adjusting asset allocation to the same risk level.
"""

import numpy as np
import pandas as pd
from . import base_optimizer, risk_models

class risk_parity(base_optimizer.BaseConvexOptimizer):
    def __init__(self,cov_matrix,returns=None):

        if returns is None and cov_matrix is None:
            raise ValueError("Either returns or cov_matrix must be provided")

        if returns is not None and not isinstance(returns, pd.DataFrame):
            raise TypeError("returns are not a dataframe")
        
        if not isinstance(cov_matrix, pd.DataFrame):
            raise TypeError("cov_matrix is not a dataframe")

        number_assets = len(cov_matrix.columns)
        tickers = list(cov_matrix.columns)

        self.cov_matrix = np.asmatrix(cov_matrix)

        self.returns=returns

        self.w0 = np.ones(number_assets) / number_assets
        self.x_t = np.ones(number_assets) / number_assets

        super().__init__(len(tickers), tickers)

    @staticmethod
    def _calculate_portfolio_var(w,cov_matrix):
        return (w*cov_matrix*w.T)[0,0]

    @staticmethod
    def _calculate_risk_contribution(w,cov_matrix):
        sigma = np.sqrt(risk_parity._calculate_portfolio_var(w,cov_matrix))
        MRC = cov_matrix*w.T
        RC = np.multiply(MRC,w.T)/sigma
        return RC
    
    @staticmethod
    def _risk_budget_objective(x,pars):
        x = np.asmatrix(x)
        cov_matrix = pars[0]
        x_t = pars[1]
        sig_p =  np.sqrt(risk_parity._calculate_portfolio_var(x,cov_matrix))
        risk_target = np.asmatrix(np.multiply(sig_p,x_t))
        asset_RC = risk_parity._calculate_risk_contribution(x,cov_matrix)
        J = sum(np.square(asset_RC-risk_target.T))[0,0]
        return J

    @staticmethod
    def _long_only_constraint(x):
        return x

    def optimize(self):
        cons = [{'type': 'ineq', 'fun': risk_parity._long_only_constraint},]
        res = self.nonconvex_objective(risk_parity._risk_budget_objective, 
                      initial_guess=self.w0, 
                      objective_args=[self.cov_matrix,self.x_t], 
                      constraints=cons,
                      weights_sum_to_one=True)
        self.set_weights(res)
        return weights
    
    def portfolio_performance(self,verbose=False,risk_free_rate=0.02,frequency=252):
        if self.returns is None:
            cov = self.cov_matrix
            mu = None
        else:
            cov = self.returns.cov() * frequency
            mu = self.returns.mean() * frequency

        return base_optimizer.portfolio_performance(
            self.weights, mu, cov, verbose, risk_free_rate
        )
