import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc  # noqa: E402


class TestMonteCarlo(unittest.TestCase):
    def test_gbm_mean_matches_theory(self):
        # E[prod(1+r)] = exp(T*(mu + sigma^2/2)) for lognormal returns
        mu, sigma, months = 0.006, 0.045, 120
        r = mc.gbm_returns(mu, sigma, months, 200_000, np.random.default_rng(1))
        w = mc.simulate(r, 1.0, 0.0)[:, -1]
        expected = np.exp(months * (mu + sigma ** 2 / 2))
        self.assertAlmostEqual(w.mean() / expected, 1.0, delta=0.01)

    def test_bootstrap_uses_only_history(self):
        hist = np.array([0.01, -0.02, 0.03, 0.00, 0.05])
        r = mc.bootstrap_returns(hist, 13, 100, np.random.default_rng(2), block=3)
        self.assertEqual(r.shape, (100, 13))
        self.assertTrue(np.isin(r, hist).all())

    def test_contributions_without_returns(self):
        w = mc.simulate(np.zeros((3, 12)), 1000, 50)
        np.testing.assert_allclose(w[:, -1], 1000 + 12 * 50)

    def test_cvar_is_worse_than_var(self):
        x = np.random.default_rng(3).normal(size=10_000)
        var, cvar = mc.var_cvar(x, 0.05)
        self.assertLess(cvar, var)
        self.assertAlmostEqual(var, -1.645, delta=0.05)

    def test_max_drawdown(self):
        w = np.array([[100, 120, 60, 90, 130]])
        self.assertAlmostEqual(mc.max_drawdown(w)[0], -0.5)

    def test_data_loads(self):
        r = mc.load_returns()
        self.assertGreater(len(r), 400)
        self.assertTrue(np.isfinite(r).all())


if __name__ == "__main__":
    unittest.main()
