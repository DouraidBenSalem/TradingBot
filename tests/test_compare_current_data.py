import copy
import unittest

import pandas as pd

from ml_trading.compare_current_data import matched_cost_sensitivity, period_failures, selection_key, split_masks


class ProtocolTests(unittest.TestCase):
    def test_same_trade_sensitivity_cannot_improve_profit(self):
        trades = [dict(side=side, lot=0.07, entry_spread_points=8,
                       exit_spread_points=12, net_pnl=1.0) for side in ("BUY", "SELL")]
        sensitivity = matched_cost_sensitivity(trades)
        self.assertEqual(sensitivity["trade_count"], 2)
        self.assertAlmostEqual(sensitivity["extra_cost"], 0.28 + 0.35)
        self.assertAlmostEqual(sensitivity["net_profit"], 2 - 0.28 - 0.35)

    def test_recent_tail_cannot_change_preselection_ranking(self):
        row = {"selection_failures": [], "periods": {
            "validation": {"metrics": {"net_profit": 3.0, "profit_factor": 1.3,
                "expectancy": 0.2, "max_equity_drawdown_bound_pct": 4.0, "wins": 10}},
            "diagnostic_tail": {"metrics": {"net_profit": -1000.0}},
            "full": {"metrics": {"net_profit": -1000.0}}}}
        changed = copy.deepcopy(row)
        changed["periods"]["diagnostic_tail"]["metrics"]["net_profit"] = 1000.0
        changed["periods"]["full"]["metrics"]["net_profit"] = 1000.0
        self.assertEqual(selection_key(row), selection_key(changed))

    def test_frequency_and_winrate_are_not_acceptance_quotas(self):
        metrics = dict(total_trades=15, net_profit=5.0, profit_factor=1.2,
                       expectancy=0.3, max_equity_drawdown_bound_pct=5.0,
                       trades_per_market_day=0.3, win_rate=40.0)
        self.assertEqual(period_failures(metrics, 10), [])
        metrics["profit_factor"] = 1.10
        self.assertTrue(any("PF" in reason for reason in period_failures(metrics, 10)))

    def test_dates_are_disjoint_and_ordered(self):
        frame = pd.DataFrame({"time": pd.bdate_range("2026-08-03", periods=20)})
        masks, boundaries = split_masks(frame)
        self.assertEqual(sum(masks["train"]), 10)
        self.assertEqual(sum(masks["validation"]), 5)
        self.assertEqual(sum(masks["diagnostic_tail"]), 5)
        self.assertFalse((masks["train"] & masks["validation"]).any())
        self.assertFalse((masks["validation"] & masks["diagnostic_tail"]).any())
        self.assertLess(boundaries["validation_start"], boundaries["diagnostic_tail_start"])


if __name__ == "__main__":
    unittest.main()
