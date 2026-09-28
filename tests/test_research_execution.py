"""Deterministic fill/accounting tests; no database or broker connection."""

import json
import unittest

import numpy as np
import pandas as pd

from ml_trading.research_execution import simulate


def candles(n=5, start="2026-09-21 10:00", **overrides):
    values = {
        "time": pd.date_range(start, periods=n, freq="min"),
        "open": np.full(n, 1.1), "high": np.full(n, 1.10002),
        "low": np.full(n, 1.09998), "close": np.full(n, 1.1),
        "spread": np.full(n, 2.0), "atr_14": np.full(n, 0.0002),
    }
    values.update(overrides)
    return pd.DataFrame(values)


class ResearchExecutionTests(unittest.TestCase):
    def test_long_short_flat_market_costs_are_symmetric(self):
        for side in (1, 2):
            with self.subTest(side=side):
                result = simulate(candles(), [side, 0, 0, 0, 0], max_holding_minutes=1)
                trade = result["trades"][0]
                # .07*100000*(2 spread points + 2 slippage points) + .98 fees
                self.assertAlmostEqual(trade["net_pnl"], -1.26, places=9)
                self.assertAlmostEqual(trade["commission"], 0.98, places=9)
                self.assertEqual(trade["entry_index"], 1)
                self.assertEqual(trade["exit_index"], 1)
                self.assertEqual(trade["reason"], "time_limit")
                json.dumps(result, allow_nan=False)

    def test_short_stop_target_are_ask_levels_without_second_spread(self):
        # Equal executable SL/TP distances produce equal BUY/SELL P&L.
        for side in (1, 2):
            for stop in (False, True):
                with self.subTest(side=side, stop=stop):
                    frame = candles(n=3)
                    if (side == 1 and not stop) or (side == 2 and stop):
                        frame.loc[1, "high"] = 1.101
                    else:
                        frame.loc[1, "low"] = 1.099
                    result = simulate(frame, [side, 0, 0], sl_atr_multiplier=1,
                                      tp_atr_multiplier=1, max_holding_minutes=1)
                    trade = result["trades"][0]
                    self.assertEqual(trade["reason"], "stop_loss" if stop else "take_profit")
                    # SL/TP frozen relative to filled entry; only exit slippage
                    # remains outside that price distance, then commission.
                    expected = (-0.0002 if stop else 0.0002) * 7000 - 0.07 - 0.98
                    self.assertAlmostEqual(trade["net_pnl"], expected, places=9)

    def test_both_stop_and_target_in_same_candle_stop_wins(self):
        frame = candles(n=3)
        frame.loc[1, ["high", "low"]] = [1.102, 1.098]
        for side in (1, 2):
            result = simulate(frame, [side, 0, 0])
            self.assertEqual(result["trades"][0]["reason"], "stop_loss")
            self.assertEqual(result["metrics"]["ambiguous_stop_target_bars"], 1)
            self.assertLess(result["metrics"]["net_profit"], 0)

    def test_adverse_open_gap_fills_at_worse_open(self):
        frame = candles(n=4)
        frame.loc[2, ["open", "high", "low", "close"]] = [1.098, 1.0981, 1.0979, 1.098]
        result = simulate(frame, [1, 0, 0, 0])
        trade = result["trades"][0]
        self.assertEqual(trade["reason"], "stop_gap")
        self.assertAlmostEqual(trade["exit_price"], 1.098 - 0.00001)
        self.assertLess(trade["exit_price"], trade["stop_loss"])

    def test_missing_minutes_flatten_and_do_not_execute_stale_signal(self):
        frame = candles(n=4)
        frame["time"] = pd.to_datetime(["2026-09-21 10:00", "2026-09-21 10:01",
                                        "2026-09-21 10:10", "2026-09-21 10:11"])
        result = simulate(frame, [1, 1, 0, 0], max_holding_minutes=None)
        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(result["trades"][0]["reason"], "data_gap")
        self.assertEqual(result["trades"][0]["exit_index"], 1)
        self.assertEqual(result["metrics"]["entry_rejections"]["noncontiguous_entry"], 1)

    def test_no_overnight_and_no_cross_midnight_signal_entry(self):
        frame = candles(n=5, start="2026-09-21 23:57")
        result = simulate(frame, [1, 0, 1, 0, 0], max_holding_minutes=None)
        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(result["trades"][0]["reason"], "day_end")
        self.assertEqual(result["trades"][0]["exit_index"], 2)
        self.assertEqual(result["metrics"]["entry_rejections"]["date_boundary_entry"], 1)

    def test_entry_spread_rechecked_on_next_open_with_signal_atr(self):
        frame = candles(n=3)
        frame.loc[1, "spread"] = 20
        result = simulate(frame, [1, 0, 0])
        self.assertEqual(result["metrics"]["total_trades"], 0)
        self.assertEqual(result["metrics"]["entry_rejections"]["entry_spread"], 1)

    def test_daily_loss_and_trade_cap_are_realized_net_constraints(self):
        frame = candles(n=9)
        signals = [1] * 9
        result = simulate(frame, signals, max_holding_minutes=1)
        self.assertEqual(result["metrics"]["total_trades"], 3)
        self.assertAlmostEqual(result["metrics"]["net_profit"], -3.78, places=9)
        result = simulate(frame, signals, max_holding_minutes=1, daily_loss_limit=2)
        self.assertEqual(result["metrics"]["total_trades"], 2)
        self.assertEqual(result["metrics"]["daily_loss_overshoot_days"], 1)

    def test_positive_day_stops_new_entries(self):
        frame = candles(n=5)
        frame.loc[1, "high"] = 1.101
        result = simulate(frame, [1] * 5)
        self.assertGreater(result["metrics"]["net_profit"], 0)
        self.assertEqual(result["metrics"]["total_trades"], 1)
        self.assertGreater(result["metrics"]["entry_rejections"]["positive_daily_pnl"], 0)

    def test_no_forced_trades_and_calendar_denominator_includes_missing_days(self):
        frame = candles(n=2)
        frame["time"] = pd.to_datetime(["2026-09-21 10:00", "2026-09-25 10:00"])
        result = simulate(frame, [0, 0])
        self.assertEqual(result["metrics"]["total_trades"], 0)
        self.assertEqual(result["metrics"]["market_days"], 5)
        self.assertEqual(result["metrics"]["observed_market_days"], 2)
        self.assertEqual(len(result["daily_results"]), 5)
        self.assertIsNone(result["metrics"]["profit_factor"])

    def test_intrabar_equity_drawdown_bound_not_only_realized_balance(self):
        frame = candles(n=3)
        frame.loc[1, ["high", "low"]] = [1.10015, 1.0998]
        result = simulate(frame, [1, 0, 0], max_holding_minutes=1)
        metrics = result["metrics"]
        self.assertGreater(metrics["max_equity_drawdown_bound_pct"], metrics["max_drawdown_pct"])
        self.assertAlmostEqual(metrics["max_drawdown_pct"], 1.26, places=8)
        self.assertFalse(metrics["margin_model_available"])

    def test_stop_on_nonpositive_balance(self):
        frame = candles(n=5)
        frame.loc[2, ["open", "high", "low", "close"]] = [1.08, 1.0801, 1.0799, 1.08]
        result = simulate(frame, [1] * 5)
        self.assertTrue(result["metrics"]["stopped_on_nonpositive_balance"])
        self.assertEqual(len(result["trades"]), 1)

    def test_input_validation_rejects_unsorted_or_unaligned_data(self):
        with self.assertRaises(ValueError):
            simulate(candles(), [0])
        with self.assertRaises(ValueError):
            simulate(candles().iloc[::-1], [0] * 5)


if __name__ == "__main__":
    unittest.main()
