"""Causality and fixed-experiment checks; no broker/database connection."""

from copy import deepcopy
import unittest

import numpy as np
import pandas as pd

from ml_trading.research_signals import (
    BASELINE_SIMULATION, CANDIDATES, build_signals, prepare_features,
)


def candles(count=460, closes=None, start="2026-01-05 13:00:00"):
    if closes is None:
        rng = np.random.default_rng(29)
        closes = 1.10 + np.cumsum(rng.normal(0.000002, 0.00007, count))
    closes = np.asarray(closes, dtype=float)
    opens = np.r_[closes[0] - 0.00002, closes[:-1]]
    return pd.DataFrame({
        "time": pd.date_range(start, periods=len(closes), freq="min", tz="UTC"),
        "open": opens,
        "high": np.maximum(opens, closes) + 0.000025,
        "low": np.minimum(opens, closes) - 0.000025,
        "close": closes,
        "tick_volume": np.full(len(closes), 100),
        "spread": np.full(len(closes), 2),
    })


class ResearchSignalsTests(unittest.TestCase):
    def test_rsi_edges_and_warmup(self):
        for name, values, expected in (
            ("rising", 1.10 + np.arange(50) * 0.00001, 100.0),
            ("falling", 1.10 - np.arange(50) * 0.00001, 0.0),
            ("flat", np.full(50, 1.10), 50.0),
        ):
            with self.subTest(name=name):
                rsi = prepare_features(candles(closes=values))["rsi_14"]
                self.assertTrue(rsi.iloc[:14].isna().all())
                np.testing.assert_array_equal(rsi.iloc[14:], expected)

    def test_features_and_all_signals_are_prefix_causal(self):
        raw = candles()
        full = prepare_features(raw)
        prefix = prepare_features(raw.iloc[:377])
        pd.testing.assert_frame_equal(full.iloc[:377], prefix)
        for candidate in CANDIDATES:
            with self.subTest(candidate=candidate["name"]):
                np.testing.assert_array_equal(
                    build_signals(full, candidate)[:377], build_signals(prefix, candidate)
                )

    def test_m5_only_visible_at_its_close(self):
        raw = candles(260, closes=1.10 + np.arange(260) * 0.00001)
        features = prepare_features(raw)
        self.assertTrue(features["m5_available_at"].iloc[:4].isna().all())
        self.assertEqual(features["m5_available_at"].iloc[4], raw["time"].iloc[5])
        self.assertEqual(features["m5_available_at"].iloc[8], raw["time"].iloc[5])
        self.assertEqual(features["m5_available_at"].iloc[9], raw["time"].iloc[10])
        self.assertEqual(features["m5_trend"].iloc[248], 0)
        self.assertEqual(features["m5_trend"].iloc[249], 1)
        known = features["m5_available_at"].notna()
        self.assertTrue((features.loc[known, "m5_available_at"] <= (
            features.loc[known, "time"] + pd.Timedelta(minutes=1)
        )).all())

    def test_missing_minute_excludes_incomplete_m5(self):
        raw = candles(15).drop(index=7)
        frame = prepare_features(raw)
        at_end_incomplete = frame.loc[frame["time"] == raw["time"].iloc[8]].iloc[0]
        self.assertEqual(at_end_incomplete["time"].minute, 9)
        self.assertEqual(at_end_incomplete["m5_available_at"].minute, 5)
        self.assertEqual(frame.iloc[-1]["m5_available_at"].minute, 15)

    def test_inputs_unchanged_and_filters_abstain_without_forcing(self):
        frame = prepare_features(candles(305))
        # Force a known, valid BUY setup, independently of random fixture prices.
        for key, value in {
            "close": 1.1012, "open": 1.10105, "high": 1.10122, "low": 1.10099,
            "ema_9": 1.101, "ema_21": 1.10095, "ema_50": 1.1007,
            "atr_14": 0.0002, "rsi_14": 60, "macd": 0.00005,
            "macd_signal": 0.00004, "macd_hist": 0.00001,
            "volume_ratio": 1, "spread": 1, "session": 2, "m5_trend": 1,
        }.items():
            frame[key] = value
        frame.loc[301, "m5_trend"] = -1
        before = frame.copy(deep=True)
        candidate_before = deepcopy(CANDIDATES)
        baseline = build_signals(frame, CANDIDATES[0])
        filtered = build_signals(frame, CANDIDATES[1])
        self.assertEqual(baseline.dtype, np.dtype("int8"))
        self.assertFalse(baseline[:300].any())
        np.testing.assert_array_equal(baseline[300:], 1)
        self.assertEqual(filtered[301], 0)
        self.assertEqual(filtered[300], 1)
        pd.testing.assert_frame_equal(before, frame)
        self.assertEqual(candidate_before, CANDIDATES)

    def test_frozen_budget_and_literal_baseline(self):
        self.assertEqual(len(CANDIDATES), 8)
        self.assertEqual(len({item["name"] for item in CANDIDATES}), 8)
        self.assertEqual(BASELINE_SIMULATION, {
            "sl_atr_multiplier": 1.4, "tp_atr_multiplier": 2.2,
            "max_holding_minutes": 45,
        })
        expected = {
            "min_score": 7, "min_atr_pips": 0.9, "max_spread_to_atr": 0.35,
            "min_ema_separation_atr": 0.10, "min_volume_ratio": 0.70,
            "pullback_tolerance_atr": 0.45, "min_body_ratio": 0.35,
            "min_close_position": 0.60, "require_hist_acceleration": True,
        }
        for key, value in expected.items():
            self.assertEqual(CANDIDATES[0]["strategy"][key], value)
        for candidate in CANDIDATES:
            self.assertNotIn("lot", candidate["simulation"])
        self.assertEqual(CANDIDATES[-1]["simulation"]["max_holding_minutes"], 20)

    def test_invalid_chronology_is_rejected_not_sorted(self):
        raw = candles(20)
        with self.assertRaisesRegex(ValueError, "unique increasing"):
            prepare_features(raw.iloc[::-1])
        with self.assertRaisesRegex(ValueError, "unique increasing"):
            prepare_features(pd.concat([raw, raw.iloc[[-1]]]))


if __name__ == "__main__":
    unittest.main()
