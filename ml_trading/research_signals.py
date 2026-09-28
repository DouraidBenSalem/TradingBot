"""Frozen, causal signal definitions for the isolated historical comparison.

This module is deliberately not imported by the live bot.  Its eight candidates
are specified before examining their results; it does not search a parameter
grid or change the deployment guard.  Input timestamps denote UTC M1 opens.
"""

from copy import deepcopy

import numpy as np
import pandas as pd

from ml_trading.technical_strategy import build_strategy


WARMUP_BARS = 300

# Literal snapshot of the current DEMO signal settings.  Do not load a mutable
# deployment report or environment variable to define a research experiment.
BASELINE_STRATEGY = {
    "min_score": 7,
    "min_atr_ratio": 0.00005,
    "max_atr_ratio": 0.00150,
    "min_atr_pips": 0.9,
    "max_spread_to_atr": 0.35,
    "min_ema_separation_atr": 0.10,
    "min_volume_ratio": 0.70,
    "pullback_tolerance_atr": 0.45,
    "min_body_ratio": 0.35,
    "min_close_position": 0.60,
    "max_rsi_buy": 70.0,
    "min_rsi_sell": 30.0,
    "require_hist_acceleration": True,
}
BASELINE_SIMULATION = {
    "sl_atr_multiplier": 1.4,
    "tp_atr_multiplier": 2.2,
    "max_holding_minutes": 45,
}


def _ema_candidate(name, label, rationale, strategy=None, simulation=None, filters=None):
    return {
        "name": name,
        "label": label,
        "strategy_type": "ema_pullback",
        "strategy": {**BASELINE_STRATEGY, **(strategy or {})},
        "simulation": {**BASELINE_SIMULATION, **(simulation or {})},
        "filters": dict(filters or {}),
        "rationale": rationale,
    }


CANDIDATES = [
    _ema_candidate(
        "ema_baseline", "EMA actuelle (reference)",
        "Regles DEMO figees; indicateurs recalcules causalement, RSI corrige.",
    ),
    _ema_candidate(
        "ema_m5_trend", "EMA + tendance M5 cloturee",
        "Exiger une tendance EMA 9/21/50 M5 alignee sur le signal M1.",
        filters={"m5_trend_agreement": True},
    ),
    _ema_candidate(
        "ema_strict_momentum", "EMA + RSI/MACD stricts",
        "Exiger RSI et MACD ensemble et limiter les entrees RSI deja etendues.",
        strategy={"min_score": 8, "max_rsi_buy": 65.0, "min_rsi_sell": 35.0},
    ),
    _ema_candidate(
        "ema_overlap_session", "EMA session 13-17 UTC",
        "Limiter les entrees a une fenetre horaire fixe, sans objectif quotidien.",
        filters={"start_hour_utc": 13, "end_hour_utc": 17},
    ),
    _ema_candidate(
        "ema_cost_volatility", "EMA filtre cout/volatilite",
        "Reserver les entrees a un spread/ATR plus faible et une volatilite moderee.",
        strategy={"max_spread_to_atr": 0.30, "min_atr_pips": 1.1,
                  "max_atr_ratio": 0.00030},
    ),
    _ema_candidate(
        "ema_wider_exits", "EMA SL 1.6 / TP 2.4 ATR",
        "Variante voisine de distances SL/TP, a lot et limites de risque constants.",
        simulation={"sl_atr_multiplier": 1.6, "tp_atr_multiplier": 2.4},
    ),
    {
        "name": "session_breakout",
        "label": "Breakout session",
        "strategy_type": "session_breakout",
        "strategy": {
            "breakout_lookback": 12, "breakout_buffer_atr": 0.05,
            "min_score": 7, "min_atr_pips": 1.15,
            "max_spread_to_atr": 0.65, "min_volume_ratio": 0.70,
            "session_start_hour": 15, "session_end_hour": 20,
            "min_body_ratio": 0.40, "min_close_position": 0.65,
        },
        "simulation": {**BASELINE_SIMULATION, "max_holding_minutes": 20},
        "filters": {},
        "rationale": "Comparer une famille de continuation par cassure du range precedent.",
    },
    {
        "name": "rsi_mean_reversion",
        "label": "RSI retour a la moyenne",
        "strategy_type": "rsi_mean_reversion",
        "strategy": {
            "oversold_rsi": 35.0, "overbought_rsi": 65.0,
            "distance_from_ema21_atr": 0.70, "max_ema_spread_atr": 0.90,
            "min_wick_to_body": 0.35, "min_score": 7,
            "min_atr_pips": 1.1, "max_spread_to_atr": 0.65,
            "min_volume_ratio": 0.65,
            "session_start_hour": 15, "session_end_hour": 20,
            "min_body_ratio": 0.30, "min_close_position": 0.60,
        },
        "simulation": {**BASELINE_SIMULATION, "max_holding_minutes": 20},
        "filters": {},
        "rationale": "Comparer une famille de rejet RSI en regime EMA non impulsif.",
    },
]


def _rolling_rsi(close, period=14):
    """Preserve the existing rolling-SMA RSI convention, with correct edges.

    Fourteen actual price changes are required.  No losses means RSI 100;
    no gains and no losses means 50.  Warmup remains NaN, never backfilled.
    """
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    result = 100.0 - 100.0 / (1.0 + gain / loss.replace(0, np.nan))
    result = result.mask((loss == 0) & (gain > 0), 100.0)
    return result.mask((loss == 0) & (gain == 0), 50.0)


def _add_completed_m5_trend(frame):
    """Attach only full five-minute candles known at each M1 close.

    Missing M1 constituents invalidate that M5 candle.  The last available
    complete candle is carried backward-asof; no unfinished aggregate can
    leak into earlier signals.  ``m5_available_at`` records its close time.
    """
    grouped = frame.assign(_m5_open=frame["time"].dt.floor("5min")).groupby(
        "_m5_open", sort=True
    )
    m5 = grouped.agg(
        close=("close", "last"),
        observations=("time", "size"),
        first_time=("time", "min"),
        last_time=("time", "max"),
    )
    complete = (
        (m5["observations"] == 5)
        & (m5["first_time"] == m5.index)
        & (m5["last_time"] == m5.index + pd.Timedelta(minutes=4))
    )
    m5 = m5.loc[complete].copy()
    for period in (9, 21, 50):
        m5[f"ema_{period}"] = m5["close"].ewm(
            span=period, adjust=False, min_periods=period
        ).mean()
    up = (
        (m5["ema_9"] > m5["ema_21"])
        & (m5["ema_21"] > m5["ema_50"])
        & (m5["close"] > m5["ema_50"])
    )
    down = (
        (m5["ema_9"] < m5["ema_21"])
        & (m5["ema_21"] < m5["ema_50"])
        & (m5["close"] < m5["ema_50"])
    )
    m5["m5_trend"] = np.select([up, down], [1, -1], default=0).astype(np.int8)
    m5["m5_available_at"] = m5.index + pd.Timedelta(minutes=5)
    # A signal from time=10:04 can see the full 10:00-10:05 M5 candle,
    # because both close at 10:05.  Its entry occurs subsequently.
    attached = pd.merge_asof(
        pd.DataFrame({"_signal_close": frame["time"] + pd.Timedelta(minutes=1)}),
        m5[["m5_available_at", "m5_trend"]].reset_index(drop=True),
        left_on="_signal_close", right_on="m5_available_at", direction="backward",
        allow_exact_matches=True,
    )
    frame["m5_trend"] = attached["m5_trend"].fillna(0).to_numpy(dtype=np.int8)
    frame["m5_available_at"] = attached["m5_available_at"].array
    return frame


def prepare_features(raw):
    """Recompute causal research features from validated chronological OHLC.

    No rows are sorted, dropped, filled, or repaired.  Invalid market data
    must be rejected by the runner; malformed chronology is rejected here
    as well.  Naive timestamps are interpreted as UTC, matching MT5 rates.
    """
    required = {"time", "open", "high", "low", "close", "tick_volume", "spread"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing research input columns: {sorted(missing)}")
    frame = raw.copy(deep=True)
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
    if (frame["time"].isna().any() or frame["time"].duplicated().any()
            or not frame["time"].is_monotonic_increasing):
        raise ValueError("Research input must have unique increasing non-null timestamps")
    if not frame["time"].equals(frame["time"].dt.floor("min")):
        raise ValueError("Research input timestamps must be aligned to M1 opens")
    for column in required - {"time"}:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    for period in (9, 21, 50):
        frame[f"ema_{period}"] = frame["close"].ewm(span=period, adjust=False).mean()
    frame["rsi_14"] = _rolling_rsi(frame["close"])
    frame["macd"] = (
        frame["close"].ewm(span=12, adjust=False).mean()
        - frame["close"].ewm(span=26, adjust=False).mean()
    )
    frame["macd_signal"] = frame["macd"].ewm(span=9, adjust=False).mean()
    frame["macd_hist"] = frame["macd"] - frame["macd_signal"]
    prior_close = frame["close"].shift(1)
    true_range = pd.concat([
        frame["high"] - frame["low"],
        (frame["high"] - prior_close).abs(),
        (frame["low"] - prior_close).abs(),
    ], axis=1).max(axis=1)
    frame["atr_14"] = true_range.rolling(14, min_periods=14).mean()
    frame["volume_ratio"] = frame["tick_volume"] / frame["tick_volume"].rolling(
        20, min_periods=20
    ).mean().replace(0, np.nan)
    frame["return_5m"] = frame["close"].pct_change(5, fill_method=None)
    frame["hour"] = frame["time"].dt.hour
    frame["session"] = np.select(
        [frame["hour"].between(13, 21), frame["hour"].between(8, 12)],
        [2, 1], default=0,
    ).astype(np.int8)
    return _add_completed_m5_trend(frame)


def build_signals(frame, candidate):
    """Return immutable-input 0/1/2 signals (flat/BUY/SELL), after warmup."""
    strategy = build_strategy(candidate["strategy_type"], deepcopy(candidate["strategy"]))
    signals = strategy.evaluate_batch(frame)["signal"].to_numpy(dtype=np.int8, copy=True)
    filters = candidate.get("filters", {})
    if filters.get("m5_trend_agreement"):
        direction = np.select([signals == 1, signals == 2], [1, -1], default=0)
        signals[frame["m5_trend"].to_numpy() != direction] = 0
    if "start_hour_utc" in filters:
        hours = frame["time"].dt.hour.to_numpy()
        allowed = (hours >= filters["start_hour_utc"]) & (hours < filters["end_hour_utc"])
        signals[~allowed] = 0
    signals[:WARMUP_BARS] = 0
    return signals
