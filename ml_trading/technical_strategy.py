"""Cost-aware EUR/USD M1 trend-pullback scalping strategy.

The strategy uses only completed-candle features. It looks for a pullback
inside an established EMA trend, followed by a rejection candle and momentum
confirmation. A trade is rejected when its current spread is too large
relative to the available ATR: for a short-horizon strategy this is a risk and
execution-quality condition, not a device to create or suppress trades.
"""

import pandas as pd

from ml_trading.technical_indicators import TechnicalIndicators


class TrendFollowingStrategy:
    """EUR/USD M1 scalping strategy: trend, pullback, confirmation and cost.

    Parameters are intentionally few and expressed in ATR where possible so
    that variants can be validated on a separate period without calibrating
    dozens of thresholds to a single month of M1 data.
    """

    NAME = "Cost-aware EMA pullback scalping"
    POINT = 0.00001

    def __init__(
        self,
        min_score=7,
        min_atr_ratio=0.00005,
        max_atr_ratio=0.00150,
        min_atr_pips=0.7,
        max_spread_to_atr=0.45,
        min_ema_separation_atr=0.10,
        min_volume_ratio=0.70,
        pullback_tolerance_atr=0.35,
        min_body_ratio=0.35,
        min_close_position=0.60,
        max_rsi_buy=70.0,
        min_rsi_sell=30.0,
        require_hist_acceleration=False,
    ):
        """Configure the signal filters.

        ``max_spread_to_atr`` is based on the spread known when the signal
        candle closes. It prevents a one-minute trade whose expected move is
        largely consumed by execution cost. It is deliberately independent
        of the number of trades observed in any backtest.
        """
        self.min_score = int(min_score)
        self.min_atr_ratio = float(min_atr_ratio)
        self.max_atr_ratio = float(max_atr_ratio)
        self.min_atr_pips = float(min_atr_pips)
        self.max_spread_to_atr = float(max_spread_to_atr)
        self.min_ema_separation_atr = float(min_ema_separation_atr)
        self.min_volume_ratio = float(min_volume_ratio)
        self.pullback_tolerance_atr = float(pullback_tolerance_atr)
        self.min_body_ratio = float(min_body_ratio)
        self.min_close_position = float(min_close_position)
        self.max_rsi_buy = float(max_rsi_buy)
        self.min_rsi_sell = float(min_rsi_sell)
        self.require_hist_acceleration = bool(require_hist_acceleration)

    def parameters(self):
        """Return the explicit signal settings for reports and validation."""
        return {
            "name": self.NAME,
            "min_score": self.min_score,
            "min_atr_ratio": self.min_atr_ratio,
            "max_atr_ratio": self.max_atr_ratio,
            "min_atr_pips": self.min_atr_pips,
            "max_spread_to_atr": self.max_spread_to_atr,
            "min_ema_separation_atr": self.min_ema_separation_atr,
            "min_volume_ratio": self.min_volume_ratio,
            "pullback_tolerance_atr": self.pullback_tolerance_atr,
            "min_body_ratio": self.min_body_ratio,
            "min_close_position": self.min_close_position,
            "max_rsi_buy": self.max_rsi_buy,
            "min_rsi_sell": self.min_rsi_sell,
            "require_hist_acceleration": self.require_hist_acceleration,
        }

    @staticmethod
    def _empty_signal():
        return {"signal": 0, "score": 0, "details": {}, "sl": None, "tp": None}

    @staticmethod
    def _is_finite(*values):
        return all(pd.notna(value) for value in values)

    def generate_signal(self, row):
        """Generate a signal without using information after ``row``."""
        signal_data = self._empty_signal()
        required = [
            "close", "open", "high", "low", "spread", "ema_9", "ema_21",
            "ema_50", "rsi_14", "macd", "macd_signal", "macd_hist",
            "atr_14", "session", "volume_ratio",
        ]
        if not all(key in row.index for key in required):
            return signal_data

        values = [row[key] for key in required if key != "session"]
        if not self._is_finite(*values):
            return signal_data

        details = {}
        if int(row["session"]) not in (1, 2):
            details["session"] = "Outside liquid London/New York sessions"
            signal_data["details"] = details
            return signal_data
        details["session"] = "London" if int(row["session"]) == 1 else "New York"

        close = float(row["close"])
        open_price = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        spread = max(float(row["spread"]), 0.0) * self.POINT
        ema_9 = float(row["ema_9"])
        ema_21 = float(row["ema_21"])
        ema_50 = float(row["ema_50"])
        atr = float(row["atr_14"])
        rsi = float(row["rsi_14"])
        macd = float(row["macd"])
        macd_signal = float(row["macd_signal"])
        macd_hist = float(row["macd_hist"])
        previous_hist = row.get("_previous_macd_hist")
        volume_ratio = float(row["volume_ratio"])

        if close <= 0 or atr <= 0:
            return signal_data

        atr_ratio = atr / close
        min_atr = max(close * self.min_atr_ratio, self.min_atr_pips * 0.0001)
        if atr < min_atr:
            details["volatility"] = "ATR too low for a one-minute setup"
            signal_data["details"] = details
            return signal_data
        if atr_ratio > self.max_atr_ratio:
            details["volatility"] = "ATR unusually high; news-risk filter"
            signal_data["details"] = details
            return signal_data
        details["volatility"] = f"ATR exploitable: {atr:.5f}"

        spread_to_atr = spread / atr
        if spread_to_atr > self.max_spread_to_atr:
            details["cost"] = (
                f"Spread/ATR too high ({spread_to_atr:.2f} > {self.max_spread_to_atr:.2f})"
            )
            signal_data["details"] = details
            return signal_data
        details["cost"] = f"Spread/ATR acceptable: {spread_to_atr:.2f}"

        if volume_ratio < self.min_volume_ratio:
            details["volume"] = f"Relative volume too low: {volume_ratio:.2f}"
            signal_data["details"] = details
            return signal_data
        details["volume"] = f"Relative volume validated: {volume_ratio:.2f}"

        bullish_trend = ema_9 > ema_21 > ema_50 and close > ema_50
        bearish_trend = ema_9 < ema_21 < ema_50 and close < ema_50
        if not (bullish_trend or bearish_trend):
            details["trend"] = "EMA alignment and price location insufficient"
            signal_data["details"] = details
            return signal_data

        direction = "up" if bullish_trend else "down"
        ema_separation = abs(ema_9 - ema_21) / atr
        if ema_separation < self.min_ema_separation_atr:
            details["trend"] = "EMAs too close; no usable directional impulse"
            signal_data["details"] = details
            return signal_data
        details["trend"] = (
            f"{'Haussiere' if direction == 'up' else 'Baissiere'}: EMA 9 "
            f"{'>' if direction == 'up' else '<'} EMA 21 "
            f"{'>' if direction == 'up' else '<'} EMA 50"
        )

        if direction == "up":
            touched_pullback = low <= ema_9 and low >= ema_21 - atr * self.pullback_tolerance_atr
            rejected_pullback = close > ema_9 and close > open_price
        else:
            touched_pullback = high >= ema_9 and high <= ema_21 + atr * self.pullback_tolerance_atr
            rejected_pullback = close < ema_9 and close < open_price
        if not (touched_pullback and rejected_pullback):
            details["pullback"] = "No clean rejected pullback in the EMA 9/21 zone"
            signal_data["details"] = details
            return signal_data
        details["pullback"] = "Retour EMA 9/21 followed by a directional rejection"

        if not TechnicalIndicators.candle_confirmation(
            open_price,
            close,
            high,
            low,
            direction=direction,
            min_body_ratio=self.min_body_ratio,
            min_close_position=self.min_close_position,
        ):
            details["candle"] = "Rejection candle is not directional enough"
            signal_data["details"] = details
            return signal_data
        details["candle"] = f"{'Bullish' if direction == 'up' else 'Bearish'} rejection candle confirmed"

        rsi_confirms = (
            50.0 <= rsi <= self.max_rsi_buy
            if direction == "up"
            else self.min_rsi_sell <= rsi <= 50.0
        )
        details["rsi"] = (
            f"RSI = {rsi:.1f}, coherent momentum"
            if rsi_confirms
            else f"RSI = {rsi:.1f}, outside confirmation range"
        )

        hist_improves = pd.isna(previous_hist) or (
            macd_hist >= float(previous_hist)
            if direction == "up"
            else macd_hist <= float(previous_hist)
        )
        macd_confirms = (
            macd_hist > 0 and macd > macd_signal
            if direction == "up"
            else macd_hist < 0 and macd < macd_signal
        )
        if self.require_hist_acceleration:
            macd_confirms = macd_confirms and hist_improves
        details["macd"] = (
            "MACD in trade direction" + (" and accelerating" if self.require_hist_acceleration else "")
            if macd_confirms
            else "MACD does not confirm the direction"
        )

        # Trend, pullback, usable volatility/cost and rejection candle make
        # six points. A valid signal must also have RSI or MACD confirmation;
        # lowering min_score never turns a bare pullback into an entry.
        score = 6 + int(rsi_confirms) + int(macd_confirms)
        signal_data["score"] = score
        signal_data["details"] = details
        if score >= self.min_score and (rsi_confirms or macd_confirms):
            signal_data["signal"] = 1 if direction == "up" else 2
            signal_data["sl"] = close - atr if direction == "up" else close + atr
            signal_data["tp"] = close + 2 * atr if direction == "up" else close - 2 * atr
        return signal_data

    def evaluate_batch(self, df):
        """Add signals without recalculating indicators or looking ahead."""
        result = df.copy()
        if result.empty:
            result["signal"] = pd.Series(dtype="int64")
            result["score"] = pd.Series(dtype="int64")
            result["signal_details"] = pd.Series(dtype="object")
            return result

        result["_previous_macd_hist"] = result["macd_hist"].shift(1)
        signals, scores, details_list = [], [], []
        for _, row in result.iterrows():
            signal_result = self.generate_signal(row)
            signals.append(signal_result["signal"])
            scores.append(signal_result["score"])
            details_list.append(signal_result["details"])

        result["signal"] = signals
        result["score"] = scores
        result["signal_details"] = details_list
        return result.drop(columns=["_previous_macd_hist"])


class _CostAwareScalpingStrategy:
    """Shared, causal execution-quality filters for M1 strategies.

    The checks are deliberately based only on the completed signal candle.
    A strategy may decide *why* to enter differently, but none of them may
    trade an M1 move whose observed spread consumes an excessive share of its
    current volatility.
    """

    POINT = 0.00001

    def __init__(
        self,
        min_atr_pips=1.1,
        max_spread_to_atr=0.65,
        min_volume_ratio=0.70,
        session_start_hour=15,
        session_end_hour=20,
        min_body_ratio=0.40,
        min_close_position=0.65,
    ):
        self.min_atr_pips = float(min_atr_pips)
        self.max_spread_to_atr = float(max_spread_to_atr)
        self.min_volume_ratio = float(min_volume_ratio)
        self.session_start_hour = int(session_start_hour)
        self.session_end_hour = int(session_end_hour)
        self.min_body_ratio = float(min_body_ratio)
        self.min_close_position = float(min_close_position)

    def _common_parameters(self):
        return {
            "min_atr_pips": self.min_atr_pips,
            "max_spread_to_atr": self.max_spread_to_atr,
            "min_volume_ratio": self.min_volume_ratio,
            "session_start_hour": self.session_start_hour,
            "session_end_hour": self.session_end_hour,
            "min_body_ratio": self.min_body_ratio,
            "min_close_position": self.min_close_position,
        }

    @staticmethod
    def _empty_batch(df):
        result = df.copy()
        result["signal"] = 0
        result["score"] = 0
        result["signal_details"] = [{} for _ in range(len(result))]
        return result

    def _common_masks(self, result):
        required = [
            "open", "high", "low", "close", "spread", "atr_14",
            "volume_ratio", "hour",
        ]
        if not all(column in result.columns for column in required):
            return None

        numeric = result[required].apply(pd.to_numeric, errors="coerce")
        close = numeric["close"]
        atr = numeric["atr_14"]
        candle_range = numeric["high"] - numeric["low"]
        body = (close - numeric["open"]).abs()
        spread_to_atr = (numeric["spread"].clip(lower=0) * self.POINT) / atr.replace(0, pd.NA)
        finite = numeric.notna().all(axis=1) & (close > 0) & (atr > 0) & (candle_range > 0)
        liquid_hours = (
            (numeric["hour"] >= self.session_start_hour)
            & (numeric["hour"] < self.session_end_hour)
        )
        execution_ok = (
            finite
            & liquid_hours
            & (atr >= self.min_atr_pips * 0.0001)
            & (spread_to_atr <= self.max_spread_to_atr)
            & (numeric["volume_ratio"] >= self.min_volume_ratio)
        )
        return {
            "numeric": numeric,
            "close": close,
            "atr": atr,
            "candle_range": candle_range,
            "body": body,
            "body_ratio": body / candle_range,
            "close_position": (close - numeric["low"]) / candle_range,
            "spread_to_atr": spread_to_atr,
            "execution_ok": execution_ok,
        }

    def _details(self, direction, score, row, extra):
        return {
            "setup": extra,
            "direction": direction,
            "session": f"{self.session_start_hour:02d}:00-{self.session_end_hour:02d}:00",
            "cost": f"Spread/ATR {float(row['spread_to_atr']):.2f}",
            "volatility": f"ATR {float(row['atr']):.5f}",
            "volume": f"Relative volume {float(row['volume_ratio']):.2f}",
            "candle": "Directional confirmation",
            "score": int(score),
        }


class SessionBreakoutScalpingStrategy(_CostAwareScalpingStrategy):
    """Continuation scalp after a causal short-term range breakout.

    EUR/USD M1 breakouts are only considered during a declared liquid session
    and in the direction of the short EMA structure.  The rolling high/low is
    shifted by one candle, so the current candle never defines its own level.
    """

    NAME = "Cost-aware session range-breakout scalping"

    def __init__(
        self,
        breakout_lookback=12,
        breakout_buffer_atr=0.05,
        min_score=7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.breakout_lookback = int(breakout_lookback)
        self.breakout_buffer_atr = float(breakout_buffer_atr)
        self.min_score = int(min_score)

    def parameters(self):
        return {
            "name": self.NAME,
            "breakout_lookback": self.breakout_lookback,
            "breakout_buffer_atr": self.breakout_buffer_atr,
            "min_score": self.min_score,
            **self._common_parameters(),
        }

    def evaluate_batch(self, df):
        result = self._empty_batch(df)
        masks = self._common_masks(result)
        required = {"ema_9", "ema_21", "ema_50", "return_5m", "macd_hist"}
        if masks is None or not required.issubset(result.columns):
            return result

        numeric = masks["numeric"]
        for column in required:
            numeric[column] = pd.to_numeric(result[column], errors="coerce")
        prior_high = result["high"].rolling(
            self.breakout_lookback, min_periods=self.breakout_lookback
        ).max().shift(1)
        prior_low = result["low"].rolling(
            self.breakout_lookback, min_periods=self.breakout_lookback
        ).min().shift(1)
        close = masks["close"]
        atr = masks["atr"]
        trend_up = (
            (numeric["ema_9"] > numeric["ema_21"])
            & (numeric["ema_21"] > numeric["ema_50"])
            & (close > numeric["ema_50"])
        )
        trend_down = (
            (numeric["ema_9"] < numeric["ema_21"])
            & (numeric["ema_21"] < numeric["ema_50"])
            & (close < numeric["ema_50"])
        )
        candle_up = (
            (close > numeric["open"])
            & (masks["body_ratio"] >= self.min_body_ratio)
            & (masks["close_position"] >= self.min_close_position)
        )
        candle_down = (
            (close < numeric["open"])
            & (masks["body_ratio"] >= self.min_body_ratio)
            & (masks["close_position"] <= 1.0 - self.min_close_position)
        )
        momentum_up = (numeric["return_5m"] > 0) & (numeric["macd_hist"] > 0)
        momentum_down = (numeric["return_5m"] < 0) & (numeric["macd_hist"] < 0)
        buy = (
            masks["execution_ok"]
            & trend_up
            & candle_up
            & momentum_up
            & (close > prior_high + self.breakout_buffer_atr * atr)
        )
        sell = (
            masks["execution_ok"]
            & trend_down
            & candle_down
            & momentum_down
            & (close < prior_low - self.breakout_buffer_atr * atr)
        )
        score = 6 + (momentum_up | momentum_down).astype(int) + (
            masks["body_ratio"] >= self.min_body_ratio + 0.10
        ).astype(int)
        buy &= score >= self.min_score
        sell &= score >= self.min_score
        result.loc[buy, "signal"] = 1
        result.loc[sell, "signal"] = 2
        result.loc[buy | sell, "score"] = score[buy | sell]
        for index in result.index[buy | sell]:
            direction = "BUY" if result.at[index, "signal"] == 1 else "SELL"
            result.at[index, "signal_details"] = self._details(
                direction,
                result.at[index, "score"],
                {
                    "spread_to_atr": masks["spread_to_atr"].at[index],
                    "atr": atr.at[index],
                    "volume_ratio": numeric["volume_ratio"].at[index],
                },
                f"{self.breakout_lookback}-minute range breakout aligned with EMA momentum",
            )
        return result


class RsiMeanReversionScalpingStrategy(_CostAwareScalpingStrategy):
    """Short-horizon reversion scalp in a non-impulsive EMA regime.

    It fades an exhaustion move only after a rejection candle, not merely
    because RSI is extreme.  A maximum EMA separation avoids fading a strong
    directional impulse where mean reversion is structurally less reliable.
    """

    NAME = "Cost-aware RSI mean-reversion scalping"

    def __init__(
        self,
        oversold_rsi=35.0,
        overbought_rsi=65.0,
        distance_from_ema21_atr=0.70,
        max_ema_spread_atr=0.90,
        min_wick_to_body=0.35,
        min_score=7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.oversold_rsi = float(oversold_rsi)
        self.overbought_rsi = float(overbought_rsi)
        self.distance_from_ema21_atr = float(distance_from_ema21_atr)
        self.max_ema_spread_atr = float(max_ema_spread_atr)
        self.min_wick_to_body = float(min_wick_to_body)
        self.min_score = int(min_score)

    def parameters(self):
        return {
            "name": self.NAME,
            "oversold_rsi": self.oversold_rsi,
            "overbought_rsi": self.overbought_rsi,
            "distance_from_ema21_atr": self.distance_from_ema21_atr,
            "max_ema_spread_atr": self.max_ema_spread_atr,
            "min_wick_to_body": self.min_wick_to_body,
            "min_score": self.min_score,
            **self._common_parameters(),
        }

    def evaluate_batch(self, df):
        result = self._empty_batch(df)
        masks = self._common_masks(result)
        required = {"ema_9", "ema_21", "ema_50", "rsi_14", "return_5m"}
        if masks is None or not required.issubset(result.columns):
            return result

        numeric = masks["numeric"]
        for column in required:
            numeric[column] = pd.to_numeric(result[column], errors="coerce")
        close = masks["close"]
        atr = masks["atr"]
        body = masks["body"].replace(0, pd.NA)
        lower_wick = (numeric[["open", "close"]].min(axis=1) - numeric["low"]).clip(lower=0)
        upper_wick = (numeric["high"] - numeric[["open", "close"]].max(axis=1)).clip(lower=0)
        ranging = (numeric["ema_9"] - numeric["ema_50"]).abs() <= self.max_ema_spread_atr * atr
        buy_rejection = (
            (close > numeric["open"])
            & (masks["close_position"] >= self.min_close_position)
            & (lower_wick / body >= self.min_wick_to_body)
        )
        sell_rejection = (
            (close < numeric["open"])
            & (masks["close_position"] <= 1.0 - self.min_close_position)
            & (upper_wick / body >= self.min_wick_to_body)
        )
        buy = (
            masks["execution_ok"]
            & ranging
            & (numeric["rsi_14"] <= self.oversold_rsi)
            & (numeric["return_5m"] < 0)
            & (close <= numeric["ema_21"] - self.distance_from_ema21_atr * atr)
            & buy_rejection
        )
        sell = (
            masks["execution_ok"]
            & ranging
            & (numeric["rsi_14"] >= self.overbought_rsi)
            & (numeric["return_5m"] > 0)
            & (close >= numeric["ema_21"] + self.distance_from_ema21_atr * atr)
            & sell_rejection
        )
        score = 6 + (ranging.astype(int)) + (
            ((lower_wick / body >= self.min_wick_to_body + 0.25)
             | (upper_wick / body >= self.min_wick_to_body + 0.25)).astype(int)
        )
        buy &= score >= self.min_score
        sell &= score >= self.min_score
        result.loc[buy, "signal"] = 1
        result.loc[sell, "signal"] = 2
        result.loc[buy | sell, "score"] = score[buy | sell]
        for index in result.index[buy | sell]:
            direction = "BUY" if result.at[index, "signal"] == 1 else "SELL"
            result.at[index, "signal_details"] = self._details(
                direction,
                result.at[index, "score"],
                {
                    "spread_to_atr": masks["spread_to_atr"].at[index],
                    "atr": atr.at[index],
                    "volume_ratio": numeric["volume_ratio"].at[index],
                },
                "RSI exhaustion, EMA-21 extension and rejection candle in a non-impulsive regime",
            )
        return result


STRATEGY_TYPES = {
    "ema_pullback": TrendFollowingStrategy,
    "session_breakout": SessionBreakoutScalpingStrategy,
    "rsi_mean_reversion": RsiMeanReversionScalpingStrategy,
}


def build_strategy(strategy_type, parameters=None):
    """Build a declared scalping strategy without dynamic code or tuning."""
    try:
        strategy_class = STRATEGY_TYPES[strategy_type]
    except KeyError as error:
        available = ", ".join(sorted(STRATEGY_TYPES))
        raise ValueError(f"Unknown strategy type '{strategy_type}'. Available: {available}") from error
    return strategy_class(**(parameters or {}))
