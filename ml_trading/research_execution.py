"""Isolated EUR/USD M1 research execution, with no database or broker actions.

Input OHLC are bid prices; ``spread`` is in 0.00001 points. Candle spread is
an execution proxy, not tick bid/ask reconstruction. Signals use completed
candles and may execute only at the next contiguous minute's open. Missing
minutes and UTC day boundaries flatten positions at the last available close;
this is an explicit research boundary convention, not a live gap detector.

The equity drawdown is a conservative OHLC bound, not a tick-exact statistic:
favourable and adverse intrabar paths are unknown. It assumes the favourable
extreme can precede the adverse extreme, bounds the latter at an executable SL,
and includes the full adverse candle excursion for a TP candle. Liquidation
equity includes round-trip commission and adverse exit slippage. No leverage,
margin, stop-out, financing or variable intrabar spread model is available.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


POINT = 0.00001
CONTRACT_SIZE = 100000.0
MINUTE_NS = 60_000_000_000


def simulate(
    df,
    signals,
    *,
    sl_atr_multiplier=1.4,
    tp_atr_multiplier=2.2,
    max_holding_minutes=45,
    max_spread_to_atr=0.35,
    commission_per_lot_side=7.0,
    slippage_points=1.0,
    spread_multiplier=1.0,
    fixed_lot=0.07,
    initial_balance=100.0,
    daily_loss_limit=10.0,
    max_trades_per_day=3,
):
    """Return unrounded metrics, trades and daily results for fixed signals.

    Signals: 0=flat, 1=BUY, 2=SELL. ATR and SL/TP distances are frozen on the
    signal candle. Limits gate new entries using realized daily net P&L; they
    do not guarantee that an open trade cannot overshoot a daily loss limit.
    Positive realized daily net P&L stops new entries for that UTC day. There
    is one position at a time and no minimum daily trading requirement.
    """
    positive = {
        "sl_atr_multiplier": sl_atr_multiplier,
        "tp_atr_multiplier": tp_atr_multiplier,
        "fixed_lot": fixed_lot,
        "initial_balance": initial_balance,
        "daily_loss_limit": daily_loss_limit,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and strictly positive")
    for name, value in {
        "commission_per_lot_side": commission_per_lot_side,
        "slippage_points": slippage_points,
        "spread_multiplier": spread_multiplier,
        "max_spread_to_atr": max_spread_to_atr,
    }.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if max_trades_per_day < 1 or int(max_trades_per_day) != max_trades_per_day:
        raise ValueError("max_trades_per_day must be a positive integer")
    if max_holding_minutes is not None and (
        not math.isfinite(max_holding_minutes)
        or max_holding_minutes < 1
        or int(max_holding_minutes) != max_holding_minutes
    ):
        raise ValueError("max_holding_minutes must be a positive integer or None")

    required = {"time", "open", "high", "low", "close", "spread", "atr_14"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    n = len(df)
    signal_values = np.asarray(signals)
    if signal_values.shape != (n,) or not np.isin(signal_values, [0, 1, 2]).all():
        raise ValueError("signals must be a one-dimensional 0/1/2 array aligned to df")
    times = pd.DatetimeIndex(pd.to_datetime(df["time"], utc=True))
    if times.hasnans or not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError("time must be strictly increasing, unique and nonmissing")
    # Explicit unit avoids pandas DatetimeIndex storage-resolution differences.
    time_ns = times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    if n and np.any(time_ns % MINUTE_NS):
        raise ValueError("time must identify whole-minute M1 candle opens")
    day_values = times.to_numpy(dtype="datetime64[D]")
    days = np.datetime_as_string(day_values, unit="D")
    o, h, l, c, spread, atr = (
        df[key].to_numpy(dtype=float) for key in
        ("open", "high", "low", "close", "spread", "atr_14")
    )
    if n and (
        not np.isfinite(np.column_stack((o, h, l, c, spread))).all()
        or np.any(spread < 0)
        or np.any(l <= 0)
        or np.any(h < np.maximum(o, c))
        or np.any(l > np.minimum(o, c))
    ):
        raise ValueError("OHLC/spread must be finite, coherent, positive prices and nonnegative spread")
    spreads = spread * POINT * spread_multiplier
    slip = slippage_points * POINT
    units = fixed_lot * CONTRACT_SIZE
    commission = 2 * fixed_lot * commission_per_lot_side

    balance = float(initial_balance)
    realized_peak = balance
    equity_peak_bound = balance
    realized_dd = 0.0
    equity_dd_bound = 0.0
    trades = []
    daily_results = {}
    skipped = {"noncontiguous_entry": 0, "date_boundary_entry": 0,
               "invalid_atr": 0, "entry_spread": 0, "daily_trade_limit": 0,
               "positive_daily_pnl": 0, "daily_loss_limit": 0}
    market_days = 0
    observed_market_days = 0
    if n:
        market_days = int(np.busday_count(day_values[0], day_values[-1] + np.timedelta64(1, "D")))
        observed_market_days = int(np.is_busday(np.unique(day_values)).sum())
        for day in pd.bdate_range(times[0].normalize(), times[-1].normalize()):
            daily_results[day.strftime("%Y-%m-%d")] = {"trades": 0, "pnl": 0.0}
    cursor = 0
    while cursor < n - 1 and balance > 0:
        direction_signal = int(signal_values[cursor])
        if not direction_signal:
            cursor += 1
            continue
        entry_index = cursor + 1
        if time_ns[entry_index] - time_ns[cursor] != MINUTE_NS:
            skipped["noncontiguous_entry"] += 1
            cursor += 1
            continue
        if days[entry_index] != days[cursor]:
            skipped["date_boundary_entry"] += 1
            cursor += 1
            continue
        signal_atr = float(atr[cursor])
        if not math.isfinite(signal_atr) or signal_atr <= 0:
            skipped["invalid_atr"] += 1
            cursor += 1
            continue
        if spreads[entry_index] / signal_atr > max_spread_to_atr:
            skipped["entry_spread"] += 1
            cursor += 1
            continue
        day_key = str(days[entry_index])
        daily = daily_results.setdefault(day_key, {"trades": 0, "pnl": 0.0})
        if daily["trades"] >= max_trades_per_day:
            skipped["daily_trade_limit"] += 1
            cursor += 1
            continue
        if daily["pnl"] > 0:
            skipped["positive_daily_pnl"] += 1
            cursor += 1
            continue
        if daily["pnl"] <= -daily_loss_limit:
            skipped["daily_loss_limit"] += 1
            cursor += 1
            continue

        is_buy = direction_signal == 1
        direction = 1.0 if is_buy else -1.0
        entry = float(o[entry_index] + spreads[entry_index] + slip if is_buy
                      else o[entry_index] - slip)
        stop = entry - direction * sl_atr_multiplier * signal_atr
        target = entry + direction * tp_atr_multiplier * signal_atr
        risk_cash = sl_atr_multiplier * signal_atr * units + slip * units + commission
        pre_balance = balance
        mae = 0.0
        mfe = 0.0
        ambiguous_bars = 0
        j = entry_index
        while True:
            # BUY exits on bid; SELL exits on ask. SL/TP already denote these
            # executable quote prices: spread MUST NOT be charged again.
            quote_spread = 0.0 if is_buy else float(spreads[j])
            exit_open = float(o[j] + quote_spread)
            exit_high = float(h[j] + quote_spread)
            exit_low = float(l[j] + quote_spread)
            exit_close = float(c[j] + quote_spread)
            stop_at_open = exit_open <= stop if is_buy else exit_open >= stop
            target_at_open = exit_open >= target if is_buy else exit_open <= target
            hit_stop = exit_low <= stop if is_buy else exit_high >= stop
            hit_target = exit_high >= target if is_buy else exit_low <= target
            reason = None
            exit_quote = None
            exact_close = False
            if stop_at_open:
                exit_quote, reason = exit_open, "stop_gap"
            elif target_at_open:
                exit_quote, reason = target, "take_profit_gap"
            elif hit_stop:
                exit_quote, reason = stop, "stop_loss"
                ambiguous_bars += int(hit_target)
            elif hit_target:
                exit_quote, reason = target, "take_profit"
            elif max_holding_minutes is not None and (
                time_ns[j] + MINUTE_NS - time_ns[entry_index]
                >= max_holding_minutes * MINUTE_NS
            ):
                exit_quote, reason, exact_close = exit_close, "time_limit", True
            elif j == n - 1:
                exit_quote, reason, exact_close = exit_close, "end_of_data", True
            elif time_ns[j + 1] - time_ns[j] != MINUTE_NS:
                exit_quote, reason, exact_close = exit_close, "data_gap", True
            elif days[j + 1] != days[j]:
                exit_quote, reason, exact_close = exit_close, "day_end", True

            if stop_at_open or target_at_open:
                adverse_quote = favourable_quote = exit_quote
            elif is_buy:
                adverse_quote = max(exit_low, stop)
                favourable_quote = min(exit_high, target)
            else:
                adverse_quote = min(exit_high, stop)
                favourable_quote = max(exit_low, target)
            adverse_pnl = (direction * (adverse_quote - entry) - slip) * units - commission
            favourable_pnl = (direction * (favourable_quote - entry) - slip) * units - commission
            mae = max(mae, -adverse_pnl)
            mfe = max(mfe, favourable_pnl)
            equity_peak_bound = max(equity_peak_bound, pre_balance + favourable_pnl)
            equity_dd_bound = max(
                equity_dd_bound,
                (equity_peak_bound - (pre_balance + adverse_pnl)) / equity_peak_bound,
            )
            if reason is not None:
                break
            j += 1

        exit_price = float(exit_quote - direction * slip)
        gross_pnl = float(direction * (exit_price - entry) * units)
        net_pnl = gross_pnl - commission
        balance += net_pnl
        daily["trades"] += 1
        daily["pnl"] += net_pnl
        realized_peak = max(realized_peak, balance)
        realized_dd = max(realized_dd, (realized_peak - balance) / realized_peak)
        # A final realized observation also covers unusual quote discontinuity.
        equity_peak_bound = max(equity_peak_bound, balance)
        equity_dd_bound = max(equity_dd_bound, (equity_peak_bound - balance) / equity_peak_bound)
        exit_timestamp = times[j] + (pd.Timedelta(minutes=1) if exact_close else pd.Timedelta(0))
        trades.append({
            "signal_time": times[cursor].isoformat(),
            "entry_time": times[entry_index].isoformat(),
            "exit_time": exit_timestamp.isoformat(),
            "exit_bar_time": times[j].isoformat(),
            "exit_time_is_bar_proxy": not exact_close,
            "signal_index": int(cursor), "entry_index": int(entry_index), "exit_index": int(j),
            "side": "BUY" if is_buy else "SELL", "lot": float(fixed_lot),
            "entry_price": entry, "exit_price": exit_price,
            "stop_loss": float(stop), "take_profit": float(target),
            "signal_atr": signal_atr,
            "entry_spread_points": float(spread[entry_index] * spread_multiplier),
            "exit_spread_points": float(spread[j] * spread_multiplier),
            "gross_pnl_after_spread_slippage": gross_pnl,
            "commission": float(commission), "net_pnl": float(net_pnl),
            "balance_before": float(pre_balance), "balance_after": float(balance),
            "reason": reason, "bars_held": int(j - entry_index + 1),
            "holding_minutes_upper_bound": int((time_ns[j] + MINUTE_NS - time_ns[entry_index]) // MINUTE_NS),
            "ambiguous_stop_target_bars": ambiguous_bars,
            "planned_stop_loss_cash_including_costs": float(risk_cash),
            "planned_risk_pct_of_balance": float(risk_cash / pre_balance * 100),
            "max_adverse_liquidation_pnl_bound": float(mae),
            "max_favourable_liquidation_pnl_bound": float(mfe),
        })
        # The exit candle's newly closed signal may lead to a fresh next-open
        # entry, subject to daily limits; no overlapping positions are possible.
        cursor = j

    pnls = np.array([trade["net_pnl"] for trade in trades], dtype=float)
    wins = int((pnls > 0).sum())
    losses = int((pnls < 0).sum())
    gross_profit = float(pnls[pnls > 0].sum())
    gross_loss = float(-pnls[pnls < 0].sum())
    total = len(trades)
    metrics = {
        "initial_balance": float(initial_balance), "final_balance": float(balance),
        "net_profit": float(pnls.sum()), "net_pnl": float(pnls.sum()),
        "total_trades": total, "trades_count": total,
        "wins": wins, "losses": losses, "breakeven": total - wins - losses,
        "win_rate": wins / total * 100 if total else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "expectancy": float(pnls.mean()) if total else 0.0,
        "gross_profit": gross_profit, "gross_loss": gross_loss,
        "max_drawdown_pct": float(realized_dd * 100),
        "max_equity_drawdown_bound_pct": float(equity_dd_bound * 100),
        "market_days": market_days, "observed_market_days": observed_market_days,
        "trades_per_market_day": total / market_days if market_days else 0.0,
        "active_trading_days": sum(day["trades"] > 0 for day in daily_results.values()),
        "total_commission": float(total * commission),
        "total_slippage_cost": float(total * 2 * slip * units),
        "gross_pnl_after_spread_slippage": float(pnls.sum() + total * commission),
        "buy_net_pnl": float(sum(t["net_pnl"] for t in trades if t["side"] == "BUY")),
        "sell_net_pnl": float(sum(t["net_pnl"] for t in trades if t["side"] == "SELL")),
        "max_planned_risk_pct": max((t["planned_risk_pct_of_balance"] for t in trades), default=0.0),
        "average_planned_risk_pct": float(np.mean([t["planned_risk_pct_of_balance"] for t in trades])) if total else 0.0,
        "ambiguous_stop_target_bars": sum(t["ambiguous_stop_target_bars"] for t in trades),
        "gap_liquidations": sum(t["reason"] == "data_gap" for t in trades),
        "daily_loss_overshoot_days": sum(day["pnl"] < -daily_loss_limit for day in daily_results.values()),
        "stopped_on_nonpositive_balance": balance <= 0,
        "margin_model_available": False,
        "entry_rejections": skipped,
    }
    return {"metrics": metrics, "trades": trades, "daily_results": daily_results}
