"""Backtester pour la stratégie technique pure Trend Following + Pullback.

Les indicateurs techniques ne sont plus recalculés : ils sont chargés
directement depuis la table `public.eurusd_features` via le module
`features_provider`.
"""

import html
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from psycopg2.extras import Json

from Database.postgresql import get_connection, insert_trades_batch
from ml_trading.features_provider import fetch_features_range, validate_features
from ml_trading.technical_strategy import TrendFollowingStrategy


ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / "reports" / "technical_backtest_report.html"
CONTRACT_SIZE = 100000
DEFAULT_INITIAL_BALANCE = float(os.getenv("BACKTEST_INITIAL_BALANCE", "100"))
DEFAULT_COMMISSION_PER_LOT = float(os.getenv("BACKTEST_COMMISSION_PER_LOT", "7"))
DEFAULT_SLIPPAGE_POINTS = float(os.getenv("BACKTEST_SLIPPAGE_POINTS", "1"))
BACKTEST_FIXED_LOT = float(os.getenv("BACKTEST_FIXED_LOT", "0.07"))
BACKTEST_DAILY_LOSS_LIMIT = float(os.getenv("BACKTEST_DAILY_LOSS_LIMIT", "10"))
MAX_TRADES_PER_DAY = 3
STRATEGY_MIN_SCORE = 7
HORIZON_BARS = 30  # Nombre de bougies pour chercher l'exit


def _ensure_tables(connection):
    """Créer les tables de backtest si nécessaire."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS public.technical_backtest_runs (
                id BIGSERIAL PRIMARY KEY,
                run_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                source_start TIMESTAMP,
                source_end TIMESTAMP,
                parameters JSONB NOT NULL,
                metrics JSONB NOT NULL,
                equity_curve JSONB NOT NULL,
                trades JSONB NOT NULL
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS technical_backtest_runs_run_at_idx ON public.technical_backtest_runs (run_at DESC)"
        )
    connection.commit()


def _get_features_bounds(connection):
    with connection.cursor() as cursor:
        cursor.execute("SELECT MIN(time), MAX(time) FROM public.eurusd_features")
        row = cursor.fetchone()
    return (row[0], row[1]) if row else (None, None)


def run(start_time=None, end_time=None):
    """
    Lancer le backtest de la stratégie technique.

    Les indicateurs sont chargés directement depuis `eurusd_features`
    sans recalcul côté application. Le filtrage temporel est strict :
    seules les bougies dont le timestamp est STRICTEMENT COMPRIS dans
    l'intervalle fermé [start_time, end_time] sont utilisées.

    Returns:
        {
            'parameters': {...},
            'metrics': {...},
            'equity_curve': [...],
            'trades': [...]
        }
    """
    connection = get_connection()
    try:
        _ensure_tables(connection)

        if start_time is not None and end_time is not None:
            if end_time <= start_time:
                raise ValueError(
                    f"Période invalide : la fin ({end_time}) doit être strictement postérieure au début ({start_time})."
                )
        min_available, max_available = _get_features_bounds(connection)
        if start_time is not None and max_available is not None and start_time > max_available:
            raise ValueError(
                f"Pas de donnée après {max_available}. Le début demandé ({start_time}) est postérieur."
            )
        if end_time is not None and min_available is not None and end_time < min_available:
            raise ValueError(
                f"Pas de donnée avant {min_available}. La fin demandée ({end_time}) est antérieure."
            )

        df, validation_report = fetch_features_range(
            start_time=start_time,
            end_time=end_time,
            validate=True,
            raise_on_invalid=False,
            freshness_check=False,
        )

        if start_time is not None or end_time is not None:
            if not df.empty and "time" in df.columns:
                mask = pd.Series(True, index=df.index)
                if start_time is not None:
                    mask &= df["time"] >= pd.Timestamp(start_time)
                if end_time is not None:
                    mask &= df["time"] <= pd.Timestamp(end_time)
                df = df.loc[mask].reset_index(drop=True)

        requested_start = str(start_time) if start_time else "-inf"
        requested_end = str(end_time) if end_time else "+inf"

        if df.empty:
            raise ValueError(
                f"Aucune bougie EUR/USD M1 dans l'intervalle demandé [{requested_start} ; {requested_end}]. "
                f"Données disponibles : [{min_available} ; {max_available}]."
            )

        first_df_ts = df["time"].iloc[0]
        last_df_ts = df["time"].iloc[-1]
        filtered_note = None
        if start_time is not None or end_time is not None:
            filtered_note = (
                f"{len(df)} bougies sélectionnées sur l'intervalle [{requested_start} ; {requested_end}] "
                f"(effectif lu : [{first_df_ts} ; {last_df_ts}])"
            )

        strategy = TrendFollowingStrategy(min_score=STRATEGY_MIN_SCORE)

        result = _simulate(
            df,
            strategy,
            initial_balance=DEFAULT_INITIAL_BALANCE,
            commission_per_lot=DEFAULT_COMMISSION_PER_LOT,
            slippage_points=DEFAULT_SLIPPAGE_POINTS,
            fixed_lot=BACKTEST_FIXED_LOT,
            daily_loss_limit=BACKTEST_DAILY_LOSS_LIMIT,
        )
        result["source_start"] = str(first_df_ts)
        result["source_end"] = str(last_df_ts)
        result["requested_start"] = requested_start
        result["requested_end"] = requested_end
        result["filtered_rows"] = len(df)
        result["filter_note"] = filtered_note

        data_analysis = _analyse_backtest_data(df, validation_report)
        result["data_analysis"] = data_analysis

        warehouse_analysis = _get_warehouse_analysis(connection)
        result["warehouse_analysis"] = warehouse_analysis

        parameters = {
            "strategy": strategy.NAME,
            "strategy_parameters": strategy.parameters(),
            "data_source": "eurusd_features (indicateurs pré-calculés)",
            "features_validation": validation_report.to_dict() if validation_report else None,
            "requested_period": {"start": requested_start, "end": requested_end},
            "effective_period": {"start": str(first_df_ts), "end": str(last_df_ts)},
            "filtered_rows": len(df),
            "filter_note": filtered_note,
            "initial_balance": DEFAULT_INITIAL_BALANCE,
            "fixed_lot": BACKTEST_FIXED_LOT,
            "commission_per_lot": DEFAULT_COMMISSION_PER_LOT,
            "slippage_points": DEFAULT_SLIPPAGE_POINTS,
            "max_trades_per_day": MAX_TRADES_PER_DAY,
            "stop_on_positive_daily_net": True,
            "third_trade_recovery_without_risk_increase": True,
            "daily_loss_limit": BACKTEST_DAILY_LOSS_LIMIT,
            "sl_atr_multiplier": 1.0,
            "tp_atr_multiplier": 2.0,
            "horizon_bars": HORIZON_BARS,
            "execution_model": "bid/ask observed spread + adverse slippage on both sides",
            "data_analysis": data_analysis,
            "warehouse_analysis": warehouse_analysis,
            "total_trades_generated": len(result.get("trades", [])),
        }
        
        backtest_run_id = None
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO public.technical_backtest_runs
                (source_start, source_end, parameters, metrics, equity_curve, trades)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    df.iloc[0]['time'],
                    df.iloc[-1]['time'],
                    Json(parameters),
                    Json(result['metrics']),
                    Json(result['equity_curve']),
                    Json(result['trades'])
                )
            )
            row = cursor.fetchone()
            if row:
                backtest_run_id = int(row[0])
        connection.commit()

        if result.get("trades"):
            # La sequence rend chaque trade d'un run idempotent en base. Elle
            # empeche l'auto-synchronisation de l'UI de dupliquer un backtest
            # deja persiste.
            for sequence, trade in enumerate(result["trades"]):
                trade["backtest_run_id"] = backtest_run_id
                trade["trade_sequence"] = sequence
            inserted_ids = insert_trades_batch(
                result["trades"],
                source="backtest",
                backtest_run_id=backtest_run_id,
                connection=connection,
            )
            for i, trade in enumerate(result["trades"]):
                if i < len(inserted_ids):
                    trade["id"] = inserted_ids[i]
            result["trade_history_ids"] = inserted_ids
            # Le JSON du run est la source du resultat recharge par la page.
            # Il doit donc contenir les IDs PostgreSQL attribues ci-dessus.
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE public.technical_backtest_runs SET trades = %s WHERE id = %s",
                    (Json(result["trades"]), backtest_run_id),
                )
            connection.commit()
        
        # Générer le rapport HTML
        _generate_report(result, parameters)
        
        return result
        
    finally:
        connection.close()


def latest_result():
    """Retourner le dernier backtest technique enregistré."""
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT source_start, source_end, parameters, metrics, equity_curve, trades
                FROM public.technical_backtest_runs
                ORDER BY run_at DESC LIMIT 1
            """)
            row = cursor.fetchone()
        if not row:
            return None
        trades = row[5]
        for trade in trades:
            volume = float(trade.get("lot", trade.get("volume", 0)))
            entry = float(trade.get("entry", 0))
            trade.setdefault("lot", round(volume, 2))
            trade.setdefault("sl_dollar", round(abs(entry - float(trade["sl"])) * volume * CONTRACT_SIZE, 2))
            trade.setdefault("tp_dollar", round(abs(float(trade["tp"]) - entry) * volume * CONTRACT_SIZE, 2))
        daily_results = {}
        for trade in trades:
            day = trade["time"][:10]
            daily = daily_results.setdefault(day, {"trades": 0, "pnl": 0.0})
            daily["trades"] += 1
            daily["pnl"] = round(daily["pnl"] + trade["net_pnl"], 2)
        return {
            "source_start": str(row[0]), "source_end": str(row[1]), "parameters": row[2],
            "metrics": row[3], "equity_curve": row[4], "trades": trades, "daily_results": daily_results
        }
    finally:
        connection.close()


def list_runs(limit=50):
    """Lister les derniers runs de backtest triés par date décroissante."""
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT source_start, source_end, parameters, metrics, equity_curve, trades
                FROM public.technical_backtest_runs
                ORDER BY run_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cursor.fetchall()
        if not rows:
            return []
        results = []
        for row in rows:
            trades = row[5]
            for trade in trades:
                volume = float(trade.get("lot", trade.get("volume", 0)))
                entry = float(trade.get("entry", 0))
                trade.setdefault("lot", round(volume, 2))
                trade.setdefault("sl_dollar", round(abs(entry - float(trade["sl"])) * volume * CONTRACT_SIZE, 2))
                trade.setdefault("tp_dollar", round(abs(float(trade["tp"]) - entry) * volume * CONTRACT_SIZE, 2))
            daily_results = {}
            for trade in trades:
                day = trade["time"][:10]
                daily = daily_results.setdefault(day, {"trades": 0, "pnl": 0.0})
                daily["trades"] += 1
                daily["pnl"] = round(daily["pnl"] + trade["net_pnl"], 2)
            results.append({
                "source_start": str(row[0]),
                "source_end": str(row[1]),
                "parameters": row[2],
                "metrics": row[3],
                "equity_curve": row[4],
                "trades": trades,
                "daily_results": daily_results,
            })
        return results
    finally:
        connection.close()


def run_multiple(periods):
    """
    Lancer le backtest pour plusieurs périodes.

    Args:
        periods: liste de tuples (start_time, end_time) en objets datetime.

    Returns:
        Liste de résultats de backtest.
    """
    results = []
    for start_time, end_time in periods:
        result = run(start_time=start_time, end_time=end_time)
        results.append(result)
    return results


def _simulate(
    df,
    strategy,
    initial_balance,
    commission_per_lot,
    slippage_points,
    fixed_lot,
    daily_loss_limit,
    max_trades_per_day=MAX_TRADES_PER_DAY,
    sl_atr_multiplier=1.0,
    tp_atr_multiplier=2.0,
    horizon_bars=HORIZON_BARS,
):
    """
    Simuler les trades basés sur la stratégie avec détails complets.
    """
    max_trades_per_day = min(3, max(1, int(max_trades_per_day or 3)))
    balance = initial_balance
    equity = []
    trades = []
    daily_count = {}
    daily_pnl = {}
    stopped_days = set()
    next_entry_index = 0
    peak = balance
    point = 0.00001

    df = strategy.evaluate_batch(df)

    all_days_in_range = set()
    market_days_in_range = set()
    if not df.empty:
        start_date = pd.to_datetime(df["time"].iloc[0]).date()
        end_date = pd.to_datetime(df["time"].iloc[-1]).date()
        delta_days = (end_date - start_date).days
        for i in range(delta_days + 1):
            all_days_in_range.add(str(start_date + pd.Timedelta(days=i)))
        market_days_in_range = set(
            pd.to_datetime(df.loc[df["session"].isin([1, 2]), "time"]).dt.date.astype(str)
        )

    for index in range(len(df) - 1):
        if index < next_entry_index:
            continue
        row = df.iloc[index]

        signal = int(row["signal"])
        if signal == 0:
            continue

        day_key = str(pd.to_datetime(row["time"]).date())

        if day_key in stopped_days:
            continue
        if daily_count.get(day_key, 0) >= max_trades_per_day:
            stopped_days.add(day_key)
            continue
        if daily_count.get(day_key, 0) > 0 and daily_pnl.get(day_key, 0.0) > 0:
            stopped_days.add(day_key)
            continue
        if daily_pnl.get(day_key, 0.0) <= -daily_loss_limit:
            stopped_days.add(day_key)
            continue

        entry_row = df.iloc[index + 1]
        # The OHLC series is bid-priced. Buy opens and sell closes occur at
        # ask, while sell opens and buy closes occur at bid. Use the entry and
        # exit candle spreads rather than applying the signal-bar spread to
        # both directions; this makes the simulated execution symmetric.
        entry_spread = max(float(entry_row.get("spread") or 0), 0.0) * point
        slippage = slippage_points * point

        if signal == 1:
            entry_price = float(entry_row["open"]) + entry_spread + slippage
        else:
            entry_price = float(entry_row["open"]) - slippage

        atr = float(row["atr_14"])
        if not np.isfinite(atr) or atr <= 0:
            continue

        if signal == 1:
            sl = entry_price - sl_atr_multiplier * atr
            tp = entry_price + tp_atr_multiplier * atr
        else:
            sl = entry_price + sl_atr_multiplier * atr
            tp = entry_price - tp_atr_multiplier * atr

        volume = fixed_lot
        commission = volume * commission_per_lot * 2
        sl_dollar = abs(entry_price - sl) * volume * CONTRACT_SIZE
        tp_dollar = abs(tp - entry_price) * volume * CONTRACT_SIZE

        entry_time = entry_row["time"]
        exit_price = None
        exit_time = None
        outcome = "TIMEOUT"
        exit_index = None
        exit_reason_code = "timeout"

        future_end = min(index + 1 + horizon_bars, len(df))
        for future_idx in range(index + 1, future_end):
            candle = df.iloc[future_idx]
            exit_time = candle["time"]
            exit_spread = max(float(candle.get("spread") or 0), 0.0) * point

            if signal == 1:
                if candle["low"] <= sl:
                    exit_price = sl - slippage
                    outcome = "LOSS"
                    exit_reason_code = "sl"
                    exit_index = future_idx
                    break
                if candle["high"] >= tp:
                    exit_price = tp - slippage
                    outcome = "WIN"
                    exit_reason_code = "tp"
                    exit_index = future_idx
                    break
            else:
                # A short is bought back at ask. Its stop/target must
                # therefore be checked against bid OHLC plus the observed
                # spread and its exit is priced at ask.
                if candle["high"] + exit_spread >= sl:
                    exit_price = sl + exit_spread + slippage
                    outcome = "LOSS"
                    exit_reason_code = "sl"
                    exit_index = future_idx
                    break
                if candle["low"] + exit_spread <= tp:
                    exit_price = tp + exit_spread + slippage
                    outcome = "WIN"
                    exit_reason_code = "tp"
                    exit_index = future_idx
                    break

        if exit_price is None:
            if future_end > index + 1:
                future_row = df.iloc[future_end - 1]
                timeout_spread = max(float(future_row.get("spread") or 0), 0.0) * point
                exit_price = float(future_row["close"])
                if signal == 1:
                    exit_price -= slippage
                else:
                    exit_price += timeout_spread + slippage
                exit_time = future_row["time"]
                exit_reason_code = "end_of_horizon"
            else:
                continue

        if signal == 1:
            price_pnl = (exit_price - entry_price) * volume * CONTRACT_SIZE
        else:
            price_pnl = (entry_price - exit_price) * volume * CONTRACT_SIZE

        net_pnl = price_pnl - commission
        # A take-profit can be too small to overcome a widened execution
        # spread and commission. The win/loss classification used by metrics
        # must always reflect realised net P&L, not merely the touched level.
        outcome = "WIN" if net_pnl > 0 else "LOSS" if net_pnl < 0 else "BREAKEVEN"
        balance_before = round(balance, 2)
        next_entry_index = (exit_index + 1) if exit_index is not None else future_end
        balance += net_pnl

        daily_count[day_key] = daily_count.get(day_key, 0) + 1
        daily_pnl[day_key] = daily_pnl.get(day_key, 0.0) + net_pnl

        if (
            daily_count[day_key] >= max_trades_per_day
            or daily_pnl[day_key] > 0
            or daily_pnl[day_key] <= -daily_loss_limit
        ):
            stopped_days.add(day_key)

        peak = max(peak, balance)
        drawdown = balance - peak

        entry_dt = pd.to_datetime(entry_time)
        exit_dt = pd.to_datetime(exit_time)
        duration_seconds = int(max(0, (exit_dt - entry_dt).total_seconds()))
        duration_minutes = round(duration_seconds / 60.0, 1)

        signal_details = row.get("signal_details") or {}
        score_breakdown = _build_score_breakdown(signal, signal_details, atr, row)

        exit_reason_text = _build_exit_reason(
            exit_reason_code, outcome, atr, sl_dollar, tp_dollar, duration_minutes
        )

        trade_details = {
            "time": str(entry_time),
            "entry_time": str(entry_time),
            "exit_time": str(exit_time),
            "entry_date": str(entry_dt.date()),
            "entry_hour": entry_dt.strftime("%H:%M:%S"),
            "exit_hour": exit_dt.strftime("%H:%M:%S"),
            "duration_seconds": duration_seconds,
            "duration_minutes": duration_minutes,
            "duration_text": _format_duration(duration_seconds),
            "side": "BUY" if signal == 1 else "SELL",
            "entry": round(entry_price, 5),
            "exit": round(exit_price, 5),
            "volume": volume,
            "ema_9": round(float(row["ema_9"]), 5),
            "ema_21": round(float(row["ema_21"]), 5),
            "ema_50": round(float(row["ema_50"]), 5),
            "rsi": round(float(row["rsi_14"]), 2),
            "macd": round(float(row["macd"]), 5),
            "macd_signal": round(float(row["macd_signal"]), 5),
            "macd_hist": round(float(row["macd_hist"]), 5),
            "atr": round(atr, 5),
            "volume_ratio": round(float(row["volume_ratio"]), 3),
            "session": int(row["session"]),
            "score": int(row["score"]),
            "score_breakdown": score_breakdown,
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "lot": round(volume, 2),
            "sl_dollar": round(sl_dollar, 2),
            "tp_dollar": round(tp_dollar, 2),
            "price_pnl": round(price_pnl, 2),
            "commission": round(commission, 2),
            "slippage": round(slippage * volume * CONTRACT_SIZE, 2),
            "net_pnl": round(net_pnl, 2),
            "balance_before": balance_before,
            "balance": round(balance, 2),
            "outcome": outcome,
            "exit_reason_code": exit_reason_code,
            "exit_reason": exit_reason_text,
            "signal_details": signal_details,
            "why_trade": _build_why_trade_text(signal, signal_details, score_breakdown, row, atr),
        }

        trades.append(trade_details)

        equity.append(
            {
                "time": str(exit_time),
                "balance": round(balance, 2),
                "drawdown": round(drawdown, 2),
            }
        )

    if trades:
        wins = [t for t in trades if t["net_pnl"] > 0]
        losses = [t for t in trades if t["net_pnl"] < 0]
        buy_trades = [t for t in trades if t["side"] == "BUY"]
        sell_trades = [t for t in trades if t["side"] == "SELL"]

        total_profit = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
        total_loss = abs(sum(t["net_pnl"] for t in trades if t["net_pnl"] < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0

        win_rate = len(wins) / len(trades) if trades else 0
        avg_win = sum(t["net_pnl"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["net_pnl"] for t in losses) / len(losses) if losses else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

        max_drawdown = min([0] + [e["drawdown"] for e in equity])

        durations = [t["duration_minutes"] for t in trades]
        avg_duration_minutes = round(sum(durations) / len(durations), 1) if durations else 0

        best_trade = max(trades, key=lambda t: t["net_pnl"]) if trades else None
        worst_trade = min(trades, key=lambda t: t["net_pnl"]) if trades else None

        days_traded_set = set(t["time"][:10] for t in trades)
        days_traded = len(days_traded_set)
        days_without_trade = len(all_days_in_range - days_traded_set)

        metrics = {
            "initial_balance": initial_balance,
            "final_balance": round(balance, 2),
            "net_profit": round(balance - initial_balance, 2),
            "return_percent": round((balance - initial_balance) / initial_balance * 100, 2),
            "total_trades": len(trades),
            "buy_trades": len(buy_trades),
            "sell_trades": len(sell_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(win_rate * 100, 2),
            "profit_factor": round(profit_factor, 2),
            "expectancy": round(expectancy, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "max_drawdown": round(max_drawdown, 2),
            "max_drawdown_percent": round(max_drawdown / initial_balance * 100, 2),
            "total_profit": round(total_profit, 2),
            "total_loss": round(total_loss, 2),
            "avg_trade_duration_minutes": avg_duration_minutes,
            "avg_trade_duration_text": _format_duration(int(avg_duration_minutes * 60)),
            "best_trade": {
                "time": best_trade["time"],
                "side": best_trade["side"],
                "net_pnl": best_trade["net_pnl"],
            }
            if best_trade
            else None,
            "worst_trade": {
                "time": worst_trade["time"],
                "side": worst_trade["side"],
                "net_pnl": worst_trade["net_pnl"],
            }
            if worst_trade
            else None,
            "days_traded": days_traded,
            "days_without_trade": days_without_trade,
            "total_days_in_range": len(all_days_in_range),
            "market_days_in_range": len(market_days_in_range),
            "trades_per_market_day": round(len(trades) / max(len(market_days_in_range), 1), 2),
        }
    else:
        metrics = {
            "initial_balance": initial_balance,
            "final_balance": round(balance, 2),
            "net_profit": 0,
            "return_percent": 0,
            "total_trades": 0,
            "buy_trades": 0,
            "sell_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "expectancy": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "max_drawdown": 0,
            "max_drawdown_percent": 0,
            "total_profit": 0,
            "total_loss": 0,
            "avg_trade_duration_minutes": 0,
            "avg_trade_duration_text": "0 s",
            "best_trade": None,
            "worst_trade": None,
            "days_traded": 0,
            "days_without_trade": len(all_days_in_range),
            "total_days_in_range": len(all_days_in_range),
            "market_days_in_range": len(market_days_in_range),
            "trades_per_market_day": 0,
        }

    daily_results = {}
    for trade in trades:
        day = trade["time"][:10]
        daily = daily_results.setdefault(day, {"trades": 0, "pnl": 0.0})
        daily["trades"] += 1
        daily["pnl"] = round(daily["pnl"] + trade["net_pnl"], 2)

    return {
        "parameters": {
            "strategy": getattr(strategy, "NAME", "Trend Following + Pullback"),
            "strategy_parameters": strategy.parameters() if hasattr(strategy, "parameters") else {},
            "initial_balance": initial_balance,
            "fixed_lot": fixed_lot,
            "commission_per_lot": commission_per_lot,
            "slippage_points": slippage_points,
            "max_trades_per_day": max_trades_per_day,
            "stop_on_positive_daily_net": True,
            "third_trade_recovery_without_risk_increase": True,
            "daily_loss_limit": daily_loss_limit,
            "sl_atr_multiplier": sl_atr_multiplier,
            "tp_atr_multiplier": tp_atr_multiplier,
            "horizon_bars": horizon_bars,
        },
        "metrics": metrics,
        "equity_curve": equity,
        "trades": trades,
        "daily_results": daily_results,
    }


def _format_duration(seconds):
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins} min {secs} s" if secs else f"{mins} min"
    hours = seconds // 3600
    mins = (seconds % 3600) // 60
    return f"{hours} h {mins} min"


def _build_score_breakdown(signal, signal_details, atr, row):
    direction = "up" if signal == 1 else "down"
    breakdown = {}
    details = signal_details or {}
    points_trend = 0
    trend = details.get("trend", "")
    trend_lower = trend.lower()
    if "Haussière" in trend or "Baissière" in trend:
        points_trend = 2
    # Les details actuels sont volontairement ASCII pour etre stables quel
    # que soit l'encodage du terminal ou de la base de donnees.
    if "haussiere" in trend_lower or "baissiere" in trend_lower:
        points_trend = 2
    breakdown["trend"] = {"points": points_trend, "max": 2, "text": trend or "Non évalué"}

    points_pullback = 2 if "zone pullback" in (details.get("pullback") or "") else 0
    if "retour ema" in (details.get("pullback") or "").lower():
        points_pullback = 2
    breakdown["pullback"] = {
        "points": points_pullback,
        "max": 2,
        "text": details.get("pullback") or "Non évalué",
    }

    volatility_text = details.get("volatility") or ""
    points_atr = 1 if "suffisante" in volatility_text else 0
    if "exploitable" in volatility_text.lower():
        points_atr = 1
    breakdown["atr"] = {
        "points": points_atr,
        "max": 1,
        "text": volatility_text or "Non évalué",
    }

    rsi_text = details.get("rsi") or ""
    points_rsi = 1 if "confirme" in rsi_text else 0
    if "coherent" in rsi_text.lower():
        points_rsi = 1
    breakdown["rsi"] = {
        "points": points_rsi,
        "max": 1,
        "text": rsi_text or "Non évalué",
    }

    macd_text = details.get("macd") or ""
    points_macd = 1 if "confirme" in macd_text else 0
    if "sens du trade" in macd_text.lower():
        points_macd = 1
    breakdown["macd"] = {
        "points": points_macd,
        "max": 1,
        "text": macd_text or "Non évalué",
    }

    candle_text = details.get("candle") or ""
    points_candle = 1 if "confirmée" in candle_text else 0
    if "confirmee" in candle_text.lower():
        points_candle = 1
    breakdown["candle"] = {
        "points": points_candle,
        "max": 1,
        "text": candle_text or "Non évalué",
    }

    total = sum(v["points"] for v in breakdown.values())
    breakdown["_total"] = {"points": total, "max": 8}
    return breakdown


def _build_why_trade_text(signal, signal_details, score_breakdown, row, atr):
    side = "BUY" if signal == 1 else "SELL"
    entry_dt = pd.to_datetime(row["time"])
    hour_str = entry_dt.strftime("%H:%M")
    lines = [f"{side} à {hour_str}", ""]

    trend = score_breakdown.get("trend", {})
    lines.append("Tendance :")
    lines.append(
        f"EMA 9 {'>' if signal == 1 else '<'} EMA 21 {'>' if signal == 1 else '<'} EMA 50 → +{trend.get('points', 0)}"
    )
    lines.append(f"   ({trend.get('text', '')})")
    lines.append("")

    pullback = score_breakdown.get("pullback", {})
    lines.append("Pullback :")
    lines.append(f"{pullback.get('text', '')} → +{pullback.get('points', 0)}")
    lines.append("")

    rsi = score_breakdown.get("rsi", {})
    lines.append("RSI :")
    rsi_val = round(float(row["rsi_14"]), 1)
    rsi_conf = "confirmation BUY" if signal == 1 else "confirmation SELL"
    lines.append(f"RSI = {rsi_val} → {rsi_conf} → +{rsi.get('points', 0)}")
    lines.append(f"   ({rsi.get('text', '')})")
    lines.append("")

    macd = score_breakdown.get("macd", {})
    lines.append("MACD :")
    macd_direction = "Momentum haussier" if signal == 1 else "Momentum baissier"
    lines.append(f"{macd_direction} → +{macd.get('points', 0)}")
    lines.append(f"   ({macd.get('text', '')})")
    lines.append("")

    atr_section = score_breakdown.get("atr", {})
    lines.append("ATR :")
    lines.append(
        f"{'Volatilité suffisante' if atr_section.get('points') else 'Volatilité faible'} → +{atr_section.get('points', 0)}"
    )
    lines.append(f"   (ATR = {round(atr, 5)})")
    lines.append("")

    candle = score_breakdown.get("candle", {})
    lines.append("Bougie :")
    candle_type = "Bougie de confirmation haussière" if signal == 1 else "Bougie de confirmation baissière"
    lines.append(f"{candle_type} → +{candle.get('points', 0)}")
    lines.append(f"   ({candle.get('text', '')})")
    lines.append("")

    total_info = score_breakdown.get("_total", {"points": 0, "max": 8})
    lines.append(f"Score final : {total_info['points']}/{total_info['max']}")
    lines.append("")
    valid = total_info["points"] >= 5
    lines.append(f"→ Trade {side} {'validé' if valid else 'refusé'}.")

    return "\n".join(lines)


def _build_exit_reason(code, outcome, atr, sl_dollar, tp_dollar, duration_minutes):
    if code == "tp":
        return (
            f"TP atteint (2 × ATR)\n"
            f"Prise de profit : +{tp_dollar:.2f} $\n"
            f"Rapport risque/rendement : 2:1"
        )
    if code == "sl":
        return (
            f"SL atteint (1 × ATR)\n"
            f"Perte : -{sl_dollar:.2f} $\n"
            f"Stop suiveur : non activé (SL fixe 1 × ATR)"
        )
    if code == "end_of_horizon":
        return (
            f"Fin de l'horizon de {HORIZON_BARS} bougies (timeout)\n"
            f"Aucun TP ni SL atteint avant {HORIZON_BARS} barres M1\n"
            f"Sortie au Close de la bougie #{HORIZON_BARS}"
        )
    return (
        f"Sortie anticipée\n"
        f"Durée : {duration_minutes} min\n"
        f"Résultat : {'+' if outcome == 'WIN' else ''}—"
    )


def _analyse_backtest_data(df, validation_report):
    """
    Analyse détaillée des données utilisées par le backtest.

    Retourne :
        - Période (début/fin)
        - Nombre total de bougies M1
        - Première / Dernière bougie
        - Données manquantes, doublons
        - Qualité des données
        - Indicateurs : premier index valide (warmup des EMA)
        - Source des données
        - Nombre de trades générés (sera rempli par l'appelant)
    """
    if df is None or df.empty:
        return {
            "error": "Aucune donnée à analyser",
            "candles_total": 0,
            "first_candle": None,
            "last_candle": None,
            "missing_count": 0,
            "duplicate_count": 0,
            "quality_label": "N/A",
            "quality_score": 0,
            "warmup_candles": 0,
            "source": "eurusd_features",
        }

    total_rows = len(df)
    first_ts = pd.to_datetime(df["time"].iloc[0])
    last_ts = pd.to_datetime(df["time"].iloc[-1])

    first_candle = {
        "time": str(first_ts),
        "open": round(float(df["open"].iloc[0]), 5),
        "high": round(float(df["high"].iloc[0]), 5),
        "low": round(float(df["low"].iloc[0]), 5),
        "close": round(float(df["close"].iloc[0]), 5),
    }
    last_candle = {
        "time": str(last_ts),
        "open": round(float(df["open"].iloc[-1]), 5),
        "high": round(float(df["high"].iloc[-1]), 5),
        "low": round(float(df["low"].iloc[-1]), 5),
        "close": round(float(df["close"].iloc[-1]), 5),
    }

    missing_count = 0
    required_cols = [
        "open", "high", "low", "close",
        "ema_9", "ema_21", "ema_50",
        "rsi_14", "atr_14", "macd", "macd_hist",
    ]
    for col in required_cols:
        if col in df.columns:
            missing_count += int(df[col].isna().sum())

    duplicate_count = 0
    if "time" in df.columns:
        duplicate_count = int(df["time"].duplicated().sum())

    expected_total = 0
    if total_rows > 1:
        total_seconds = (last_ts - first_ts).total_seconds()
        expected_total = int(total_seconds / 60) + 1

    missing_in_range = max(0, expected_total - total_rows)

    warmup_candles = 0
    for col in ("ema_50",):
        if col in df.columns:
            first_valid = df[col].first_valid_index()
            if first_valid is not None and isinstance(first_valid, (int, np.integer)):
                warmup_candles = max(warmup_candles, int(first_valid))

    validation = validation_report.to_dict() if validation_report else None
    null_ratio = (validation.get("null_ratio") if validation else 0) or 0
    range_violations = validation.get("value_range_violations") if validation else {}
    n_violations = len(range_violations or {})

    score = 100
    score -= min(50, null_ratio * 100 * 10)
    score -= min(25, duplicate_count * 2)
    score -= min(25, n_violations * 5)
    score = max(0, int(score))

    if score >= 90:
        quality_label = "Excellente"
    elif score >= 75:
        quality_label = "Bonne"
    elif score >= 50:
        quality_label = "Correcte"
    else:
        quality_label = "À vérifier"

    period_start_dt = first_ts.to_pydatetime()
    period_end_dt = last_ts.to_pydatetime()
    days_delta = (period_end_dt.date() - period_start_dt.date()).days + 1
    hours_covered = (last_ts - first_ts).total_seconds() / 3600.0

    return {
        "period_start": str(period_start_dt),
        "period_end": str(period_end_dt),
        "period_start_date": period_start_dt.strftime("%d/%m/%Y"),
        "period_start_hour": period_start_dt.strftime("%H:%M"),
        "period_end_date": period_end_dt.strftime("%d/%m/%Y"),
        "period_end_hour": period_end_dt.strftime("%H:%M"),
        "candles_total": total_rows,
        "candles_expected_in_range": expected_total,
        "days_in_range": days_delta,
        "hours_covered": round(hours_covered, 1),
        "first_candle": first_candle,
        "last_candle": last_candle,
        "missing_values_total": missing_count,
        "missing_candles_in_range": missing_in_range,
        "duplicate_count": duplicate_count,
        "null_ratio_percent": round(null_ratio * 100, 2),
        "value_range_violations": n_violations,
        "quality_label": quality_label,
        "quality_score": score,
        "warmup_candles": warmup_candles,
        "indicator_candles_used": max(0, total_rows - warmup_candles),
        "source": "public.eurusd_features (PostgreSQL)",
        "validation_report": validation,
    }


def _get_warehouse_analysis(connection):
    """
    Diagnostic complet de la table du Data Warehouse eurusd_features.

    Utilise une connexion déjà ouverte (réutilise celle du run()).
    """
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                COUNT(*),
                MIN(time), MAX(time),
                COUNT(*) - COUNT(DISTINCT time) AS duplicate_time
            FROM public.eurusd_features
            """
        )
        row = cursor.fetchone()
        total_rows, min_time, max_time, duplicate_count = (
            row if row else (0, None, None, 0)
        )

        total_nulls = 0
        ohlc_anomalies = 0
        atr_nulls = 0
        ema_nulls = 0
        rsi_nulls = 0
        macd_nulls = 0
        if total_rows and total_rows > 0:
            cursor.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL
                                      OR ema_9 IS NULL OR ema_21 IS NULL OR ema_50 IS NULL
                                      OR rsi_14 IS NULL OR atr_14 IS NULL
                                      OR macd IS NULL OR macd_hist IS NULL),
                    COUNT(*) FILTER (WHERE high < low OR high < open OR high < close
                                      OR low > open OR low > close),
                    COUNT(*) FILTER (WHERE atr_14 IS NULL OR atr_14 <= 0),
                    COUNT(*) FILTER (WHERE ema_9 IS NULL OR ema_21 IS NULL OR ema_50 IS NULL),
                    COUNT(*) FILTER (WHERE rsi_14 IS NULL OR rsi_14 < 0 OR rsi_14 > 100),
                    COUNT(*) FILTER (WHERE macd IS NULL OR macd_hist IS NULL)
                FROM public.eurusd_features
                """
            )
            r = cursor.fetchone()
            total_nulls = int(r[0] or 0)
            ohlc_anomalies = int(r[1] or 0)
            atr_nulls = int(r[2] or 0)
            ema_nulls = int(r[3] or 0)
            rsi_nulls = int(r[4] or 0)
            macd_nulls = int(r[5] or 0)

        gap_count = 0
        missing_minutes = 0
        if total_rows and total_rows > 1:
            cursor.execute(
                """
                WITH ordered AS (
                    SELECT time, LAG(time) OVER (ORDER BY time) AS previous_time
                    FROM public.eurusd_features
                )
                SELECT COUNT(*),
                       COALESCE(SUM(FLOOR(EXTRACT(EPOCH FROM (time - previous_time)) / 60)::BIGINT - 1), 0)
                FROM ordered
                WHERE previous_time IS NOT NULL
                  AND time > previous_time + (1 * INTERVAL '1 minute')
                  AND EXTRACT(ISODOW FROM previous_time) NOT IN (6, 7)
                """
            )
            r = cursor.fetchone()
            gap_count = int(r[0] or 0)
            missing_minutes = int(r[1] or 0)

    available_days = 0
    m1_candles_available = total_rows
    if min_time and max_time:
        available_days = (max_time.date() - min_time.date()).days + 1

    last_data_available_str = str(max_time) if max_time else None

    total_issues = total_nulls + duplicate_count + ohlc_anomalies + gap_count
    if total_issues == 0:
        wh_status = "Sain"
    elif total_issues <= 5:
        wh_status = "Mineurs"
    elif total_issues <= 50:
        wh_status = "À surveiller"
    else:
        wh_status = "Critique"

    return {
        "table_name": "public.eurusd_features",
        "source_ohlcv_table": "public.eurusd_m1",
        "total_rows": int(total_rows or 0),
        "period_available_start": str(min_time) if min_time else None,
        "period_available_end": str(max_time) if max_time else None,
        "period_start_formatted": (
            min_time.strftime("%d/%m/%Y %H:%M") if min_time else "-"
        ),
        "period_end_formatted": (
            max_time.strftime("%d/%m/%Y %H:%M") if max_time else "-"
        ),
        "m1_candles_available": int(m1_candles_available or 0),
        "available_days": available_days,
        "last_data_available": last_data_available_str,
        "last_data_formatted": (
            max_time.strftime("%d/%m/%Y %H:%M") if max_time else "-"
        ),
        "total_null_values": total_nulls,
        "duplicate_count": int(duplicate_count or 0),
        "ohlc_anomalies": ohlc_anomalies,
        "atr_nulls_or_invalid": atr_nulls,
        "ema_nulls": ema_nulls,
        "rsi_nulls_or_invalid": rsi_nulls,
        "macd_nulls": macd_nulls,
        "gap_count": gap_count,
        "missing_candles_minutes": missing_minutes,
        "total_issues": total_issues,
        "status": wh_status,
    }


def get_warehouse_analysis_snapshot():
    """Point d'entrée public : diagnostic du warehouse avec sa propre connexion."""
    connection = get_connection()
    try:
        return _get_warehouse_analysis(connection)
    finally:
        connection.close()


def get_preset_periods():
    """
    Retourne les 4 backtests prédéfinis :
      1. Dernières 24h
      2. Derniers 7 jours
      3. Derniers 30 jours
      4. Période complète

    Les périodes sont calculées à partir des données réellement disponibles
    dans public.eurusd_features (jamais de dates codées en dur).
    """
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT MIN(time), MAX(time) FROM public.eurusd_features")
            row = cursor.fetchone()
        min_ts, max_ts = row if row else (None, None)
    finally:
        connection.close()

    empty_preset = {
        "start": None, "end": None, "start_date": "", "start_hour": "",
        "end_date": "", "end_hour": "", "label": "", "subtitle": "",
        "objective": "", "disabled": True,
    }

    if not max_ts or not min_ts:
        empty_preset["label"] = "Aucune donnée disponible"
        return [empty_preset.copy() for _ in range(4)]

    def _fmt(dt):
        return (
            dt.strftime("%d/%m/%Y"),
            dt.strftime("%H:%M"),
            dt,
        )

    presets = []

    p1_end = max_ts
    p1_start = max(p1_end - timedelta(hours=24), min_ts)
    sd, sh, _ = _fmt(p1_start)
    ed, eh, _ = _fmt(p1_end)
    presets.append({
        "id": "last24h",
        "label": "Dernières 24 h",
        "subtitle": "Comportement récent du bot",
        "objective": "Observer le comportement récent du bot sur les dernières 24h disponibles.",
        "start": p1_start.isoformat(),
        "end": p1_end.isoformat(),
        "start_date": sd, "start_hour": sh,
        "end_date": ed, "end_hour": eh,
        "days_label": "24 h",
        "disabled": False,
    })

    p2_end = max_ts
    p2_start = max(p2_end - timedelta(days=7), min_ts)
    sd, sh, _ = _fmt(p2_start)
    ed, eh, _ = _fmt(p2_end)
    presets.append({
        "id": "last7d",
        "label": "Derniers 7 jours",
        "subtitle": "Période courte représentative",
        "objective": "Analyser le comportement sur une semaine (sessions Londres + NY répétées).",
        "start": p2_start.isoformat(),
        "end": p2_end.isoformat(),
        "start_date": sd, "start_hour": sh,
        "end_date": ed, "end_hour": eh,
        "days_label": "7 j",
        "disabled": False,
    })

    p3_end = max_ts
    p3_start = max(p3_end - timedelta(days=30), min_ts)
    sd, sh, _ = _fmt(p3_start)
    ed, eh, _ = _fmt(p3_end)
    presets.append({
        "id": "last30d",
        "label": "Derniers 30 jours",
        "subtitle": "Stabilité sur 1 mois",
        "objective": "Évaluer la stabilité et la robustesse de la stratégie sur ~1 mois.",
        "start": p3_start.isoformat(),
        "end": p3_end.isoformat(),
        "start_date": sd, "start_hour": sh,
        "end_date": ed, "end_hour": eh,
        "days_label": "30 j",
        "disabled": False,
    })

    sd, sh, _ = _fmt(min_ts)
    ed, eh, _ = _fmt(max_ts)
    presets.append({
        "id": "full",
        "label": "Période complète",
        "subtitle": "Tout l'historique disponible",
        "objective": "Vision globale : performance brute de la stratégie sur tout l'historique.",
        "start": min_ts.isoformat(),
        "end": max_ts.isoformat(),
        "start_date": sd, "start_hour": sh,
        "end_date": ed, "end_hour": eh,
        "days_label": f"{(max_ts.date() - min_ts.date()).days + 1} j",
        "disabled": False,
    })

    return presets


def _generate_report(result, parameters):
    """Générer un rapport HTML du backtest."""
    trades_html = ""
    for trade in result['trades']:
        outcome_color = "green" if trade['outcome'] == 'WIN' else "red" if trade['outcome'] == 'LOSS' else "orange"
        trades_html += f"""
        <tr>
            <td>{trade['time']}</td>
            <td>{trade['exit_time']}</td>
            <td>{trade['side']}</td>
            <td>{trade['entry']:.5f}</td>
            <td>{trade['ema_9']:.5f}</td>
            <td>{trade['ema_21']:.5f}</td>
            <td>{trade['ema_50']:.5f}</td>
            <td>{trade['rsi']:.1f}</td>
            <td>{trade['macd']:.5f}</td>
            <td>{trade['atr']:.5f}</td>
            <td><strong>{trade['score']}/8</strong></td>
            <td>{trade['sl']:.5f}</td>
            <td>{trade['tp']:.5f}</td>
            <td>{trade['lot']:.2f}</td>
            <td>{trade['sl_dollar']:.2f} $</td>
            <td>{trade['tp_dollar']:.2f} $</td>
            <td>{trade['exit']:.5f}</td>
            <td>{trade['price_pnl']:.2f}</td>
            <td>{trade['commission']:.2f}</td>
            <td>{trade['slippage']:.2f}</td>
            <td style="color: {outcome_color}; font-weight: bold">{trade['net_pnl']:.2f}</td>
            <td>{trade['balance']:.2f}</td>
            <td>{trade['outcome']}</td>
        </tr>
        """
    
    metrics = result['metrics']
    html_content = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>Backtest Stratégie Technique</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            h1 {{ color: #333; }}
            h2 {{ color: #555; margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
            .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
            .metric-box {{ background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .metric-label {{ font-size: 12px; color: #999; text-transform: uppercase; }}
            .metric-value {{ font-size: 20px; font-weight: bold; color: #333; }}
            .positive {{ color: #27ae60; }}
            .negative {{ color: #e74c3c; }}
            table {{ width: 100%; border-collapse: collapse; background: white; margin: 20px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            th {{ background: #34495e; color: white; padding: 12px; text-align: left; }}
            td {{ padding: 10px; border-bottom: 1px solid #ddd; font-size: 12px; }}
            tr:hover {{ background: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>Backtest Stratégie Technique - EUR/USD M1</h1>
        <p><small>Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
        
        <h2>Stratégie</h2>
        <p><strong>Type:</strong> Trend Following + Pullback rejete + Momentum</p>
        <p><strong>Indicateurs:</strong> EMA 9/21/50, RSI 14, MACD, ATR 14</p>
        <p><strong>Score minimum:</strong> 7/8</p>
        
        <h2>Métriques Principales</h2>
        <div class="metrics">
            <div class="metric-box">
                <div class="metric-label">Balance Finale</div>
                <div class="metric-value {('positive' if metrics['net_profit'] >= 0 else 'negative')}">${metrics['final_balance']:.2f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Profit Net</div>
                <div class="metric-value {('positive' if metrics['net_profit'] >= 0 else 'negative')}">${metrics['net_profit']:.2f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Retour %</div>
                <div class="metric-value {('positive' if metrics['return_percent'] >= 0 else 'negative')}">{metrics['return_percent']:.2f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Nombre de Trades</div>
                <div class="metric-value">{metrics['total_trades']}</div>
            </div>
        </div>
        
        <div class="metrics">
            <div class="metric-box">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{metrics['win_rate']:.2f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value {('positive' if metrics['profit_factor'] > 1.2 else 'negative')}">{metrics['profit_factor']:.2f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Expectancy</div>
                <div class="metric-value {('positive' if metrics['expectancy'] > 0 else 'negative')}">${metrics['expectancy']:.2f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">${metrics['max_drawdown']:.2f} ({metrics['max_drawdown_percent']:.2f}%)</div>
            </div>
        </div>
        
        <h2>Détails des Trades</h2>
        <table>
            <thead>
                <tr>
                    <th>Heure Entrée</th>
                    <th>Heure Sortie</th>
                    <th>Type</th>
                    <th>Prix Entrée</th>
                    <th>EMA 9</th>
                    <th>EMA 21</th>
                    <th>EMA 50</th>
                    <th>RSI</th>
                    <th>MACD</th>
                    <th>ATR</th>
                    <th>Score</th>
                    <th>SL (prix)</th>
                    <th>TP (prix)</th>
                    <th>Lot</th>
                    <th>SL ($)</th>
                    <th>TP ($)</th>
                    <th>Prix Sortie</th>
                    <th>P&L Prix</th>
                    <th>Commission</th>
                    <th>Slippage</th>
                    <th>P&L Net</th>
                    <th>Balance</th>
                    <th>Résultat</th>
                </tr>
            </thead>
            <tbody>
                {trades_html}
            </tbody>
        </table>
        
        <h2>Résumé</h2>
        <ul>
            <li>Trades gagnants: {metrics['winning_trades']}</li>
            <li>Trades perdants: {metrics['losing_trades']}</li>
            <li>Profit total: ${metrics['total_profit']:.2f}</li>
            <li>Perte totale: ${metrics['total_loss']:.2f}</li>
            <li>Gain moyen: ${metrics['avg_win']:.2f}</li>
            <li>Perte moyenne: ${metrics['avg_loss']:.2f}</li>
        </ul>
    </body>
    </html>
    """
    
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html_content, encoding="utf-8")
