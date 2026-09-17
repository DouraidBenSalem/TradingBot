"""Repeatable, read-only validation for EUR/USD M1 scalping variants.

This module intentionally does not optimise a parameter per observed day or
force an order quota. It compares a small number of economically motivated
variants and requires positive results on both chronological sub-periods
before a configuration can be considered for deployment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml_trading.backtest_technical import (
    BACKTEST_DAILY_LOSS_LIMIT,
    BACKTEST_FIXED_LOT,
    DEFAULT_COMMISSION_PER_LOT,
    DEFAULT_INITIAL_BALANCE,
    DEFAULT_SLIPPAGE_POINTS,
    _simulate,
)
from ml_trading.features_provider import fetch_features_range
from ml_trading.technical_strategy import build_strategy


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_PATH = ROOT / "reports" / "scalping_validation_report.json"
DEFAULT_MARKDOWN_REPORT_PATH = ROOT / "reports" / "scalping_validation_report.md"


# Each candidate represents a recognisable, economically motivated choice.
# This is deliberately a short pre-declared list rather than an optimiser:
# the final holdout period is never used to rank candidates.
CANDIDATES = {
    "ema_pullback_wider_stop": {
        "strategy_type": "ema_pullback",
        "strategy": {
            "min_score": 7,
            "max_spread_to_atr": 0.35,
            "min_atr_pips": 0.9,
            "min_ema_separation_atr": 0.10,
            "min_volume_ratio": 0.70,
            "pullback_tolerance_atr": 0.45,
            "min_body_ratio": 0.35,
            "min_close_position": 0.60,
            "require_hist_acceleration": True,
        },
        "simulation": {"sl_atr_multiplier": 1.4, "tp_atr_multiplier": 2.2, "horizon_bars": 45},
    },
    "ema_pullback_execution_selective": {
        "strategy_type": "ema_pullback",
        "strategy": {
            "min_score": 7,
            "max_spread_to_atr": 0.30,
            "min_atr_pips": 1.0,
            "min_ema_separation_atr": 0.15,
            "min_volume_ratio": 0.80,
            "pullback_tolerance_atr": 0.30,
            "min_body_ratio": 0.45,
            "min_close_position": 0.65,
            "require_hist_acceleration": True,
        },
        "simulation": {"sl_atr_multiplier": 1.0, "tp_atr_multiplier": 2.0, "horizon_bars": 30},
    },
    "session_breakout_balanced": {
        "strategy_type": "session_breakout",
        "strategy": {
            "breakout_lookback": 12,
            "breakout_buffer_atr": 0.05,
            "min_score": 7,
            "min_atr_pips": 1.15,
            "max_spread_to_atr": 0.65,
            "min_volume_ratio": 0.70,
            "session_start_hour": 15,
            "session_end_hour": 20,
            "min_body_ratio": 0.40,
            "min_close_position": 0.65,
        },
        "simulation": {"sl_atr_multiplier": 1.15, "tp_atr_multiplier": 1.60, "horizon_bars": 20},
    },
    "session_breakout_selective": {
        "strategy_type": "session_breakout",
        "strategy": {
            "breakout_lookback": 20,
            "breakout_buffer_atr": 0.12,
            "min_score": 8,
            "min_atr_pips": 1.35,
            "max_spread_to_atr": 0.55,
            "min_volume_ratio": 0.85,
            "session_start_hour": 15,
            "session_end_hour": 19,
            "min_body_ratio": 0.50,
            "min_close_position": 0.72,
        },
        "simulation": {"sl_atr_multiplier": 1.20, "tp_atr_multiplier": 1.80, "horizon_bars": 25},
    },
    "rsi_mean_reversion_balanced": {
        "strategy_type": "rsi_mean_reversion",
        "strategy": {
            "oversold_rsi": 35.0,
            "overbought_rsi": 65.0,
            "distance_from_ema21_atr": 0.70,
            "max_ema_spread_atr": 0.90,
            "min_wick_to_body": 0.35,
            "min_score": 7,
            "min_atr_pips": 1.10,
            "max_spread_to_atr": 0.65,
            "min_volume_ratio": 0.65,
            "min_body_ratio": 0.30,
            "min_close_position": 0.60,
            "session_start_hour": 15,
            "session_end_hour": 20,
        },
        "simulation": {"sl_atr_multiplier": 1.10, "tp_atr_multiplier": 1.35, "horizon_bars": 18},
    },
    "rsi_mean_reversion_selective": {
        "strategy_type": "rsi_mean_reversion",
        "strategy": {
            "oversold_rsi": 30.0,
            "overbought_rsi": 70.0,
            "distance_from_ema21_atr": 0.95,
            "max_ema_spread_atr": 0.65,
            "min_wick_to_body": 0.60,
            "min_score": 8,
            "min_atr_pips": 1.25,
            "max_spread_to_atr": 0.55,
            "min_volume_ratio": 0.80,
            "min_body_ratio": 0.40,
            "min_close_position": 0.65,
            "session_start_hour": 15,
            "session_end_hour": 20,
        },
        "simulation": {"sl_atr_multiplier": 1.20, "tp_atr_multiplier": 1.50, "horizon_bars": 20},
    },
}

MINIMUM_MARKET_DAYS = 30
MINIMUM_FULL_SAMPLE_TRADES = 30
MAX_ACCEPTABLE_DRAWDOWN_PERCENT = 15.0


def _run_frame(
    df: pd.DataFrame,
    strategy_type: str,
    strategy_args: dict,
    simulation_args: dict,
) -> dict:
    return _simulate(
        df,
        build_strategy(strategy_type, strategy_args),
        initial_balance=DEFAULT_INITIAL_BALANCE,
        commission_per_lot=DEFAULT_COMMISSION_PER_LOT,
        slippage_points=DEFAULT_SLIPPAGE_POINTS,
        fixed_lot=BACKTEST_FIXED_LOT,
        daily_loss_limit=BACKTEST_DAILY_LOSS_LIMIT,
        **simulation_args,
    )["metrics"]


def _summary(metrics: dict) -> dict:
    keys = (
        "total_trades", "winning_trades", "losing_trades", "trades_per_market_day", "net_profit", "return_percent",
        "win_rate", "profit_factor", "expectancy", "max_drawdown_percent",
        "avg_win", "avg_loss", "market_days_in_range",
    )
    return {key: metrics[key] for key in keys}


def _period_failures(metrics: dict, min_trades: int) -> list[str]:
    """Return auditable gate failures rather than a bare boolean."""
    failures = []
    checks = (
        (metrics["total_trades"] >= min_trades, f"trades {metrics['total_trades']} < {min_trades}"),
        (metrics["trades_per_market_day"] >= 1.0, f"frequency {metrics['trades_per_market_day']} < 1.0/day"),
        (metrics["net_profit"] > 0, f"net profit {metrics['net_profit']} <= 0"),
        (metrics["profit_factor"] >= 1.10, f"profit factor {metrics['profit_factor']} < 1.10"),
        (metrics["expectancy"] > 0, f"expectancy {metrics['expectancy']} <= 0"),
        (metrics["win_rate"] >= 50.0, f"win rate {metrics['win_rate']}% < 50%"),
        (
            abs(metrics["max_drawdown_percent"]) <= MAX_ACCEPTABLE_DRAWDOWN_PERCENT,
            f"drawdown {metrics['max_drawdown_percent']}% exceeds {MAX_ACCEPTABLE_DRAWDOWN_PERCENT}%",
        ),
    )
    return [message for passed, message in checks if not passed]


def _passes_period(metrics: dict, min_trades: int) -> bool:
    """Acceptance conditions for an independently evaluated period."""
    return not _period_failures(metrics, min_trades)


def _walk_forward_windows(data: pd.DataFrame, dates: list, holdout_date) -> list[dict]:
    """Build expanding, chronological train -> validation folds before holdout.

    The final holdout is excluded at construction time.  This keeps the
    stability check informative without leaking the final evaluation into
    candidate selection.
    """
    pre_holdout_dates = [date for date in dates if date < holdout_date]
    if len(pre_holdout_dates) < 12:
        return []
    validation_size = max(3, len(pre_holdout_dates) // 4)
    starts = (len(pre_holdout_dates) - 2 * validation_size, len(pre_holdout_dates) - validation_size)
    windows = []
    for validation_start_index in starts:
        if validation_start_index < validation_size:
            continue
        validation_end_index = min(validation_start_index + validation_size, len(pre_holdout_dates))
        train_start = pre_holdout_dates[0]
        validation_start = pre_holdout_dates[validation_start_index]
        # The final fold ends at the holdout boundary, never at the end of
        # the complete data set.
        validation_end = (
            pre_holdout_dates[validation_end_index]
            if validation_end_index < len(pre_holdout_dates)
            else holdout_date
        )
        train = data[(data["time"].dt.date >= train_start) & (data["time"].dt.date < validation_start)].reset_index(drop=True)
        validation_mask = (
            (data["time"].dt.date >= validation_start)
            & (data["time"].dt.date < validation_end)
        )
        validation = data[validation_mask].reset_index(drop=True)
        windows.append({
            "train": train,
            "validation": validation,
            "train_start": str(train_start),
            "validation_start": str(validation_start),
            "validation_end": str(validation_end),
        })
    return windows


def _selection_key(result: dict) -> tuple:
    """Rank on train + validation only; never inspect the final holdout."""
    train = result["in_sample"]
    validation = result["selection_validation"]
    return (
        result["selection_passed"],
        result["walk_forward_passed"],
        validation["net_profit"] > 0,
        train["net_profit"] > 0,
        validation["profit_factor"],
        validation["net_profit"],
        validation["win_rate"],
        validation["trades_per_market_day"],
        validation["max_drawdown_percent"],
    )


def validate_candidates(df: pd.DataFrame | None = None) -> dict:
    """Compare declared candidates with a final untouched chronological holdout.

    Candidates are ranked strictly on in-sample and selection-validation
    results.  The last chronological block is evaluated once as a holdout and
    never influences which strategy wins.  Nothing is written to PostgreSQL.
    """
    if df is None:
        df, _ = fetch_features_range(validate=False, freshness_check=False)
    if df.empty:
        raise ValueError("No EUR/USD features available for validation")

    data = df.copy()
    data["time"] = pd.to_datetime(data["time"])
    dates = sorted(data["time"].dt.date.unique())
    if len(dates) < 10:
        raise ValueError("At least ten calendar dates are required for chronological validation")
    first_validation_date = dates[len(dates) // 2]
    holdout_date = dates[len(dates) * 3 // 4]
    in_sample = data[data["time"].dt.date < first_validation_date].reset_index(drop=True)
    selection_validation = data[
        (data["time"].dt.date >= first_validation_date)
        & (data["time"].dt.date < holdout_date)
    ].reset_index(drop=True)
    holdout = data[data["time"].dt.date >= holdout_date].reset_index(drop=True)
    walk_forward_windows = _walk_forward_windows(data, dates, holdout_date)

    results = []
    for name, candidate in CANDIDATES.items():
        strategy_type = candidate["strategy_type"]
        strategy_args = candidate["strategy"]
        simulation_args = candidate["simulation"]
        full = _run_frame(data, strategy_type, strategy_args, simulation_args)
        train = _run_frame(in_sample, strategy_type, strategy_args, simulation_args)
        validation = _run_frame(selection_validation, strategy_type, strategy_args, simulation_args)
        final_holdout = _run_frame(holdout, strategy_type, strategy_args, simulation_args)
        full_summary = _summary(full)
        train_summary = _summary(train)
        validation_summary = _summary(validation)
        holdout_summary = _summary(final_holdout)
        train_min_trades = max(5, train_summary["market_days_in_range"] // 2)
        validation_min_trades = max(5, validation_summary["market_days_in_range"] // 2)
        train_failures = _period_failures(train_summary, train_min_trades)
        validation_failures = _period_failures(validation_summary, validation_min_trades)
        walk_forward = []
        for fold in walk_forward_windows:
            fold_train = _summary(_run_frame(fold["train"], strategy_type, strategy_args, simulation_args))
            fold_validation = _summary(_run_frame(fold["validation"], strategy_type, strategy_args, simulation_args))
            fold_train_min = max(5, fold_train["market_days_in_range"] // 2)
            fold_validation_min = max(5, fold_validation["market_days_in_range"] // 2)
            fold_failures = {
                "train": _period_failures(fold_train, fold_train_min),
                "validation": _period_failures(fold_validation, fold_validation_min),
            }
            walk_forward.append({
                "period": {key: fold[key] for key in ("train_start", "validation_start", "validation_end")},
                "train": fold_train,
                "validation": fold_validation,
                "passed": not fold_failures["train"] and not fold_failures["validation"],
                "rejection_reasons": fold_failures,
            })
        walk_forward_passed = bool(walk_forward) and all(fold["passed"] for fold in walk_forward)
        selection_passed = not train_failures and not validation_failures and walk_forward_passed
        selection_reasons = []
        if train_failures:
            selection_reasons.append({"period": "train", "failures": train_failures})
        if validation_failures:
            selection_reasons.append({"period": "validation", "failures": validation_failures})
        if not walk_forward:
            selection_reasons.append({"period": "walk_forward", "failures": ["insufficient pre-holdout dates"]})
        elif not walk_forward_passed:
            selection_reasons.append({"period": "walk_forward", "failures": ["one or more chronological folds failed"]})
        results.append({
            "name": name,
            "strategy_type": strategy_type,
            "strategy": strategy_args,
            "simulation": simulation_args,
            "full": full_summary,
            "in_sample": train_summary,
            "selection_validation": validation_summary,
            "final_holdout": holdout_summary,
            "walk_forward": walk_forward,
            "walk_forward_passed": walk_forward_passed,
            "selection_passed": selection_passed,
            "selection_rejection_reasons": selection_reasons,
        })

    results.sort(key=_selection_key, reverse=True)
    best_observed = results[0] if results else None
    # A positive result on one tiny slice is not a selection.  If no
    # candidate meets both train and selection-validation gates, retaining
    # the previous production rules is safer than deploying the least bad
    # backtest result.
    selected = next((result for result in results if result["selection_passed"]), None)
    data_sufficient = len(dates) >= MINIMUM_MARKET_DAYS
    holdout_min_trades = (
        max(5, selected["final_holdout"]["market_days_in_range"] // 2)
        if selected
        else 5
    )
    holdout_failures = _period_failures(selected["final_holdout"], holdout_min_trades) if selected else ["no candidate passed selection"]
    deployment_failures = []
    if not data_sufficient:
        deployment_failures.append(f"only {len(dates)} market dates; {MINIMUM_MARKET_DAYS} required")
    if selected is None:
        deployment_failures.append("no candidate passed the train, validation and walk-forward gates")
    if selected and holdout_failures:
        deployment_failures.extend(f"holdout: {failure}" for failure in holdout_failures)
    if selected and selected["full"]["total_trades"] < MINIMUM_FULL_SAMPLE_TRADES:
        deployment_failures.append(f"full sample trades {selected['full']['total_trades']} < {MINIMUM_FULL_SAMPLE_TRADES}")
    if selected and selected["full"]["net_profit"] <= 0:
        deployment_failures.append(f"full sample net profit {selected['full']['net_profit']} <= 0")
    deployment_approved = bool(
        selected
        and data_sufficient
        and selected["selection_passed"]
        and not holdout_failures
        and selected["full"]["total_trades"] >= MINIMUM_FULL_SAMPLE_TRADES
        and selected["full"]["net_profit"] > 0
    )
    return {
        "period": {"start": str(data["time"].min()), "end": str(data["time"].max())},
        "validation_dates": {
            "selection_validation_start": str(first_validation_date),
            "final_holdout_start": str(holdout_date),
        },
        "execution_assumptions": {
            "commission_per_lot_round_turn": DEFAULT_COMMISSION_PER_LOT * 2,
            "slippage_points_each_side": DEFAULT_SLIPPAGE_POINTS,
            "uses_observed_bid_ask_spread": True,
        },
        "acceptance_criteria": {
            "minimum_market_days": MINIMUM_MARKET_DAYS,
            "minimum_full_sample_trades": MINIMUM_FULL_SAMPLE_TRADES,
            "minimum_trades_per_market_day": 1.0,
            "minimum_profit_factor": 1.10,
            "minimum_win_rate_percent": 50.0,
            "maximum_drawdown_percent": MAX_ACCEPTABLE_DRAWDOWN_PERCENT,
            "positive_net_profit_and_expectancy_required": True,
            "walk_forward_required_when_available": True,
        },
        "data_sufficient_for_deployment": data_sufficient,
        "results": results,
        "walk_forward_fold_count": len(walk_forward_windows),
        "best_observed_on_train_and_validation": best_observed["name"] if best_observed else None,
        "selected_on_train_and_validation": selected["name"] if selected else None,
        "deployment_approved": deployment_approved,
        "deployment_rejection_reasons": deployment_failures,
        "approved": [selected["name"]] if deployment_approved else [],
    }


def save_validation_report(result: dict, output_path: Path | str = DEFAULT_REPORT_PATH) -> Path:
    """Persist the comparison artefact locally; it never writes market DB data."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    _save_markdown_report(result, path.with_suffix(".md"))
    return path


def _save_markdown_report(result: dict, path: Path) -> None:
    """Write a concise, human-readable companion to the machine JSON report."""
    lines = [
        "# EUR/USD M1 scalping validation",
        "",
        f"Period: {result['period']['start']} to {result['period']['end']}",
        f"Decision: {'DEMO eligible' if result['deployment_approved'] else 'NO TRADE'}",
        f"Selected on pre-holdout data: {result['selected_on_train_and_validation'] or 'none'}",
        "",
        "| Candidate | Train P&L / PF | Validation P&L / PF | Holdout P&L / PF | Decision |",
        "|---|---:|---:|---:|---|",
    ]
    for item in result["results"]:
        train, validation, holdout = item["in_sample"], item["selection_validation"], item["final_holdout"]
        decision = "accepted" if item["selection_passed"] else "rejected: " + "; ".join(
            failure for group in item["selection_rejection_reasons"] for failure in group["failures"]
        )
        lines.append(
            f"| {item['name']} | {train['net_profit']:.2f} / {train['profit_factor']:.2f} | "
            f"{validation['net_profit']:.2f} / {validation['profit_factor']:.2f} | "
            f"{holdout['net_profit']:.2f} / {holdout['profit_factor']:.2f} | {decision} |"
        )
    lines.extend(["", "## Deployment gate", ""])
    if result["deployment_approved"]:
        lines.append("All robustness gates passed. A DEMO rollout may be considered; REAL remains a separate operational decision.")
    else:
        lines.extend(f"- {reason}" for reason in result["deployment_rejection_reasons"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    validation = validate_candidates()
    save_validation_report(validation)
    print(json.dumps(validation, indent=2, default=str))
