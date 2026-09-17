"""Deployment gate for the EUR/USD M1 scalping bot.

The live signal generator may only use a strategy explicitly approved by the
chronological validation report.  This prevents a locally attractive but
unvalidated parameter set from becoming the default trading behaviour.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ml_trading.technical_strategy import build_strategy


ROOT = Path(__file__).resolve().parent.parent
VALIDATION_REPORT_PATH = ROOT / "reports" / "scalping_validation_report.json"

# This is a deliberately fixed DEMO experiment, not a fallback production
# strategy.  Keeping it here makes the exact live rule set reviewable.
DEMO_CANDIDATE_NAME = "ema_pullback_wider_stop"
DEMO_CANDIDATE = {
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
}


def load_demo_strategy():
    """Build the one explicitly authorised strategy for an MT5 DEMO trial.

    It never marks the candidate as REAL-approved: the historical validation
    report remains authoritative for a future production decision.
    """
    strategy = build_strategy(DEMO_CANDIDATE["strategy_type"], DEMO_CANDIDATE["strategy"])
    return strategy, {
        "approved": False,
        "demo_candidate": DEMO_CANDIDATE_NAME,
        "strategy_type": DEMO_CANDIDATE["strategy_type"],
        "simulation": DEMO_CANDIDATE["simulation"].copy(),
        "fixed_lot": float(os.getenv("MT5_DEMO_LOT", "0.07")),
        "mode": "DEMO",
        "reason": "Explicit MT5 DEMO observation; not approved for REAL trading",
    }


class ValidationGuardStrategy:
    """Return no signal while no strategy has passed deployment validation."""

    NAME = "Scalping validation guard (no approved strategy)"

    def __init__(self, reason: str):
        self.reason = reason

    def parameters(self):
        return {"name": self.NAME, "reason": self.reason}

    def evaluate_batch(self, df):
        result = df.copy()
        result["signal"] = 0
        result["score"] = 0
        detail = {"validation": self.reason}
        result["signal_details"] = [detail.copy() for _ in range(len(result))]
        return result


def load_deployed_strategy(report_path: Path | str = VALIDATION_REPORT_PATH):
    """Load an approved report and build its exact declared strategy.

    The report is intentionally treated as invalid unless it explicitly says
    it is deployable and contains a matching candidate definition.  Failure
    is fail-closed: the bot emits NO TRADE rather than falling back to an
    unvalidated strategy.
    """
    path = Path(report_path)
    if not path.exists():
        reason = "No scalping validation report found; deployment is blocked"
        return ValidationGuardStrategy(reason), {"approved": False, "reason": reason}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        reason = f"Unreadable scalping validation report: {error}"
        return ValidationGuardStrategy(reason), {"approved": False, "reason": reason}

    selected_name = report.get("selected_on_train_and_validation")
    if not report.get("deployment_approved") or not selected_name:
        reason = "No strategy passed the validation and holdout deployment gates"
        return ValidationGuardStrategy(reason), {
            "approved": False,
            "reason": reason,
            "selected_candidate": selected_name,
            "report_period": report.get("period"),
        }

    candidate = next(
        (item for item in report.get("results", []) if item.get("name") == selected_name),
        None,
    )
    if not candidate:
        reason = f"Approved candidate '{selected_name}' is missing from the report"
        return ValidationGuardStrategy(reason), {"approved": False, "reason": reason}
    try:
        strategy = build_strategy(candidate["strategy_type"], candidate.get("strategy"))
    except (KeyError, TypeError, ValueError) as error:
        reason = f"Approved candidate cannot be constructed: {error}"
        return ValidationGuardStrategy(reason), {"approved": False, "reason": reason}
    return strategy, {
        "approved": True,
        "selected_candidate": selected_name,
        "strategy_type": candidate["strategy_type"],
        # These values are part of the validated rule set, not adjustable
        # live defaults.  The executor must apply them to the actual fill.
        "simulation": candidate.get("simulation", {}).copy(),
        "fixed_lot": float(os.getenv("MT5_DEMO_LOT", "0.07")),
        "report_period": report.get("period"),
    }
