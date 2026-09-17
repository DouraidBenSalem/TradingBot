"""Orchestration durable du pipeline temps réel.

Ce module centralise l'état consommé par le dashboard et garde un seul objet
``TechnicalTradingBot`` pendant toute la vie du processus.  Les étapes sont
isolées afin qu'une erreur de rapport ou de backtest n'empêche jamais le
journal de monitoring d'expliquer l'état courant.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "runtime" / "pipeline_state.json"
_LOCK = threading.RLock()
_BOT = None


def _default_state() -> Dict[str, Any]:
    return {
        "status": "IDLE",
        "last_processed_timestamp": None,
        "last_update": None,
        "signal": {},
        "audit": [],
        "multi_backtests": [],
    }


def load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return _default_state()
    try:
        loaded = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _default_state()
    state = _default_state()
    state.update(loaded if isinstance(loaded, dict) else {})
    return state


def save_state(state: Dict[str, Any]) -> None:
    """Atomically publish dashboard state so readers never see partial JSON."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    temporary.replace(STATE_PATH)


def _audit(state: Dict[str, Any], component: str, status: str,
           message: str = "", action: str = "RUN") -> None:
    entries = state.setdefault("audit", [])
    entries.append({
        "time": datetime.now().isoformat(timespec="seconds"),
        "component": component,
        "action": action,
        "status": status,
        "message": message,
    })
    state["audit"] = entries[-2000:]


def _bot():
    global _BOT
    if _BOT is None:
        from ml_trading.technical_trading_bot import TechnicalTradingBot
        _BOT = TechnicalTradingBot()
    return _BOT


def _run_step(state: Dict[str, Any], component: str, function, *, fatal: bool = False):
    try:
        result = function()
        message = ""
        if component == "Stratégie technique" and isinstance(result, dict):
            message = f"Signal {result.get('signal', 0)} score {result.get('score', 0)}/8"
        _audit(state, component, "SUCCESS", message)
        return result
    except Exception as error:
        _audit(state, component, "ERROR", str(error))
        state.setdefault("errors", {})[component] = str(error)
        if fatal:
            raise
        return None


def initialize() -> Dict[str, Any]:
    """Initialiser les tables, les features et l'état affiché par l'UI."""
    with _LOCK:
        state = load_state()
        state["status"] = "RUNNING"
        state["last_update"] = datetime.now().isoformat(timespec="seconds")
        save_state(state)

        def database_setup():
            from Database.postgresql import create_tables, ensure_monitoring_tables
            create_tables()
            ensure_monitoring_tables()

        def feature_sync():
            from app.data.feature_engineering import sync_features
            return sync_features()

        def quality_report():
            from app.data.dataquality import generate_report
            return generate_report(str(ROOT / "data_quality_report.html"))

        _run_step(state, "Base de données", database_setup)
        _run_step(state, "Features", feature_sync)
        _run_step(state, "Data Quality", quality_report)
        signal = _run_step(state, "Stratégie technique", lambda: _bot().update())
        if isinstance(signal, dict):
            state["signal"] = signal
        state["status"] = "RUNNING"
        state["last_update"] = datetime.now().isoformat(timespec="seconds")
        save_state(state)
        return state


def on_new_candle(candle_time: Optional[Any] = None) -> Dict[str, Any]:
    """Mettre à jour les features et le bot une seule fois par bougie fermée."""
    with _LOCK:
        state = load_state()
        candle_text = str(candle_time) if candle_time is not None else None
        if candle_text and state.get("last_processed_timestamp") == candle_text:
            return state
        state["status"] = "RUNNING"
        state["last_processed_timestamp"] = candle_text
        state["last_update"] = datetime.now().isoformat(timespec="seconds")
        save_state(state)

        def feature_sync():
            from app.data.feature_engineering import sync_features
            return sync_features([candle_time] if candle_time is not None else None)

        _run_step(state, "Features", feature_sync)
        signal = _run_step(state, "Stratégie technique", lambda: _bot().update())
        if isinstance(signal, dict):
            state["signal"] = signal
        state["status"] = "RUNNING" if not str((signal or {}).get("status", "")).lower().startswith("error") else "ERROR"
        state["last_update"] = datetime.now().isoformat(timespec="seconds")
        save_state(state)
        return state
