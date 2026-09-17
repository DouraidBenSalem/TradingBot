"""
Collecteur de signaux temps réel et gestionnaire de trades pour la stratégie technique.

Ce module :
1. Récupère les features pré-calculées depuis la table eurusd_features
2. Génère les signaux en temps réel (sans recalcul d'indicateurs)
3. Gère les trades selon les règles de risque
4. Assure le suivi quotidien MT5 DEMO (obs-only, pas d'effet sur la stratégie)
"""

import atexit
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from ml_trading.features_provider import fetch_latest_features
from ml_trading.scalping_deployment import load_demo_strategy, load_deployed_strategy
from Database.postgresql import (
    ensure_daily_sessions_table,
    get_daily_session_by_date,
    upsert_daily_session,
    close_daily_session,
    insert_bot_decision,
    record_bot_event,
    upsert_mt5_trade,
)
from ml_trading import mt5_demo
from ml_trading.daily_monitoring import (
    categorize_signal_rejection,
    top_reason,
    build_no_trade_message,
    build_trade_message,
)


ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "runtime" / "trading_state.json"
TRADE_LOG_FILE = ROOT / "runtime" / "trade_log.json"

_DAILY_SESSION_STRATEGY = "ema_pullback_wider_stop"
_DAILY_SESSION_SYMBOL = "EURUSD"
_DAILY_SESSION_LOT = mt5_demo.DEMO_LOT
_MAX_TRADES_PER_DAY = min(3, max(1, int(os.getenv("MT5_MAX_TRADES_PER_DAY", "3"))))
_BOT_START_RECORDED = False


def _as_datetime(value: Any) -> Optional[datetime]:
    """Convert MT5/JSON timestamps to a naive local datetime."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    if hasattr(value, "to_pydatetime"):
        converted = value.to_pydatetime()
        return converted.replace(tzinfo=None) if converted.tzinfo else converted
    if not value:
        return None
    try:
        converted = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return converted.replace(tzinfo=None) if converted.tzinfo else converted
    except (TypeError, ValueError):
        return None


def evaluate_daily_trade_policy(
    records: list[Dict[str, Any]],
    current_time: Any,
    max_trades: int = _MAX_TRADES_PER_DAY,
) -> Dict[str, Any]:
    """Evaluate the daily stop rule from trades actually opened in MT5.

    Rejected signals never enter ``records`` and therefore never consume one
    of the three daily slots.  Net P/L uses closed positions only.
    """
    current_dt = _as_datetime(current_time) or datetime.now()
    limit = min(3, max(1, int(max_trades or 3)))
    by_ticket: Dict[str, Dict[str, Any]] = {}
    for index, record in enumerate(records or []):
        opened_at = _as_datetime(record.get("opened_at") or record.get("entry_time"))
        if opened_at is None or opened_at.date() != current_dt.date():
            continue
        ticket = record.get("ticket") or record.get("position_id") or record.get("order_ticket")
        key = str(ticket) if ticket is not None else f"record-{index}"
        existing = by_ticket.get(key)
        # Prefer the reconciled CLOSED version if an older OPEN snapshot exists.
        if existing is None or str(record.get("status") or "").upper() == "CLOSED":
            by_ticket[key] = record

    trades = list(by_ticket.values())
    closed = [
        trade for trade in trades
        if str(trade.get("status") or "").upper() == "CLOSED"
        and trade.get("profit_loss") is not None
    ]
    open_count = sum(1 for trade in trades if str(trade.get("status") or "").upper() == "OPEN")
    net_pnl = round(sum(float(trade.get("profit_loss") or 0.0) for trade in closed), 2)
    executed_count = len(trades)
    winning_trades = sum(1 for trade in closed if float(trade.get("profit_loss") or 0.0) > 0)
    losing_trades = sum(1 for trade in closed if float(trade.get("profit_loss") or 0.0) < 0)

    stop_reason = None
    stop_message = None
    if executed_count >= limit:
        stop_reason = "MAX_TRADES_PER_DAY"
        stop_message = f"Limite journaliere atteinte ({limit} trades executes)."
    elif closed and net_pnl > 0:
        stop_reason = "DAILY_NET_TARGET_REACHED"
        stop_message = f"Resultat net journalier positif atteint ({net_pnl:+.2f})."

    recovery_target = None
    if executed_count == 2 and len(closed) == 2 and net_pnl <= 0 and stop_reason is None:
        # Smallest cent-denominated profit that would make the day net positive.
        recovery_target = round(abs(net_pnl) + 0.01, 2)

    block_reason = stop_reason
    if block_reason is None and open_count:
        block_reason = "POSITION_ALREADY_OPEN"

    return {
        "date": current_dt.date().isoformat(),
        "max_trades": limit,
        "executed_trades": executed_count,
        "closed_trades": len(closed),
        "open_trades": open_count,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "net_pnl": net_pnl,
        "stopped": stop_reason is not None,
        "stop_reason": stop_reason,
        "stop_message": stop_message,
        "can_open_new_trade": block_reason is None,
        "block_reason": block_reason,
        "next_trade_number": executed_count + 1 if block_reason is None else None,
        "recovery_target": recovery_target,
    }


class TechnicalTradingBot:
    """Bot de trading utilisant la stratégie technique pure."""
    
    def __init__(self):
        # The live bot is fail-closed: it uses exactly the declared strategy
        # that passed train/validation/holdout checks, or emits NO TRADE.
        self.demo_mode = os.getenv("MT5_DEMO_MODE", "false").strip().lower() == "true"
        self.strategy, self.deployment = (
            load_demo_strategy() if self.demo_mode else load_deployed_strategy()
        )
        self.state = self._load_state()
        self.trade_log = self._load_trade_log()
        self._daily_session_initialized = False
        self._daily_session_id: Optional[int] = None
        self._atexit_registered = False
        global _BOT_START_RECORDED
        if not _BOT_START_RECORDED:
            started_at = datetime.now().isoformat(timespec="seconds")
            self.state["bot_started_at"] = started_at
            self.state["last_bot_activity"] = started_at
            try:
                record_bot_event("BOT_STARTED", "Bot started", component="BOT")
            except Exception as error:
                self.state["monitoring_persistence_error"] = str(error)
            self._save_state()
            _BOT_START_RECORDED = True
        if not self._atexit_registered:
            try:
                atexit.register(self._daily_session_atexit_cleanup)
                self._atexit_registered = True
            except Exception:
                pass
    
    def _load_state(self):
        """Charger l'état du bot."""
        if not STATE_FILE.exists():
            return {
                "last_update": None,
                "open_trades": [],
                "trades_today": 0,
                "pnl_today": 0.0,
                "balance": 100.0,
                "status": "READY"
            }
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    
    def _save_state(self):
        """Sauvegarder l'état du bot."""
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
    
    def _load_trade_log(self):
        """Charger l'historique des trades."""
        if not TRADE_LOG_FILE.exists():
            return []
        with open(TRADE_LOG_FILE, 'r') as f:
            return json.load(f)
    
    def _save_trade_log(self):
        """Sauvegarder l'historique des trades."""
        TRADE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TRADE_LOG_FILE, 'w') as f:
            json.dump(self.trade_log, f, indent=2, default=str)
    
    def _get_latest_data(self, lookback_candles=100, freshness_check=True):
        """Récupérer les dernières bougies avec leurs features pré-calculées.

        Les indicateurs techniques ne sont plus recalculés : ils sont
        chargés directement depuis la table eurusd_features.
        """
        df, validation_report = fetch_latest_features(
            lookback_candles=lookback_candles,
            validate=True,
            raise_on_invalid=False,
            freshness_check=freshness_check,
        )
        self._last_validation = validation_report
        return df

    def _risk_levels(self, signal, close, atr):
        """Apply the selected candidate's validated ATR stop/target settings."""
        simulation = self.deployment.get("simulation", {})
        sl_multiple = float(simulation.get("sl_atr_multiplier", 1.0))
        tp_multiple = float(simulation.get("tp_atr_multiplier", 2.0))
        if signal == 1:
            return close - sl_multiple * atr, close + tp_multiple * atr
        if signal == 2:
            return close + sl_multiple * atr, close - tp_multiple * atr
        return None, None

    def _record_decision(self, current_time, signal, score, latest, sl, tp,
                         action="NO_TRADE", refusal_reason=None,
                         entry_reason=None, execution=None):
        """Persist one decision per completed candle for dashboard/audit use.

        It is an audit record.  DEMO routing is handled separately and can
        only be activated by its explicit environment flag.
        """
        if self.state.get("last_decision_time") == str(current_time):
            return None
        signal_label = "BUY" if signal == 1 else "SELL" if signal == 2 else "NO_SIGNAL"
        action = (action or "NO_TRADE").upper().replace(" ", "_")
        decision_label = f"{action.replace('_', ' ')} EXECUTED" if action in ("BUY", "SELL") else "NO TRADE"
        conditions = dict(latest["signal_details"] or {})
        decision = {
            "occurred_at": str(current_time),
            "symbol": mt5_demo.DEMO_SYMBOL if self.demo_mode else "EURUSD",
            "timeframe": "M1",
            "current_price": float(latest["close"]),
            "signal": signal_label,
            "signal_score": score,
            "conditions": conditions,
            "decision": decision_label,
            "action": action,
            "refusal_reason": refusal_reason,
            "entry_reason": entry_reason,
            "stop_loss": round(sl, 5) if sl is not None else None,
            "take_profit": round(tp, 5) if tp is not None else None,
            "lot_size": float((execution or {}).get("lot") or _DAILY_SESSION_LOT),
            "trade_ticket": (execution or {}).get("ticket"),
            "status": "EXECUTED" if action in ("BUY", "SELL") else "REJECTED",
            "diagnostics": {
                "ema_9": float(latest["ema_9"]),
                "ema_21": float(latest["ema_21"]),
                "ema_50": float(latest["ema_50"]),
                "rsi_14": float(latest["rsi_14"]),
                "macd": float(latest["macd"]),
                "macd_hist": float(latest["macd_hist"]),
                "atr_14": float(latest["atr_14"]),
                "execution": execution or {},
                "deployment": self.deployment,
            },
        }
        try:
            decision_id = insert_bot_decision(decision)
            self.state["last_decision_time"] = str(current_time)
            self.state["last_bot_activity"] = datetime.now().isoformat(timespec="seconds")
            if signal in (1, 2):
                record_bot_event(
                    "SIGNAL_DETECTED",
                    f"{signal_label} signal detected (score {score}/8)",
                    component="STRATEGY", symbol=decision["symbol"],
                    details={"decision_id": decision_id, "conditions": conditions},
                    occurred_at=current_time,
                )
            event_type = "TRADE_EXECUTED" if action in ("BUY", "SELL") else "DECISION_NO_TRADE"
            message = entry_reason if action in ("BUY", "SELL") else refusal_reason
            record_bot_event(
                event_type,
                message or decision_label,
                severity="INFO",
                component="STRATEGY",
                symbol=decision["symbol"],
                ticket=decision["trade_ticket"],
                details={"decision_id": decision_id, "signal": signal_label, "score": score},
                occurred_at=current_time,
            )
            return decision_id
        except Exception as error:
            self.state["last_decision_log_error"] = str(error)
            return None

    def _sync_demo_history(self):
        """Persist newly closed MT5 DEMO deals and calculate observed metrics."""
        if not self.demo_mode:
            return
        since = datetime.now() - timedelta(days=30)
        records = mt5_demo.trade_history(since)
        seen = set(self.state.get("demo_seen_position_tickets", []))
        closed = []
        for item in records:
            ticket = int(item["ticket"])
            upsert_mt5_trade(item)
            if item.get("status") == "CLOSED" and item.get("profit_loss") is not None:
                closed.append({"net_pnl": float(item["profit_loss"]), **item})
                if ticket not in seen:
                    record_bot_event(
                        "TRADE_CLOSED",
                        f"Trade #{ticket} closed: {float(item['profit_loss']):+.2f}",
                        component="EXECUTION", symbol=item.get("symbol"), ticket=ticket,
                        details=item,
                    )
                    seen.add(ticket)
        history_fields = (
            "ticket", "position_id", "order_ticket", "opened_at", "closed_at",
            "symbol", "side", "profit_loss", "commission", "swap", "status",
        )
        self.state["demo_trade_history"] = [
            {key: item.get(key) for key in history_fields} for item in records[-2000:]
        ]
        self.state["demo_seen_position_tickets"] = list(seen)[-2000:]
        self.state["demo_closed_trades"] = closed[-2000:]
        self.state["demo_history_since"] = datetime.now().isoformat(timespec="seconds")
        wins = [trade["net_pnl"] for trade in closed if trade["net_pnl"] > 0]
        losses = [trade["net_pnl"] for trade in closed if trade["net_pnl"] < 0]
        gross_profit, gross_loss = sum(wins), abs(sum(losses))
        equity = [sum(trade["net_pnl"] for trade in closed[:index + 1]) for index in range(len(closed))]
        peak, drawdown = 0.0, 0.0
        for value in equity:
            peak = max(peak, value)
            drawdown = min(drawdown, value - peak)
        self.state["demo_metrics"] = {
            "total_trades": len(closed), "winning_trades": len(wins), "losing_trades": len(losses),
            "win_rate": round(100 * len(wins) / len(closed), 2) if closed else 0.0,
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else 0.0,
            "expectancy": round(sum(trade["net_pnl"] for trade in closed) / len(closed), 2) if closed else 0.0,
            "net_pnl": round(sum(trade["net_pnl"] for trade in closed), 2),
            "max_drawdown": round(drawdown, 2),
        }

    def _daily_trade_policy(self, current_time: Any) -> Dict[str, Any]:
        policy = evaluate_daily_trade_policy(
            self.state.get("demo_trade_history") or [],
            current_time,
            _MAX_TRADES_PER_DAY,
        )
        self.state["daily_trade_policy"] = policy
        day_key = policy["date"]
        self.state.setdefault("trades_per_day", {})[day_key] = {
            "count": policy["executed_trades"],
            "pnl": policy["net_pnl"],
            "stopped": policy["stopped"],
            "stop_reason": policy["stop_reason"],
            "closed_count": policy["closed_trades"],
            "open_count": policy["open_trades"],
            "recovery_target": policy["recovery_target"],
        }
        return policy

    def _record_daily_stop_if_needed(self, policy: Dict[str, Any]) -> None:
        reason = policy.get("stop_reason")
        if not reason:
            return
        event_key = f"{policy.get('date')}:{reason}"
        if self.state.get("daily_stop_event_key") == event_key:
            return
        try:
            record_bot_event(
                "DAILY_TRADING_STOPPED",
                policy.get("stop_message") or "Trading journalier arrete.",
                component="RISK",
                symbol=mt5_demo.DEMO_SYMBOL,
                details=policy,
            )
            self.state["daily_stop_event_key"] = event_key
        except Exception as error:
            self.state["monitoring_persistence_error"] = str(error)

    def _daily_session_account_info(self) -> Dict[str, Any]:
        account_login = self.state.get("mt5_demo", {}).get("account_login") or 0
        account_server = self.state.get("mt5_demo", {}).get("account_server")
        account_is_demo = bool(self.state.get("mt5_demo", {}).get("account_is_demo", False))
        return {
            "account_login": int(account_login) if account_login else 0,
            "account_server": account_server,
            "account_is_demo": account_is_demo,
        }

    def _ensure_daily_session(self, current_time: datetime) -> Optional[int]:
        if not self.demo_mode:
            return None
        try:
            ensure_daily_sessions_table()
        except Exception as exc:
            self.state["daily_session_init_error"] = str(exc)
            return None
        info = self._daily_session_account_info()
        existing = get_daily_session_by_date(
            current_time,
            info.get("account_login") or 0,
            _DAILY_SESSION_STRATEGY,
            _DAILY_SESSION_SYMBOL,
        )
        now = datetime.now()
        if existing:
            prev_status = existing.get("status") or ""
            prev_date = existing.get("session_date")
            same_day = (
                prev_date is not None
                and current_time.date() == (prev_date.date() if hasattr(prev_date, "date") else prev_date)
            )
            if same_day and prev_status in ("RUNNING", "ERROR", "TRADE", "NO_TRADE", "STOPPED"):
                self._daily_session_id = int(existing["id"])
                if prev_status == "ERROR":
                    try:
                        upsert_daily_session({
                            "session_date": current_time.date(),
                            "account_login": info.get("account_login") or 0,
                            "account_server": info.get("account_server"),
                            "account_is_demo": bool(info.get("account_is_demo")),
                            "strategy": _DAILY_SESSION_STRATEGY,
                            "symbol": _DAILY_SESSION_SYMBOL,
                            "lot": _DAILY_SESSION_LOT,
                            "status": "RUNNING",
                            "message": "Reprise après une session précédente en état ERROR.",
                        })
                    except Exception:
                        pass
                self._daily_session_initialized = True
                return self._daily_session_id
            elif not same_day and prev_status == "RUNNING":
                try:
                    close_daily_session(
                        int(existing["id"]),
                        overrides={"status": "ERROR", "message": "Session laissée ouverte après crash/redémarrage."},
                    )
                except Exception:
                    pass
        opened_at = now
        partial: Dict[str, Any] = {
            "session_date": current_time.date(),
            "opened_at": opened_at,
            "status": "RUNNING",
            "strategy": _DAILY_SESSION_STRATEGY,
            "symbol": _DAILY_SESSION_SYMBOL,
            "lot": _DAILY_SESSION_LOT,
            "account_login": int(info.get("account_login") or 0),
            "account_server": info.get("account_server"),
            "account_is_demo": bool(info.get("account_is_demo")),
            "total_signals": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "rejected_signals": 0,
            "trades_opened": 0,
            "trades_won": 0,
            "trades_lost": 0,
            "trades_cancelled": 0,
            "pnl": 0.0,
            "rejected_breakdown": {},
            "message": "Session MT5 DEMO RUNNING.",
        }
        try:
            sid = upsert_daily_session(partial)
            if sid and sid > 0:
                self._daily_session_id = sid
                self._daily_session_initialized = True
                return sid
        except Exception as exc:
            self.state["daily_session_init_error"] = str(exc)
        return None

    def _update_daily_session_cycle(
        self,
        current_time: datetime,
        signal_value: int,
        score: int,
        signal_details: Dict[str, Any],
        execution: Optional[Dict[str, Any]] = None,
        latest_candle: Optional[Dict[str, Any]] = None,
        daily_policy: Optional[Dict[str, Any]] = None,
        refusal_reason: Optional[str] = None,
    ) -> None:
        if not self.demo_mode:
            return
        if not self._daily_session_initialized or self._daily_session_id is None:
            self._ensure_daily_session(current_time)
            if self._daily_session_id is None:
                return
        info = self._daily_session_account_info()
        try:
            current = get_daily_session_by_date(
                current_time,
                info.get("account_login") or 0,
                _DAILY_SESSION_STRATEGY,
                _DAILY_SESSION_SYMBOL,
            )
        except Exception:
            current = None
        row: Dict[str, Any] = dict(current) if current else {}

        total_signals = int(row.get("total_signals") or 0) + 1
        buy_signals = int(row.get("buy_signals") or 0) + (1 if int(signal_value or 0) == 1 else 0)
        sell_signals = int(row.get("sell_signals") or 0) + (1 if int(signal_value or 0) == 2 else 0)

        submitted = bool((execution or {}).get("submitted"))
        rejection_reason = categorize_signal_rejection(
            signal_value=int(signal_value or 0),
            score=int(score or 0),
            signal_details=signal_details or {},
            execution_reason=(execution or {}).get("reason") or refusal_reason,
            execution_submitted=submitted,
        )
        rejected_signals = int(row.get("rejected_signals") or 0)
        if rejection_reason is not None:
            rejected_signals += 1

        rejected_breakdown: Dict[str, int] = dict(row.get("rejected_breakdown") or {})
        if rejection_reason is not None:
            rejected_breakdown[rejection_reason] = int(rejected_breakdown.get(rejection_reason) or 0) + 1
        no_trade_reason = top_reason(rejected_breakdown)

        last_signal = {
            "time": current_time.isoformat(timespec="seconds"),
            "signal": int(signal_value or 0),
            "signal_label": (
                "BUY" if int(signal_value or 0) == 1
                else "SELL" if int(signal_value or 0) == 2
                else "NO_TRADE"
            ),
            "score": int(score or 0),
            "submitted": submitted,
            "rejection_reason": rejection_reason,
            "candle": latest_candle or {},
            "signal_details": (signal_details or {}),
            "execution_reason": (execution or {}).get("reason") or refusal_reason,
        }

        policy = daily_policy or self._daily_trade_policy(current_time)
        trades_opened = int(policy.get("executed_trades") or 0)
        trades_won = int(policy.get("winning_trades") or 0)
        trades_lost = int(policy.get("losing_trades") or 0)
        pnl = float(policy.get("net_pnl") or 0.0)
        last_signal["daily_trade_policy"] = policy

        if trades_opened > 0:
            message = build_trade_message(
                total_signals=total_signals,
                trades_opened=trades_opened,
                trades_won=trades_won,
                trades_lost=trades_lost,
                pnl=pnl,
            )
            final_no_trade_reason = None
        else:
            final_no_trade_reason = no_trade_reason
            message = build_no_trade_message(
                total_signals=total_signals,
                buy_signals=buy_signals,
                sell_signals=sell_signals,
                rejected_signals=rejected_signals,
                rejected_breakdown=rejected_breakdown,
                no_trade_reason=no_trade_reason,
            )

        partial: Dict[str, Any] = {
            "session_date": current_time.date(),
            "strategy": _DAILY_SESSION_STRATEGY,
            "symbol": _DAILY_SESSION_SYMBOL,
            "lot": _DAILY_SESSION_LOT,
            "account_login": int(info.get("account_login") or 0),
            "account_server": info.get("account_server"),
            "account_is_demo": bool(info.get("account_is_demo")),
            "total_signals": total_signals,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "rejected_signals": rejected_signals,
            "trades_opened": trades_opened,
            "trades_won": trades_won,
            "trades_lost": trades_lost,
            "pnl": pnl,
            "no_trade_reason": final_no_trade_reason,
            "rejected_breakdown": rejected_breakdown,
            "last_signal": last_signal,
            "message": message,
            "status": "STOPPED" if policy.get("stopped") else ("TRADE" if trades_opened else "NO_TRADE"),
        }
        try:
            self._daily_session_id = upsert_daily_session(partial) or self._daily_session_id
        except Exception as exc:
            self.state["daily_session_update_error"] = str(exc)

    def _daily_session_atexit_cleanup(self) -> None:
        if not self.demo_mode:
            return
        try:
            record_bot_event("BOT_STOPPED", "Bot stopped", component="BOT")
        except Exception:
            pass
        if self._daily_session_id is None:
            return
        try:
            info = self._daily_session_account_info()
            policy = self._daily_trade_policy(datetime.now())
            overrides: Dict[str, Any] = {
                "account_login": int(info.get("account_login") or 0),
                "account_server": info.get("account_server"),
                "account_is_demo": bool(info.get("account_is_demo")),
                "trades_opened": int(policy.get("executed_trades") or 0),
                "trades_won": int(policy.get("winning_trades") or 0),
                "trades_lost": int(policy.get("losing_trades") or 0),
                "pnl": float(policy.get("net_pnl") or 0.0),
            }
            close_daily_session(int(self._daily_session_id), overrides=overrides)
        except Exception:
            return

    def update(self, freshness_check=True):
        """
        Mettre à jour l'état du bot et générer les signaux.

        Args:
            freshness_check: Si True, vérifie la fraîcheur des données.
                Mettre False pour les tests / usages historiques.
        
        Returns:
            {
                'signal': 0/1/2 (NO TRADE / BUY / SELL),
                'details': {...},
                'trades_executed': [...],
                'status': str
            }
        """
        try:
            df = self._get_latest_data(lookback_candles=100, freshness_check=freshness_check)
            if df.empty:
                self.state['status'] = 'ERROR: Aucune feature disponible (eurusd_features vide)'
                self._save_state()
                return {'signal': 0, 'details': {}, 'status': 'No data', 'trades_executed': []}
            
            df_with_signals = self.strategy.evaluate_batch(df)
            
            latest = df_with_signals.iloc[-1]
            current_time = latest['time']
            signal = int(latest['signal'])
            score = int(latest['score'])
            close = float(latest['close'])
            atr = float(latest['atr_14'])
            sl, tp = self._risk_levels(signal, close, atr)
            
            self.state['last_update'] = str(current_time)
            self.state['strategy_deployment'] = self.deployment
            self.state['trading_mode'] = 'DEMO' if self.demo_mode else ('DEMO_SIGNAL_ONLY' if self.deployment.get('approved') else 'NO_TRADE')
            if self.demo_mode:
                self.state['mt5_demo'] = mt5_demo.market_snapshot()
                if self.state['mt5_demo'].get('connected'):
                    self._sync_demo_history()
                for position in self.state['mt5_demo'].get('positions') or []:
                    try:
                        record_bot_event(
                            'POSITION_MONITORING',
                            f"Monitoring {position.get('side', 'position')} #{position.get('ticket')}: "
                            f"P/L {float(position.get('profit') or 0):+.2f}",
                            component='EXECUTION',
                            symbol=position.get('symbol') or mt5_demo.DEMO_SYMBOL,
                            ticket=position.get('ticket'),
                            details=position,
                        )
                    except Exception as error:
                        self.state['monitoring_persistence_error'] = str(error)
            decision_details = dict(latest['signal_details'] or {})
            decision_details.update({
                'deployment_mode': self.state['trading_mode'],
                'selected_strategy': self.deployment.get('demo_candidate') or self.deployment.get('selected_candidate'),
                'sl': round(sl, 5) if sl is not None else None,
                'tp': round(tp, 5) if tp is not None else None,
                'mt5_order_routing': 'enabled only when MT5_DEMO_EXECUTION_ENABLED=true' if self.demo_mode else 'disabled',
            })
            if getattr(self, '_last_validation', None):
                self.state['last_features_validation'] = self._last_validation.to_dict()
            self.state['status'] = 'UPDATED'
            
            current_dt = _as_datetime(current_time) or datetime.now()
            daily_policy = self._daily_trade_policy(current_dt)
            self._record_daily_stop_if_needed(daily_policy)
            decision_details['daily_trade_policy'] = daily_policy
            
            # This process is intentionally signal-only.  No MT5 order is
            # represented as executed until a dedicated, broker-checked
            # executor is enabled after DEMO approval.
            trades_executed = []
            signals_ready = []
            latest = latest.copy()
            latest['signal_details'] = decision_details

            last_entry_time = self.state.get('last_entry_time')
            duplicate_candle = last_entry_time == str(current_time)

            execution: Optional[Dict[str, Any]] = None
            trade_signal: Optional[Dict[str, Any]] = None
            action = 'NO_TRADE'
            refusal_reason: Optional[str] = None
            entry_reason: Optional[str] = None

            if signal == 0:
                reason_code = categorize_signal_rejection(signal, score, decision_details)
                refusal_reason = (
                    f"Signal conditions were checked but confirmation was insufficient ({reason_code})."
                )
            elif duplicate_candle:
                refusal_reason = 'An opportunity on this candle has already been processed.'
            elif daily_policy['stopped']:
                refusal_reason = (
                    f"{daily_policy['stop_reason']}: {daily_policy['stop_message']} "
                    "No additional position is allowed today."
                )
            elif daily_policy['open_trades'] > 0:
                refusal_reason = 'POSITION_ALREADY_OPEN: A strategy position is still open.'
            else:
                trade_number = int(daily_policy['executed_trades']) + 1
                recovery_target = daily_policy.get('recovery_target')
                trade_signal = {
                    'time': str(current_time),
                    'signal': signal,
                    'side': 'BUY' if signal == 1 else 'SELL',
                    'score': score,
                    'entry_price': close,
                    'sl': round(sl, 5),
                    'tp': round(tp, 5),
                    'lot': _DAILY_SESSION_LOT,
                    'ema_9': float(latest['ema_9']),
                    'ema_21': float(latest['ema_21']),
                    'ema_50': float(latest['ema_50']),
                    'rsi': float(latest['rsi_14']),
                    'macd': float(latest['macd']),
                    'atr': float(latest['atr_14']),
                    'trade_number_today': trade_number,
                    'daily_net_before_entry': daily_policy['net_pnl'],
                    'daily_recovery_target': recovery_target,
                    'details': latest['signal_details']
                }
                signals_ready.append(trade_signal)
                if self.demo_mode:
                    try:
                        execution = mt5_demo.execute_signal(
                            signal,
                            atr,
                            max_spread_to_atr=float(self.strategy.max_spread_to_atr),
                        )
                    except Exception as error:
                        execution = {"submitted": False, "reason": f"MT5 connection/execution error: {error}"}
                        try:
                            record_bot_event(
                                'EXECUTION_ERROR', str(error), severity='ERROR',
                                component='EXECUTION', symbol=mt5_demo.DEMO_SYMBOL,
                                details={'signal': signal, 'score': score},
                            )
                        except Exception:
                            pass
                    trade_signal['mt5_demo_execution'] = execution
                    if execution.get('submitted'):
                        action = 'BUY' if signal == 1 else 'SELL'
                        entry_reason = (
                            f"{action} executed because the signal conditions and validation filters "
                            f"were confirmed (score {score}/8). Trade {trade_number}/{_MAX_TRADES_PER_DAY} today."
                        )
                        if recovery_target is not None:
                            entry_reason += (
                                f" Third and final trade: daily recovery objective {recovery_target:.2f}; "
                                "the validated lot, SL and TP remain unchanged."
                            )
                    else:
                        refusal_reason = execution.get('reason') or 'MT5 refused the order.'
                else:
                    refusal_reason = 'Order routing is disabled outside explicit MT5 DEMO mode.'
                self.state['last_entry_time'] = str(current_time)

            decision_id = self._record_decision(
                current_time, signal, score, latest, sl, tp,
                action=action,
                refusal_reason=refusal_reason,
                entry_reason=entry_reason,
                execution=execution,
            )
            if trade_signal is not None:
                trade_signal['decision_id'] = decision_id
            if execution and execution.get('submitted'):
                open_trade = {
                    'ticket': execution['ticket'],
                    'position_id': execution.get('position_id') or execution['ticket'],
                    'order_ticket': execution.get('order_ticket') or execution['ticket'],
                    'opened_at': str(current_time),
                    'symbol': mt5_demo.DEMO_SYMBOL,
                    'side': action,
                    'entry_price': execution['entry'],
                    'volume': execution['lot'],
                    'stop_loss': execution['sl'],
                    'take_profit': execution['tp'],
                    'status': 'OPEN',
                    'entry_reason': entry_reason,
                    'decision_id': decision_id,
                    'execution': execution,
                }
                try:
                    upsert_mt5_trade(open_trade)
                except Exception as error:
                    self.state['trade_persistence_error'] = str(error)
                trades_executed.append(open_trade)
                history = list(self.state.get('demo_trade_history') or [])
                history = [
                    item for item in history
                    if str(item.get('ticket')) != str(open_trade.get('ticket'))
                ]
                history.append(open_trade)
                self.state['demo_trade_history'] = history[-2000:]
                daily_policy = self._daily_trade_policy(current_dt)
                self._record_daily_stop_if_needed(daily_policy)

            self._save_state()

            latest_candle = {
                'time': str(current_time),
                'close': close,
                'ema_9': float(latest['ema_9']),
                'ema_21': float(latest['ema_21']),
                'ema_50': float(latest['ema_50']),
                'rsi': float(latest['rsi_14']),
                'macd': float(latest['macd']),
                'atr': float(latest['atr_14']),
            }
            try:
                self._update_daily_session_cycle(
                    current_time=current_dt,
                    signal_value=int(signal or 0),
                    score=int(score or 0),
                    signal_details=dict(latest['signal_details'] or {}),
                    execution=execution,
                    latest_candle=latest_candle,
                    daily_policy=daily_policy,
                    refusal_reason=refusal_reason,
                )
            except Exception as exc:
                self.state['daily_session_update_error'] = str(exc)
                self._save_state()
            
            return {
                'signal': signal,
                'score': score,
                'latest_candle': latest_candle,
                'details': latest['signal_details'],
                'sl': round(sl, 5) if sl is not None else None,
                'tp': round(tp, 5) if tp is not None else None,
                'trading_mode': self.state['trading_mode'],
                'decision_id': decision_id,
                'mt5_demo': self.state.get('mt5_demo'),
                'demo_metrics': self.state.get('demo_metrics', {}),
                'daily_trade_policy': daily_policy,
                'strategy_deployment': self.deployment,
                'status': 'OK',
                'trades_executed': trades_executed,
                'signals_ready': signals_ready,
            }
        
        except Exception as e:
            self.state['status'] = f'ERROR: {str(e)}'
            self.state['last_bot_activity'] = datetime.now().isoformat(timespec='seconds')
            try:
                record_bot_event('BOT_ERROR', str(e), severity='ERROR', component='BOT')
            except Exception:
                pass
            self._save_state()
            return {'signal': 0, 'details': {'error': str(e)}, 'status': 'Error', 'trades_executed': []}


def run():
    """Exécuter une mise à jour du bot de trading."""
    bot = TechnicalTradingBot()
    result = bot.update()
    
    print("=" * 70)
    print("TECHNICAL TRADING BOT - UPDATE")
    print("=" * 70)
    print(f"\nStatus: {result['status']}")
    print(f"Signal: {result['signal']} ({'NO TRADE' if result['signal'] == 0 else 'BUY' if result['signal'] == 1 else 'SELL'})")
    if result.get('strategy_deployment'):
        print(f"Deployment approved: {result['strategy_deployment'].get('approved', False)}")
    
    if result['signal'] > 0:
        print(f"Score: {result['score']}/8")
        print(f"\nLatest Candle:")
        for k, v in result['latest_candle'].items():
            print(f"  {k}: {v}")
        
        print(f"\nSignal Details:")
        for k, v in result['details'].items():
            print(f"  {k}: {v}")
    
    if result.get('signals_ready'):
        print(f"\nSignals ready for reviewed DEMO execution: {len(result['signals_ready'])}")
        for trade in result['signals_ready']:
            print(f"  - {trade['time']}: {trade['side']} @ {trade['entry_price']:.5f} (Score: {trade['score']}/8)")
    
    print("\n" + "=" * 70)
    
    return result


if __name__ == "__main__":
    run()
