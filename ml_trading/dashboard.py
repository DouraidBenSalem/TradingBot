"""Dashboard unique de suivi de la stratégie technique EUR/USD."""

import json
import os
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, url_for

from app.data import dataquality
from ml_trading.backtest_technical import (
    latest_result,
    list_runs,
    run as run_backtest,
    run_multiple,
    get_preset_periods,
    get_warehouse_analysis_snapshot,
)
from ml_trading.features_provider import get_latest_feature_timestamp
from ml_trading.pipeline import ROOT, load_state, save_state
from ml_trading.technical_trading_bot import TechnicalTradingBot
from ml_trading import mt5_demo

from Database.postgresql import (
    get_connection,
    ensure_trade_history_table,
    insert_trades_batch,
    list_trade_history,
    count_trade_history,
    get_trade_by_id,
    delete_trade_by_id,
    ensure_daily_sessions_table,
    get_daily_session_by_date,
    list_daily_sessions,
    delete_daily_sessions,
    close_daily_session,
    ensure_monitoring_tables,
    list_mt5_trades,
    delete_mt5_trades,
    list_bot_decisions,
    list_bot_events,
    record_bot_event,
    list_metric_snapshots,
    list_mt5_connection_events,
)


def _session_row_to_dict(row) -> dict:
    """Convertir une RealDictRow de session en dict JSON-sérialisable."""
    if not row:
        return {}
    out: dict = {}
    for key, val in dict(row).items():
        if val is None:
            out[key] = None
        elif isinstance(val, datetime):
            out[key] = val.isoformat(timespec="seconds")
        elif hasattr(val, "isoformat"):
            out[key] = val.isoformat()
        else:
            out[key] = val
    out["id"] = int(row["id"]) if row.get("id") is not None else None
    return out


def _monitoring_row_to_dict(row) -> dict:
    """Return a JSON-safe representation of a PostgreSQL monitoring row."""
    if not row:
        return {}
    result = {}
    for key, value in dict(row).items():
        if isinstance(value, datetime):
            result[key] = value.isoformat(timespec="seconds")
        elif hasattr(value, "isoformat"):
            result[key] = value.isoformat()
        else:
            result[key] = value
    return result


def _parse_iso(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except (TypeError, ValueError):
        return None


def _monitoring_payload(demo=None, state=None, trade_filters=None):
    """Build the shared, persisted monitoring model for both dashboards."""
    state = state or load_state()
    signal = state.get("signal", {})
    runtime_state = {}
    runtime_path = ROOT / "runtime" / "trading_state.json"
    try:
        runtime_state = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {}
    except (OSError, ValueError):
        runtime_state = {}
    if demo is None:
        demo = signal.get("mt5_demo") or runtime_state.get("mt5_demo") or state.get("mt5_demo") or {}
    demo = dict(demo or {})
    demo.setdefault(
        "connection_status",
        "CONNECTED" if demo.get("connected") else
        "CONNECTION_ERROR" if demo.get("connection_error") else "DISCONNECTED",
    )
    demo.setdefault("terminal_open", bool(demo.get("connected")))
    demo.setdefault("account_available", bool(demo.get("account_login")))

    history_error = None
    trade_filters = trade_filters or {}
    try:
        ensure_monitoring_tables()
        all_trade_rows = list_mt5_trades(limit=5000)
        filtered_trade_rows = list_mt5_trades(limit=1000, **trade_filters)
        decision_rows = list_bot_decisions(limit=300)
        event_rows = list_bot_events(limit=400)
        metric_rows = list_metric_snapshots(limit=1440)
        connection_rows = list_mt5_connection_events(limit=100)
    except Exception as error:
        history_error = str(error)
        all_trade_rows = filtered_trade_rows = []
        decision_rows = event_rows = metric_rows = connection_rows = []

    all_trades = [_monitoring_row_to_dict(row) for row in all_trade_rows]
    trades = [_monitoring_row_to_dict(row) for row in filtered_trade_rows]
    decisions = [_monitoring_row_to_dict(row) for row in decision_rows]
    events = [_monitoring_row_to_dict(row) for row in event_rows]
    metric_snapshots = [_monitoring_row_to_dict(row) for row in reversed(metric_rows)]
    connection_events = [_monitoring_row_to_dict(row) for row in connection_rows]

    closed = sorted(
        [trade for trade in all_trades if trade.get("status") == "CLOSED" and trade.get("profit_loss") is not None],
        key=lambda item: item.get("closed_at") or item.get("opened_at") or "",
    )
    open_count = len([trade for trade in all_trades if trade.get("status") == "OPEN"])
    live_positions = demo.get("positions") or []
    open_count = max(open_count, len(live_positions))
    wins = [float(trade["profit_loss"]) for trade in closed if float(trade["profit_loss"]) > 0]
    losses = [float(trade["profit_loss"]) for trade in closed if float(trade["profit_loss"]) < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    net_profit = sum(float(trade["profit_loss"]) for trade in closed)
    starting_equity = float(demo.get("balance") or 0) - net_profit
    cumulative = 0.0
    peak = starting_equity
    max_drawdown = 0.0
    performance_curve = []
    for trade in closed:
        cumulative += float(trade["profit_loss"])
        equity = starting_equity + cumulative
        peak = max(peak, equity)
        drawdown = equity - peak
        max_drawdown = min(max_drawdown, drawdown)
        performance_curve.append({
            "time": trade.get("closed_at") or trade.get("opened_at"),
            "equity": round(equity, 2),
            "cumulative_profit": round(cumulative, 2),
            "drawdown": round(drawdown, 2),
        })

    if metric_snapshots:
        equity_curve = [
            {
                "time": row.get("captured_at"),
                "equity": row.get("equity"),
                "balance": row.get("balance"),
                "free_margin": row.get("free_margin"),
            }
            for row in metric_snapshots
        ]
        observed_equities = [float(row["equity"]) for row in metric_snapshots if row.get("equity") is not None]
        if observed_equities:
            observed_peak = observed_equities[0]
            observed_drawdowns = []
            for observed_equity in observed_equities:
                observed_peak = max(observed_peak, observed_equity)
                observed_drawdowns.append(observed_equity - observed_peak)
            max_drawdown = min(max_drawdown, min(observed_drawdowns))
    else:
        equity_curve = [
            {"time": point["time"], "equity": point["equity"], "balance": point["equity"]}
            for point in performance_curve
        ]

    activity_buckets = {}
    for item in decisions:
        timestamp = item.get("occurred_at") or ""
        bucket = timestamp[:13] + ":00" if len(timestamp) >= 13 else timestamp
        values = activity_buckets.setdefault(bucket, {"time": bucket, "decisions": 0, "trades": 0, "events": 0})
        values["decisions"] += 1
        if item.get("action") in ("BUY", "SELL"):
            values["trades"] += 1
    for item in events:
        timestamp = item.get("occurred_at") or ""
        bucket = timestamp[:13] + ":00" if len(timestamp) >= 13 else timestamp
        activity_buckets.setdefault(bucket, {"time": bucket, "decisions": 0, "trades": 0, "events": 0})["events"] += 1
    activity = sorted(activity_buckets.values(), key=lambda item: item["time"])[-48:]

    last_update_value = runtime_state.get("last_update") or state.get("last_update")
    last_update = _parse_iso(last_update_value)
    age_seconds = (datetime.now() - last_update).total_seconds() if last_update else None
    runtime_status = runtime_state.get("status") or state.get("status", "")
    bot_status = "RUNNING" if age_seconds is not None and age_seconds <= 180 and not str(runtime_status).startswith("ERROR") else "IDLE"
    if str(runtime_status).startswith("ERROR"):
        bot_status = "ERROR"
    recorded_start = next((row.get("occurred_at") for row in events if row.get("event_type") == "BOT_STARTED"), None)
    started_at = _parse_iso(runtime_state.get("bot_started_at") or state.get("bot_started_at") or recorded_start)
    uptime_seconds = max(0, int((datetime.now() - started_at).total_seconds())) if started_at else None
    if metric_snapshots and observed_equities:
        current_drawdown = min(0.0, observed_equities[-1] - max(observed_equities))
    elif performance_curve:
        current_drawdown = performance_curve[-1]["drawdown"]
    else:
        current_drawdown = min(0.0, float(demo.get("equity") or 0) - float(demo.get("balance") or 0))

    last_connected = next((row.get("occurred_at") for row in connection_events if row.get("status") == "CONNECTED"), None)
    if not last_connected and demo.get("connected"):
        last_connected = demo.get("last_connection_at") or demo.get("checked_at") or last_update_value
    metrics = {
        "total_trades": len(all_trades),
        "open_trades": open_count,
        "closed_trades": len(closed),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(100 * len(wins) / len(closed), 2) if closed else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else (None if gross_profit else 0.0),
        "total_profit_loss": round(net_profit, 2),
        "expectancy": round(net_profit / len(closed), 2) if closed else 0.0,
        "current_drawdown": round(current_drawdown, 2),
        "max_drawdown": round(max_drawdown, 2),
    }
    return {
        "demo": demo,
        "bot": {
            "status": bot_status,
            "uptime_seconds": uptime_seconds,
            "last_activity": runtime_state.get("last_bot_activity") or last_update_value,
            "last_update": last_update_value,
            "last_connection": last_connected,
        },
        "signal": signal,
        "metrics": metrics,
        "trades": trades,
        "decisions": decisions,
        "events": events,
        "connection_events": connection_events,
        "charts": {
            "equity": equity_curve,
            "performance": performance_curve,
            "activity": activity,
            "win_loss": {"wins": len(wins), "losses": len(losses)},
        },
        "history_error": history_error,
    }

app = Flask(__name__, template_folder=str(ROOT / "templates"))
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.jinja_env.auto_reload = True
STATE_PATH = ROOT / "runtime" / "pipeline_state.json"

_FRENCH_DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
_TIME_RE = re.compile(r"^\d{2}:\d{2}$")


def _parse_french_datetime(date_str, time_str, field_label):
    if not date_str or not time_str:
        raise ValueError(f"{field_label} : date et heure sont requises.")
    if not _FRENCH_DATE_RE.match(date_str.strip()):
        raise ValueError(f"{field_label} : format de date invalide '{date_str}'. Format attendu : JJ/MM/AAAA.")
    if not _TIME_RE.match(time_str.strip()):
        raise ValueError(f"{field_label} : format d'heure invalide '{time_str}'. Format attendu : HH:MM.")
    try:
        day, month, year = (int(x) for x in date_str.strip().split("/"))
    except Exception as exc:
        raise ValueError(f"{field_label} : date '{date_str}' non lisible. Format attendu : JJ/MM/AAAA.") from exc
    try:
        hour, minute = (int(x) for x in time_str.strip().split(":"))
    except Exception as exc:
        raise ValueError(f"{field_label} : heure '{time_str}' non lisible. Format attendu : HH:MM.") from exc
    if not (1 <= month <= 12):
        raise ValueError(f"{field_label} : mois invalide ({month}).")
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"{field_label} : heure invalide ({hour}:{minute}).")
    try:
        dt = datetime(year, month, day, hour, minute, 0)
    except ValueError as exc:
        raise ValueError(f"{field_label} : combinaison date/heure invalide '{date_str} {time_str}' ({exc}).") from exc
    return dt


def _get_absolute_features_range():
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT MIN(time), MAX(time) FROM public.eurusd_features")
            row = cursor.fetchone()
        return (row[0], row[1]) if row else (None, None)
    finally:
        connection.close()


def _database_snapshot():
    try:
        quality = dataquality.collect_quality()
        return {**quality, "status": "SUCCESS", "quality_status": quality.get("status")}
    except Exception as error:
        return {"status": "ERROR", "error": str(error)}


def snapshot():
    state = load_state()
    public_state = {
        key: value for key, value in state.items()
        if key not in {"last_backtest", "multi_backtests", "form_errors", "form_values"}
    }
    return {
        "state": public_state,
        "database": _database_snapshot(),
        "strategy": state.get("signal", {}),
        "actions": [],
    }


@app.get("/")
def index():
    item = snapshot()
    return render_template("dashboard.html", **item)


@app.get("/api/status")
def api_status():
    return jsonify(snapshot())


@app.get("/mt5-demo")
def mt5_demo_page():
    return render_template("mt5_demo.html")


def _strategy_risk_snapshot():
    """Expose the exact live strategy and risk rules without duplicating them."""
    from ml_trading.scalping_deployment import load_demo_strategy, load_deployed_strategy

    demo_mode = os.getenv("MT5_DEMO_MODE", "false").strip().lower() == "true"
    strategy, deployment = load_demo_strategy() if demo_mode else load_deployed_strategy()
    parameters = strategy.parameters() if hasattr(strategy, "parameters") else {}
    simulation = deployment.get("simulation") or {}
    sl_atr = float(simulation.get("sl_atr_multiplier", 0) or 0)
    tp_atr = float(simulation.get("tp_atr_multiplier", 0) or 0)
    reward_risk = round(tp_atr / sl_atr, 2) if sl_atr else None
    strategy_type = deployment.get("strategy_type") or "validation_guard"
    profiles = {
        "ema_pullback": {
            "advantages": [
                "Suit la tendance dominante avec l’alignement EMA 9 / 21 / 50.",
                "Attend un pullback et une bougie de rejet avant l’entrée.",
                "Adapte le stop et l’objectif à la volatilité via l’ATR.",
                "Filtre le spread, le volume, le RSI et le MACD sur bougie clôturée.",
            ],
            "limitations": [
                "Les moyennes mobiles réagissent avec retard lors des retournements rapides.",
                "Les filtres stricts réduisent le nombre d’opportunités.",
                "Le M1 reste sensible au bruit, au slippage et aux variations de spread.",
                "Un lot fixe ne maintient pas un pourcentage de risque constant lorsque l’ATR varie.",
            ],
        },
        "validation_guard": {
            "advantages": ["Bloque toute exécution réelle tant qu’aucune stratégie n’est validée."],
            "limitations": ["Aucun signal exploitable n’est émis dans ce mode de sécurité."],
        },
    }
    profile = profiles.get(strategy_type, {
        "advantages": ["Paramètres issus du candidat sélectionné par la validation chronologique."],
        "limitations": ["Les performances historiques ne garantissent pas les résultats futurs."],
    })
    return {
        "mode": deployment.get("mode") or ("DEMO" if demo_mode else "VALIDATED"),
        "approved": bool(deployment.get("approved")),
        "name": parameters.get("name") or deployment.get("demo_candidate") or deployment.get("selected_candidate") or strategy_type,
        "candidate": deployment.get("demo_candidate") or deployment.get("selected_candidate"),
        "strategy_type": strategy_type,
        "parameters": parameters,
        "advantages": profile["advantages"],
        "limitations": profile["limitations"],
        "risk": {
            "max_trades_per_day": min(3, max(1, int(os.getenv("MT5_MAX_TRADES_PER_DAY", "3")))),
            "fixed_lot": float(deployment.get("fixed_lot") or os.getenv("MT5_DEMO_LOT", "0.07")),
            "quality_score": int(parameters.get("min_score", 0) or 0),
            "quality_score_max": 8,
            "stop_loss_atr": sl_atr or None,
            "take_profit_atr": tp_atr or None,
            "reward_risk": reward_risk,
            "horizon_bars": simulation.get("horizon_bars"),
            # No monetary cap or daily target is enforced by the live bot yet.
            # Keep these explicit instead of presenting display-only settings
            # as protections that execution would not actually apply.
            "max_risk_per_trade": None,
            "max_daily_loss": None,
            "daily_target": None,
            "stop_on_positive_daily_net": True,
            "third_trade_recovery": True,
            "one_open_position": True,
            "duplicate_candle_blocked": True,
            "demo_account_only": True,
            "execution_enabled": os.getenv("MT5_DEMO_EXECUTION_ENABLED", "false").strip().lower() == "true",
        },
        "reason": deployment.get("reason"),
    }


@app.get("/strategy-risk")
def strategy_risk_page():
    return render_template("strategy_risk.html", strategy=_strategy_risk_snapshot())


@app.get("/scalping-validation")
def scalping_validation_page():
    path = ROOT / "reports" / "scalping_validation_report.json"
    return render_template("scalping_validation.html", report_name=path.name)


@app.get("/api/scalping-validation")
def api_scalping_validation():
    path = ROOT / "reports" / "scalping_validation_report.json"
    if not path.exists():
        return jsonify({"ok": False, "error": "No validation report is available yet."}), 404
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return jsonify({"ok": False, "error": f"Validation report is unreadable: {error}"}), 500
    return jsonify({"ok": True, "report": report, "report_name": path.name})


@app.get("/api/mt5-demo")
def api_mt5_demo():
    """Live DEMO snapshot plus PostgreSQL history for the demo strategy only."""
    state = load_state()
    signal = state.get("signal", {})
    demo_enabled = os.getenv("MT5_DEMO_MODE", "false").strip().lower() == "true"
    demo = signal.get("mt5_demo") or {}
    if not demo_enabled:
        demo = {
            "mode": "DEMO", "connection_status": "DISCONNECTED",
            "terminal_open": False, "connected": False,
            "account_available": False, "account_is_demo": False,
            "execution_enabled": False,
            "connection_error": "MT5_DEMO_MODE is disabled. Add MT5_DEMO_MODE=true to .env and restart the application.",
        }
    elif not demo:
        # Use a read-only probe only when the bot has not published a snapshot
        # yet.  Normal status transitions are sampled by the bot every cycle;
        # avoiding a second synchronous MT5 connection keeps HTTP responsive.
        demo = mt5_demo.market_snapshot()
    filters = {
        key: request.args.get(key) or None
        for key in ("symbol", "start_date", "end_date", "side", "performance", "status")
    }
    monitoring = _monitoring_payload(demo=demo, state=state, trade_filters=filters)
    daily: dict = {}
    daily_sessions: list[dict] = []
    if demo_enabled:
        try:
            ensure_daily_sessions_table()
            account_login = int(demo.get("account_login") or 0)
            today_row = get_daily_session_by_date(
                account_login=account_login,
            )
            daily = _session_row_to_dict(today_row)
            daily_sessions = [
                _session_row_to_dict(r) for r in list_daily_sessions(limit=365)
            ]
        except Exception as error:
            daily = {"_error": str(error)}
            daily_sessions = []
    monitoring.update({"daily": daily, "daily_sessions": daily_sessions})
    return jsonify(monitoring)


def _delete_ids_from_request():
    payload = request.get_json(silent=True) or {}
    values = payload.get("ids")
    if not isinstance(values, list) or not values:
        raise ValueError("Le champ 'ids' doit contenir au moins un identifiant.")
    if len(values) > 500:
        raise ValueError("Une suppression est limitée à 500 lignes.")
    try:
        ids = sorted({int(value) for value in values if int(value) > 0})
    except (TypeError, ValueError) as error:
        raise ValueError("Tous les identifiants doivent être des entiers positifs.") from error
    if not ids:
        raise ValueError("Aucun identifiant valide n’a été fourni.")
    return ids


@app.delete("/api/mt5-demo/trades")
def api_mt5_demo_trades_delete():
    try:
        requested_ids = _delete_ids_from_request()
        deleted_ids = delete_mt5_trades(requested_ids)
        if deleted_ids:
            record_bot_event(
                "HISTORY_DELETED",
                f"{len(deleted_ids)} ligne(s) supprimée(s) du Trade History.",
                component="DASHBOARD",
                details={"resource": "mt5_trade_history", "ids": deleted_ids},
            )
        skipped_ids = sorted(set(requested_ids) - set(deleted_ids))
        return jsonify({
            "ok": True,
            "requested": len(requested_ids),
            "deleted": len(deleted_ids),
            "deleted_ids": deleted_ids,
            "skipped_ids": skipped_ids,
            "warning": "Les trades ouverts ne peuvent pas être supprimés." if skipped_ids else None,
        })
    except ValueError as error:
        return jsonify({"ok": False, "error": str(error)}), 400
    except Exception as error:
        return jsonify({"ok": False, "error": str(error)}), 500


@app.delete("/api/mt5-demo/daily-sessions")
def api_mt5_demo_daily_sessions_delete():
    try:
        requested_ids = _delete_ids_from_request()
        deleted_ids = delete_daily_sessions(requested_ids)
        if deleted_ids:
            record_bot_event(
                "DAILY_SUMMARY_DELETED",
                f"{len(deleted_ids)} résumé(s) de journée supprimé(s).",
                component="DASHBOARD",
                details={"resource": "mt5_demo_daily_sessions", "ids": deleted_ids},
            )
        skipped_ids = sorted(set(requested_ids) - set(deleted_ids))
        return jsonify({
            "ok": True,
            "requested": len(requested_ids),
            "deleted": len(deleted_ids),
            "deleted_ids": deleted_ids,
            "skipped_ids": skipped_ids,
            "warning": "La session RUNNING ne peut pas être supprimée." if skipped_ids else None,
        })
    except ValueError as error:
        return jsonify({"ok": False, "error": str(error)}), 400
    except Exception as error:
        return jsonify({"ok": False, "error": str(error)}), 500


@app.get("/api/monitoring")
def api_monitoring():
    """Global monitoring feed used by the application dashboard."""
    state = load_state()
    signal = state.get("signal", {})
    demo = signal.get("mt5_demo") or {}
    if os.getenv("MT5_DEMO_MODE", "false").strip().lower() == "true" and not demo:
        demo = mt5_demo.market_snapshot()
    return jsonify(_monitoring_payload(demo=demo, state=state))


@app.get("/api/data-quality")
def api_data_quality():
    return jsonify(_database_snapshot())


@app.get("/data-quality")
def data_quality():
    return render_template("report.html", title="Data Quality EUR/USD M1")


@app.get("/reports/data-quality")
def data_quality_report():
    path = ROOT / "data_quality_report.html"
    if not path.exists():
        dataquality.generate_report(str(path))
    return path.read_text(encoding="utf-8")


@app.get("/api/features-range")
def api_features_range():
    min_ts, max_ts = _get_absolute_features_range()
    return jsonify({
        "min": min_ts.isoformat() if min_ts else None,
        "max": max_ts.isoformat() if max_ts else None,
    })


@app.get("/backtest")
def backtest_page():
    state = load_state()
    last = state.get("last_backtest") or latest_result()
    multi = state.get("multi_backtests", [])
    presets = get_preset_periods()
    warehouse = None
    try:
        warehouse = get_warehouse_analysis_snapshot()
    except Exception:
        warehouse = None
    return render_template(
        "backtest.html",
        result=last,
        multi_backtests=multi,
        preset_periods=presets,
        warehouse_snapshot=warehouse,
    )


@app.get("/api/preset-periods")
def api_preset_periods():
    try:
        return jsonify({"periods": get_preset_periods()})
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.get("/api/warehouse-analysis")
def api_warehouse_analysis():
    try:
        return jsonify(get_warehouse_analysis_snapshot())
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.post("/api/run-backtest-json")
def api_run_backtest_json():
    """Endpoint AJAX : lance le backtest et retourne directement le résultat JSON.

    Body form:
      - date_debut_0, heure_debut_0, date_fin_0, heure_fin_0
      - (optionnel) date_debut_{i}... pour multi-périodes
    """
    try:
        periods, errors = _parse_periods(request.form)
        if errors:
            return jsonify({"ok": False, "error": " ; ".join(errors)}), 400
        if not periods:
            return jsonify({"ok": False, "error": "Aucune période valide spécifiée"}), 400

        results = run_multiple(periods)
        state = load_state()
        state["multi_backtests"] = results
        state["last_update"] = datetime.now().isoformat(timespec="seconds")
        save_state(state)
        return jsonify({"ok": True, "results": results})
    except ValueError as error:
        return jsonify({"ok": False, "error": str(error)}), 400
    except Exception as error:
        return jsonify({"ok": False, "error": f"Erreur serveur : {error}"}), 500


@app.get("/api/backtest")
def api_backtest():
    return jsonify(load_state().get("last_backtest") or latest_result() or {})


@app.post("/actions/run-backtest")
def action_backtest():
    if request.form.get("confirmed") == "1":
        state = load_state()
        try:
            date_debut = request.form.get("date_debut", "").strip()
            heure_debut = request.form.get("heure_debut", "").strip()
            date_fin = request.form.get("date_fin", "").strip()
            heure_fin = request.form.get("heure_fin", "").strip()
            start_time = _parse_french_datetime(date_debut, heure_debut, "Début de période") if date_debut or heure_debut else None
            end_time = _parse_french_datetime(date_fin, heure_fin, "Fin de période") if date_fin or heure_fin else None
            state["last_backtest"] = run_backtest(start_time=start_time, end_time=end_time)
            state["last_update"] = datetime.now().isoformat(timespec="seconds")
            save_state(state)
        except ValueError as error:
            state.setdefault("form_errors", {})
            state["form_errors"]["run_backtest"] = str(error)
            save_state(state)
    return redirect(url_for("backtest_page"))


def _parse_periods(form):
    """Extraire la liste des périodes (start_time, end_time) depuis le formulaire.

    Chaque période est définie par 4 champs :
      - date_debut_{i} (JJ/MM/AAAA) + heure_debut_{i} (HH:MM)
      - date_fin_{i}   (JJ/MM/AAAA) + heure_fin_{i}   (HH:MM)
    """
    periods = []
    errors = []
    index = 0
    while True:
        date_d = form.get(f"date_debut_{index}", "").strip()
        heur_d = form.get(f"heure_debut_{index}", "").strip()
        date_f = form.get(f"date_fin_{index}", "").strip()
        heur_f = form.get(f"heure_fin_{index}", "").strip()
        has_any = bool(date_d or heur_d or date_f or heur_f)
        if not has_any:
            break
        period_label = f"Période #{index + 1}"
        try:
            start_time = _parse_french_datetime(date_d, heur_d, f"{period_label} / Début")
            end_time = _parse_french_datetime(date_f, heur_f, f"{period_label} / Fin")
        except ValueError as error:
            errors.append(str(error))
            index += 1
            continue
        if end_time <= start_time:
            errors.append(
                f"{period_label} : la fin ({date_f} {heur_f}) doit être strictement postérieure au début ({date_d} {heur_d})."
            )
            index += 1
            continue
        min_ts, max_ts = _get_absolute_features_range()
        if min_ts and end_time < min_ts:
            errors.append(
                f"{period_label} : la période se termine avant la première donnée disponible ({min_ts.strftime('%d/%m/%Y %H:%M')})."
            )
            index += 1
            continue
        if max_ts and start_time > max_ts:
            errors.append(
                f"{period_label} : la période débute après la dernière donnée disponible ({max_ts.strftime('%d/%m/%Y %H:%M')})."
            )
            index += 1
            continue
        periods.append((start_time, end_time))
        index += 1
    return periods, errors


@app.post("/actions/run-backtest-multiple")
def action_backtest_multiple():
    if request.form.get("confirmed") == "1":
        state = load_state()
        periods, errors = _parse_periods(request.form)
        if errors:
            state.setdefault("form_errors", {})
            state["form_errors"]["run_backtest_multiple"] = errors
            save_state(state)
            return redirect(url_for("backtest_page"))
        state.pop("form_errors", None)
        if periods:
            results = run_multiple(periods)
            state["multi_backtests"] = results
            state["last_update"] = datetime.now().isoformat(timespec="seconds")
            save_state(state)
    return redirect(url_for("backtest_page"))


@app.get("/api/backtest-runs")
def api_backtest_runs():
    return jsonify(list_runs())


@app.get("/api/multi-backtest")
def api_multi_backtest():
    return jsonify(load_state().get("multi_backtests", []))


@app.post("/api/run-backtest-multiple")
def api_run_backtest_multiple():
    periods, errors = _parse_periods(request.form)
    if errors:
        return jsonify({"results": [], "error": " ; ".join(errors)}), 400
    if not periods:
        return jsonify({"results": [], "error": "Aucune période valide spécifiée"}), 400
    try:
        results = run_multiple(periods)
        state = load_state()
        state["multi_backtests"] = results
        state["last_update"] = datetime.now().isoformat(timespec="seconds")
        save_state(state)
        return jsonify({"results": results})
    except Exception as error:
        return jsonify({"results": [], "error": str(error)}), 500


@app.get("/audit")
def audit():
    return render_template("audit.html", audit=load_state().get("audit", [])[::-1])


@app.get("/actions")
def actions():
    return redirect(url_for("index"))


# =====================================================================
# Trade History Persistance (section-trades + justifications structurées)
# =====================================================================

def _row_to_trade_dict(row) -> dict:
    """Convertir un RealDictRow trade_history en dict front-end compatible."""
    if not row:
        return {}
    raw = row.get("raw_trade") or {}
    t = dict(raw) if isinstance(raw, dict) else {}

    # Les colonnes relationnelles font autorite sur le JSON brut. Tester
    # explicitement ``is not None`` est important : 0.0 est une valeur
    # parfaitement valide pour le P&L, le slippage, les prix ou les compteurs
    # et ne doit jamais etre remplace par une ancienne valeur du JSON.
    database_fields = {
        "backtest_run_id": "backtest_run_id",
        "trade_sequence": "trade_sequence",
        "source": "source",
        "decision_type": "decision_type",
        "entry_hour": "entry_hour",
        "exit_hour": "exit_hour",
        "duration_seconds": "duration_seconds",
        "duration_minutes": "duration_minutes",
        "duration_text": "duration_text",
        "side": "side",
        "entry_price": "entry",
        "exit_price": "exit",
        "sl_price": "sl",
        "tp_price": "tp",
        "sl_dollar": "sl_dollar",
        "tp_dollar": "tp_dollar",
        "volume": "volume",
        "lot": "lot",
        "ema_9": "ema_9",
        "ema_21": "ema_21",
        "ema_50": "ema_50",
        "rsi": "rsi",
        "macd": "macd",
        "macd_hist": "macd_hist",
        "atr": "atr",
        "score": "score",
        "score_breakdown": "score_breakdown",
        "signal_details": "signal_details",
        "price_pnl": "price_pnl",
        "commission": "commission",
        "slippage": "slippage",
        "net_pnl": "net_pnl",
        "balance_before": "balance_before",
        "balance_after": "balance",
        "outcome": "outcome",
        "exit_reason_code": "exit_reason_code",
        "exit_reason": "exit_reason",
        "justification_text": "why_trade",
        "justification_struct": "justification_struct",
    }
    for database_column, trade_field in database_fields.items():
        if row.get(database_column) is not None:
            t[trade_field] = row[database_column]

    t["id"] = int(row["id"])
    t["created_at"] = str(row["created_at"]) if row.get("created_at") else None
    t["backtest_run_id"] = row.get("backtest_run_id") if row.get("backtest_run_id") is not None else t.get("backtest_run_id")
    t["trade_sequence"] = row.get("trade_sequence") if row.get("trade_sequence") is not None else t.get("trade_sequence")
    t["source"] = row.get("source") if row.get("source") is not None else t.get("source")
    t["decision_type"] = row.get("decision_type") if row.get("decision_type") is not None else t.get("decision_type")
    t["entry_time"] = str(row["entry_time"]) if row.get("entry_time") is not None else t.get("entry_time") or t.get("time")
    t["exit_time"] = str(row["exit_time"]) if row.get("exit_time") is not None else t.get("exit_time")
    t["entry_date"] = str(row["entry_date"]) if row.get("entry_date") is not None else t.get("entry_date")
    t["entry_hour"] = row.get("entry_hour") if row.get("entry_hour") is not None else t.get("entry_hour")
    t["exit_hour"] = row.get("exit_hour") if row.get("exit_hour") is not None else t.get("exit_hour")
    t["duration_seconds"] = row.get("duration_seconds") if row.get("duration_seconds") is not None else t.get("duration_seconds")
    t["duration_minutes"] = row.get("duration_minutes") if row.get("duration_minutes") is not None else t.get("duration_minutes")
    t["duration_text"] = row.get("duration_text") if row.get("duration_text") is not None else t.get("duration_text")
    t["side"] = row.get("side") if row.get("side") is not None else t.get("side")
    t["entry"] = row.get("entry_price") if row.get("entry_price") is not None else t.get("entry")
    t["exit"] = row.get("exit_price") if row.get("exit_price") is not None else t.get("exit")
    t["sl"] = row.get("sl_price") if row.get("sl_price") is not None else t.get("sl")
    t["tp"] = row.get("tp_price") if row.get("tp_price") is not None else t.get("tp")
    t["sl_dollar"] = row.get("sl_dollar") if row.get("sl_dollar") is not None else t.get("sl_dollar")
    t["tp_dollar"] = row.get("tp_dollar") if row.get("tp_dollar") is not None else t.get("tp_dollar")
    t["volume"] = row.get("volume") if row.get("volume") is not None else t.get("volume")
    t["lot"] = row.get("lot") if row.get("lot") is not None else t.get("lot")
    t["ema_9"] = row.get("ema_9") if row.get("ema_9") is not None else t.get("ema_9")
    t["ema_21"] = row.get("ema_21") if row.get("ema_21") is not None else t.get("ema_21")
    t["ema_50"] = row.get("ema_50") if row.get("ema_50") is not None else t.get("ema_50")
    t["rsi"] = row.get("rsi") if row.get("rsi") is not None else t.get("rsi")
    t["macd"] = row.get("macd") if row.get("macd") is not None else t.get("macd")
    t["macd_hist"] = row.get("macd_hist") if row.get("macd_hist") is not None else t.get("macd_hist")
    t["atr"] = row.get("atr") if row.get("atr") is not None else t.get("atr")
    t["score"] = row.get("score") if row.get("score") is not None else t.get("score")
    t["score_breakdown"] = row.get("score_breakdown") if row.get("score_breakdown") is not None else t.get("score_breakdown")
    t["signal_details"] = row.get("signal_details") if row.get("signal_details") is not None else t.get("signal_details")
    t["price_pnl"] = row.get("price_pnl") if row.get("price_pnl") is not None else t.get("price_pnl")
    t["commission"] = row.get("commission") if row.get("commission") is not None else t.get("commission")
    t["slippage"] = row.get("slippage") if row.get("slippage") is not None else t.get("slippage")
    t["net_pnl"] = row.get("net_pnl") if row.get("net_pnl") is not None else t.get("net_pnl")
    t["balance_before"] = row.get("balance_before") if row.get("balance_before") is not None else t.get("balance_before")
    t["balance"] = row.get("balance_after") if row.get("balance_after") is not None else t.get("balance")
    t["outcome"] = row.get("outcome") if row.get("outcome") is not None else t.get("outcome")
    t["exit_reason_code"] = row.get("exit_reason_code") if row.get("exit_reason_code") is not None else t.get("exit_reason_code")
    t["exit_reason"] = row.get("exit_reason") if row.get("exit_reason") is not None else t.get("exit_reason")
    t["why_trade"] = row.get("justification_text") if row.get("justification_text") is not None else t.get("why_trade")
    t["justification_struct"] = (
        row.get("justification_struct")
        if row.get("justification_struct") is not None
        else t.get("justification_struct")
    )
    return t


@app.get("/api/trade-history")
def api_trade_history_list():
    """Récupérer l'historique persistant des trades (justifications incluses).

    Query params optionnels : side, outcome, decision_type, limit (déf 500), offset.
    """
    try:
        ensure_trade_history_table()
        side = request.args.get("side")
        outcome = request.args.get("outcome")
        decision_type = request.args.get("decision_type")
        limit = max(1, min(int(request.args.get("limit", 500)), 5000))
        offset = max(0, int(request.args.get("offset", 0)))
        rows = list_trade_history(
            limit=limit,
            offset=offset,
            side_filter=side,
            outcome_filter=outcome,
            decision_type=decision_type,
        )
        total = count_trade_history(
            side_filter=side,
            outcome_filter=outcome,
            decision_type=decision_type,
        )
        trades = [_row_to_trade_dict(r) for r in rows]
        return jsonify({
            "ok": True,
            "total": total,
            "trades": trades,
        })
    except ValueError as error:
        return jsonify({"ok": False, "error": f"Pagination invalide : {error}"}), 400
    except Exception as error:
        return jsonify({"ok": False, "error": str(error)}), 500


@app.get("/api/trade-history/<int:trade_id>")
def api_trade_history_get(trade_id: int):
    """Récupérer un trade unique par ID (incluant justification structurée)."""
    try:
        ensure_trade_history_table()
        row = get_trade_by_id(trade_id)
        if not row:
            return jsonify({"ok": False, "error": f"Trade #{trade_id} introuvable"}), 404
        return jsonify({
            "ok": True,
            "trade": _row_to_trade_dict(row),
        })
    except Exception as error:
        return jsonify({"ok": False, "error": str(error)}), 500


@app.delete("/api/trade-history/<int:trade_id>")
def api_trade_history_delete(trade_id: int):
    """Supprimer un trade par ID (mise à jour front + DB)."""
    try:
        ensure_trade_history_table()
        deleted = delete_trade_by_id(trade_id)
        if not deleted:
            return jsonify({
                "ok": False,
                "error": f"Trade #{trade_id} introuvable ou déjà supprimé",
            }), 404
        return jsonify({
            "ok": True,
            "deleted_id": trade_id,
            "remaining": count_trade_history(),
        })
    except Exception as error:
        return jsonify({"ok": False, "error": str(error)}), 500


@app.post("/api/trade-history/save-batch")
def api_trade_history_save_batch():
    """Endpoint frontal : sauvegarder explicitement un lot de trades.

    Utilisé par le front pour pousser le contenu actuel de `section-trades`
    directement vers la base persistante.

    Body JSON : { "trades": [...], "source": "backtest|manual", "backtest_run_id": null }
    """
    try:
        payload = request.get_json(silent=True) or {}
        trades = payload.get("trades") or []
        source = payload.get("source") or "backtest"
        backtest_run_id = payload.get("backtest_run_id")
        if not isinstance(trades, list) or not trades:
            return jsonify({
                "ok": False,
                "error": "Aucun trade fourni (champ 'trades' attendu en JSON).",
            }), 400
        # Un trade deja identifie est deja persiste. Le client peut envoyer la
        # liste complete apres un rafraichissement : l'ignorer ici evite toute
        # duplication accidentelle de l'historique PostgreSQL.
        pending_trades = [trade for trade in trades if not trade.get("id")]
        if not pending_trades:
            return jsonify({
                "ok": True,
                "inserted_ids": [],
                "inserted_count": 0,
            })
        ensure_trade_history_table()
        ids = insert_trades_batch(pending_trades, source=source, backtest_run_id=backtest_run_id)
        return jsonify({
            "ok": True,
            "inserted_ids": ids,
            "inserted_count": len(ids),
        })
    except Exception as error:
        return jsonify({"ok": False, "error": str(error)}), 500


# =====================================================================
# MT5 DEMO — Suivi quotidien (sessions/journées)
# =====================================================================

def _mt5_demo_account_login() -> int:
    try:
        state = load_state()
        demo_snap = (state.get("signal") or {}).get("mt5_demo") or {}
        login = demo_snap.get("account_login")
        if login:
            return int(login)
    except Exception:
        pass
    try:
        snap = mt5_demo.market_snapshot() or {}
        if snap.get("account_login"):
            return int(snap["account_login"])
    except Exception:
        pass
    return 0


@app.get("/api/mt5-demo/daily-sessions")
def api_mt5_demo_daily_sessions_list():
    """Liste des sessions quotidiennes MT5 DEMO + session du jour."""
    demo_enabled = os.getenv("MT5_DEMO_MODE", "false").strip().lower() == "true"
    if not demo_enabled:
        return jsonify({
            "ok": True,
            "demo_enabled": False,
            "today": {},
            "sessions": [],
            "error": "MT5_DEMO_MODE is disabled in .env.",
        })
    try:
        ensure_daily_sessions_table()
        account_login = _mt5_demo_account_login()
        today_row = get_daily_session_by_date(account_login=account_login)
        history_rows = list_daily_sessions(limit=365)
        return jsonify({
            "ok": True,
            "demo_enabled": True,
            "today": _session_row_to_dict(today_row),
            "sessions": [_session_row_to_dict(r) for r in history_rows],
        })
    except Exception as error:
        return jsonify({"ok": False, "error": str(error)}), 500


@app.get("/api/mt5-demo/daily-sessions/today")
def api_mt5_demo_daily_sessions_today():
    """Uniquement la session MT5 DEMO du jour."""
    demo_enabled = os.getenv("MT5_DEMO_MODE", "false").strip().lower() == "true"
    if not demo_enabled:
        return jsonify({
            "ok": True,
            "demo_enabled": False,
            "session": {},
            "error": "MT5_DEMO_MODE is disabled in .env.",
        })
    try:
        ensure_daily_sessions_table()
        account_login = _mt5_demo_account_login()
        today_row = get_daily_session_by_date(account_login=account_login)
        return jsonify({
            "ok": True,
            "demo_enabled": True,
            "session": _session_row_to_dict(today_row),
        })
    except Exception as error:
        return jsonify({"ok": False, "error": str(error)}), 500


@app.post("/api/mt5-demo/daily-sessions/close")
def api_mt5_demo_daily_sessions_close():
    """Fermeture explicite de la session du jour.

    Body JSON optionnel :
      { "session_id": <id ou null pour today>, "overrides": { ... champs ... } }
    """
    demo_enabled = os.getenv("MT5_DEMO_MODE", "false").strip().lower() == "true"
    if not demo_enabled:
        return jsonify({"ok": False, "error": "MT5_DEMO_MODE is disabled in .env."}), 400
    try:
        ensure_daily_sessions_table()
        payload = request.get_json(silent=True) or {}
        session_id = payload.get("session_id")
        if not session_id:
            account_login = _mt5_demo_account_login()
            today_row = get_daily_session_by_date(account_login=account_login)
            if not today_row:
                return jsonify({
                    "ok": False,
                    "error": "Aucune session RUNNING détectée pour aujourd'hui (pas encore de cycle exécuté).",
                }), 404
            session_id = int(today_row["id"])
        overrides = payload.get("overrides") or None
        closed = close_daily_session(int(session_id), overrides=overrides)
        if not closed:
            return jsonify({"ok": False, "error": f"Session #{session_id} introuvable."}), 404
        account_login = _mt5_demo_account_login()
        updated = get_daily_session_by_date(account_login=account_login)
        return jsonify({
            "ok": True,
            "closed_id": int(session_id),
            "session": _session_row_to_dict(updated),
        })
    except Exception as error:
        return jsonify({"ok": False, "error": str(error)}), 500



def start_dashboard(host="127.0.0.1", port=5000):
    thread = threading.Thread(target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False), name="technical-dashboard", daemon=True)
    thread.start()
    return thread


def run(host="127.0.0.1", port=5000):
    app.run(host=host, port=port, debug=False, use_reloader=False)
