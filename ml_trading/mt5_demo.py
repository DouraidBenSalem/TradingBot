"""Fail-closed MT5 DEMO execution and observation for EUR/USD M1.

The module has no import-time connection side effect.  It sends an order only
when the connected account explicitly identifies as DEMO *and* the operator
sets ``MT5_DEMO_EXECUTION_ENABLED=true``.  It is intentionally unsuitable for
REAL trading.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path


DEMO_MAGIC = int(os.getenv("MT5_DEMO_MAGIC", "26091401"))
DEMO_SYMBOL = os.getenv("MT5_EURUSD_SYMBOL", "EURUSD.m")
DEMO_LOT = float(os.getenv("MT5_DEMO_LOT", "0.07"))
MT5_CONNECT_TIMEOUT_MS = int(os.getenv("MT5_CONNECT_TIMEOUT_MS", "5000"))


def _terminal_path() -> str | None:
    """Resolve the broker-specific MT5 terminal instead of auto-detection."""
    configured = os.getenv("MT5_PATH")
    if configured and Path(configured).is_file():
        return configured
    program_files = os.getenv("ProgramFiles")
    if not program_files:
        return None
    matches = sorted(Path(program_files).glob("*MetaTrader 5*/terminal64.exe"))
    return str(matches[0]) if len(matches) == 1 else None


def _enabled() -> bool:
    return os.getenv("MT5_DEMO_EXECUTION_ENABLED", "false").strip().lower() == "true"


def _mt5():
    try:
        import MetaTrader5 as mt5
    except ImportError as error:
        raise RuntimeError("MetaTrader5 Python package is not installed") from error
    return mt5


def _connect():
    mt5 = _mt5()
    terminal = mt5.terminal_info()
    if terminal is not None and bool(getattr(terminal, "connected", True)):
        return mt5
    login, password, server = (
        os.getenv("MT5_LOGIN"),
        os.getenv("MT5_PASSWORD"),
        os.getenv("MT5_SERVER"),
    )
    if not (login and password and server):
        raise RuntimeError("MT5_LOGIN, MT5_PASSWORD and MT5_SERVER are required for the DEMO monitor")
    terminal_path = _terminal_path()
    initialize_args = (terminal_path,) if terminal_path else ()
    if not mt5.initialize(
        *initialize_args,
        login=int(login),
        password=password,
        server=server,
        timeout=MT5_CONNECT_TIMEOUT_MS,
    ):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    return mt5


def _account(mt5):
    account = mt5.account_info()
    if account is None:
        raise RuntimeError(f"MT5 account information unavailable: {mt5.last_error()}")
    demo_mode = getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", None)
    if demo_mode is None or account.trade_mode != demo_mode:
        raise RuntimeError("MT5 account is not explicitly DEMO; execution is blocked")
    return account


def _symbol_and_tick(mt5):
    info = mt5.symbol_info(DEMO_SYMBOL)
    if info is None:
        raise RuntimeError(f"MT5 symbol {DEMO_SYMBOL!r} is unavailable")
    if not info.visible and not mt5.symbol_select(DEMO_SYMBOL, True):
        raise RuntimeError(f"MT5 symbol {DEMO_SYMBOL!r} cannot be selected")
    info = mt5.symbol_info(DEMO_SYMBOL)
    tick = mt5.symbol_info_tick(DEMO_SYMBOL)
    if tick is None or tick.ask <= 0 or tick.bid <= 0:
        raise RuntimeError(f"No executable bid/ask tick for {DEMO_SYMBOL}")
    return info, tick


def _normalize_volume(volume, info):
    step = float(info.volume_step)
    normalized = round(round(volume / step) * step, 8)
    if normalized < float(info.volume_min) or normalized > float(info.volume_max):
        raise RuntimeError(
            f"Demo lot {volume} violates broker limits [{info.volume_min}, {info.volume_max}]"
        )
    return normalized


def _normalize_price(value, info):
    return round(float(value), int(info.digits))


def _resolve_filling_mode(mt5, request_template):
    candidates = [
        ("FOK",    getattr(mt5, "ORDER_FILLING_FOK", 0)),
        ("IOC",    getattr(mt5, "ORDER_FILLING_IOC", 1)),
        ("RETURN", getattr(mt5, "ORDER_FILLING_RETURN", 2)),
    ]
    last_reason = None
    for name, value in candidates:
        trial = dict(request_template)
        trial["type_filling"] = value
        check = mt5.order_check(trial)
        if check is not None and getattr(check, "retcode", None) == 0:
            return value, name
        if check is not None:
            last_reason = getattr(check, "comment", None) or f"retcode {getattr(check, 'retcode', None)}"
        else:
            last_reason = str(mt5.last_error())
    return None, last_reason or "no filling mode accepted by broker"


def market_snapshot() -> dict:
    """Return connection, demo-account, quote and current position state."""
    checked_at = datetime.now().isoformat(timespec="seconds")
    output = {
        "mode": "DEMO",
        "strategy": "ema_pullback_wider_stop",
        "execution_enabled": _enabled(),
        "connection_status": "CONNECTING",
        "checked_at": checked_at,
        "terminal_open": False,
        "connected": False,
        "account_available": False,
        "account_is_demo": False,
        "symbol": DEMO_SYMBOL,
        "connection_error": None,
        "spread_points": None,
        "last_bid": None,
        "last_ask": None,
        "positions": [],
    }
    try:
        mt5 = _mt5()
        terminal_before = mt5.terminal_info()
        output["terminal_open"] = terminal_before is not None
        output["connection_status"] = "RECONNECTING" if terminal_before is not None else "CONNECTING"
        if terminal_before is None:
            try:
                from Database.postgresql import record_mt5_snapshot
                record_mt5_snapshot(dict(output))
            except Exception:
                pass
        mt5 = _connect()
        terminal = mt5.terminal_info()
        output["terminal_open"] = terminal is not None
        if terminal is None or not bool(getattr(terminal, "connected", True)):
            raise RuntimeError("MT5 terminal is open but disconnected from the broker")
        account = mt5.account_info()
        if account is None:
            raise RuntimeError(f"MT5 account information unavailable: {mt5.last_error()}")
        output["account_available"] = True
        demo_mode = getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", None)
        if demo_mode is None or account.trade_mode != demo_mode:
            raise RuntimeError("MT5 account is not explicitly DEMO; execution is blocked")
        info, tick = _symbol_and_tick(mt5)
        positions = mt5.positions_get(symbol=DEMO_SYMBOL) or ()
        output.update({
            "connection_status": "CONNECTED",
            "connected": True,
            "account_is_demo": True,
            "account_login": account.login,
            "account_server": account.server,
            "balance": float(account.balance),
            "equity": float(account.equity),
            "free_margin": float(getattr(account, "margin_free", 0.0)),
            "margin": float(getattr(account, "margin", 0.0)),
            "margin_level": float(getattr(account, "margin_level", 0.0)),
            "floating_profit": float(getattr(account, "profit", 0.0)),
            "currency": getattr(account, "currency", None),
            "leverage": int(getattr(account, "leverage", 0)),
            "last_connection_at": checked_at,
            "last_bid": float(tick.bid),
            "last_ask": float(tick.ask),
            "spread_points": round((float(tick.ask) - float(tick.bid)) / float(info.point), 1),
            "positions": [
                {
                    "ticket": int(position.ticket), "type": int(position.type),
                    "side": "BUY" if int(position.type) == 0 else "SELL",
                    "symbol": str(position.symbol),
                    "volume": float(position.volume), "price_open": float(position.price_open),
                    "price_current": float(position.price_current),
                    "sl": float(position.sl), "tp": float(position.tp), "profit": float(position.profit),
                    "time": datetime.fromtimestamp(position.time).isoformat(timespec="seconds"),
                    "comment": str(getattr(position, "comment", "")),
                }
                for position in positions if int(position.magic) == DEMO_MAGIC
            ],
        })
    except Exception as error:
        output["connection_error"] = str(error)
        output["connection_status"] = "CONNECTION_ERROR" if output["terminal_open"] else "DISCONNECTED"
    try:
        from Database.postgresql import record_mt5_snapshot
        record_mt5_snapshot(output)
    except Exception as error:
        output["persistence_error"] = str(error)
    return output


def execute_signal(signal: int, atr: float, max_spread_to_atr: float = 0.35) -> dict:
    """Submit one broker-checked market order on a DEMO account only."""
    if signal not in (1, 2):
        return {"submitted": False, "reason": "NO TRADE"}
    if not _enabled():
        return {"submitted": False, "reason": "MT5_DEMO_EXECUTION_ENABLED is not true"}

    mt5 = _connect()
    account = _account(mt5)
    if not account.trade_allowed or not account.trade_expert:
        return {"submitted": False, "reason": "MT5 account does not allow expert trading"}
    info, tick = _symbol_and_tick(mt5)
    existing = [p for p in (mt5.positions_get(symbol=DEMO_SYMBOL) or ()) if int(p.magic) == DEMO_MAGIC]
    if existing:
        return {"submitted": False, "reason": "duplicate blocked: a DEMO strategy position is already open"}
    if atr <= 0:
        return {"submitted": False, "reason": "invalid ATR"}
    live_spread_ratio = (float(tick.ask) - float(tick.bid)) / atr
    if live_spread_ratio > max_spread_to_atr:
        return {
            "submitted": False,
            "reason": f"live spread/ATR {live_spread_ratio:.2f} exceeds {max_spread_to_atr:.2f}",
        }

    side = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
    price = float(tick.ask) if signal == 1 else float(tick.bid)
    sl_distance, tp_distance = 1.4 * atr, 2.2 * atr
    sl = price - sl_distance if signal == 1 else price + sl_distance
    tp = price + tp_distance if signal == 1 else price - tp_distance
    minimum_stop = float(info.trade_stops_level) * float(info.point)
    if abs(price - sl) < minimum_stop or abs(price - tp) < minimum_stop:
        return {"submitted": False, "reason": "SL/TP violates broker stop-distance constraint"}
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": DEMO_SYMBOL,
        "volume": _normalize_volume(DEMO_LOT, info),
        "type": side,
        "price": _normalize_price(price, info),
        "sl": _normalize_price(sl, info),
        "tp": _normalize_price(tp, info),
        "deviation": int(os.getenv("MT5_DEMO_DEVIATION_POINTS", "10")),
        "magic": DEMO_MAGIC,
        "comment": "ema_pullback_wider_stop_demo",
        "type_time": mt5.ORDER_TIME_GTC,
    }
    filling_value, filling_name = _resolve_filling_mode(mt5, request)
    if filling_value is None:
        return {"submitted": False, "reason": f"MT5 no supported filling mode: {filling_name}"}
    request["type_filling"] = filling_value
    request["comment"] = f"ema_pullback_wider_stop_{filling_name.lower()}"
    result = mt5.order_send(request)
    success_code = getattr(mt5, "TRADE_RETCODE_DONE", 10009)
    if result is None or result.retcode != success_code:
        retcode = getattr(result, "retcode", None) if result is not None else None
        if retcode == 10027:
            reason = "MT5 AutoTrading is disabled (client). Enable the 'Auto Trading' button in MT5 Terminal toolbar."
        else:
            comment = getattr(result, "comment", "") if result is not None else str(mt5.last_error())
            reason = f"MT5 order_send rejected (retcode={retcode}): {comment or result or mt5.last_error()}"
        return {"submitted": False, "reason": reason, "filling_mode": filling_name}
    position_id = int(result.order)
    try:
        created_deals = mt5.history_deals_get(ticket=int(result.deal)) or ()
        if created_deals:
            position_id = int(created_deals[0].position_id or position_id)
    except Exception:
        pass
    return {
        "submitted": True, "ticket": int(result.order), "deal": int(result.deal),
        "position_id": position_id, "order_ticket": int(result.order),
        "side": "BUY" if signal == 1 else "SELL", "entry": float(result.price),
        "sl": request["sl"], "tp": request["tp"], "lot": request["volume"],
        "spread_points": round((float(tick.ask) - float(tick.bid)) / float(info.point), 1),
        "filling_mode": filling_name,
    }


def _strategy_position_deals(mt5, since: datetime, until: datetime):
    """Return every deal for positions opened by this strategy.

    MT5 may assign magic ``0`` to a manual, SL or TP closing deal. Filtering
    each deal by magic therefore loses the exit and leaves a closed position
    displayed as OPEN. Ownership is determined from the entry deal, then all
    deals sharing its ``position_id`` are retained.
    """
    all_deals = mt5.history_deals_get(since, until, group=f"*{DEMO_SYMBOL}*") or ()
    entry_in = getattr(mt5, "DEAL_ENTRY_IN", 0)
    strategy_positions = {
        int(getattr(deal, "position_id", 0) or 0)
        for deal in all_deals
        if int(getattr(deal, "magic", 0) or 0) == DEMO_MAGIC
        and int(getattr(deal, "entry", entry_in)) == entry_in
    }
    return [
        deal for deal in all_deals
        if int(getattr(deal, "position_id", 0) or 0) in strategy_positions
    ], strategy_positions


def closed_deals(since: datetime) -> list[dict]:
    """Read closed DEMO deals for PostgreSQL reconciliation; never mutates MT5."""
    snapshot = market_snapshot()
    if not snapshot["connected"]:
        return []
    mt5 = _connect()
    deals, _ = _strategy_position_deals(mt5, since, datetime.now())
    exit_types = {
        getattr(mt5, "DEAL_ENTRY_OUT", 1),
        getattr(mt5, "DEAL_ENTRY_OUT_BY", 3),
    }
    return [deal._asdict() for deal in deals if int(deal.entry) in exit_types]


def trade_history(since: datetime) -> list[dict]:
    """Return complete MT5 positions by pairing their entry and exit deals."""
    snapshot = market_snapshot()
    if not snapshot["connected"]:
        return []
    mt5 = _connect()
    now = datetime.now()
    strategy_deals, strategy_position_ids = _strategy_position_deals(mt5, since, now)
    deals = sorted(
        strategy_deals,
        key=lambda item: int(getattr(item, "time_msc", 0) or (item.time * 1000)),
    )
    orders = [
        order for order in (mt5.history_orders_get(since, now, group=f"*{DEMO_SYMBOL}*") or ())
        if int(getattr(order, "position_id", 0) or getattr(order, "position_by_id", 0) or 0)
        in strategy_position_ids
    ]
    entry_in = getattr(mt5, "DEAL_ENTRY_IN", 0)
    entry_out = getattr(mt5, "DEAL_ENTRY_OUT", 1)
    entry_out_by = getattr(mt5, "DEAL_ENTRY_OUT_BY", 3)
    grouped = {}
    for deal in deals:
        raw = deal._asdict()
        position_id = int(raw.get("position_id") or raw.get("order") or raw.get("ticket") or 0)
        if not position_id:
            continue
        item = grouped.setdefault(position_id, {"entries": [], "exits": [], "all": []})
        item["all"].append(raw)
        if int(raw.get("entry", entry_in)) == entry_in:
            item["entries"].append(raw)
        elif int(raw.get("entry", entry_out)) in (entry_out, entry_out_by):
            item["exits"].append(raw)

    order_by_position = {}
    for order in orders:
        raw = order._asdict()
        position_id = int(raw.get("position_id") or raw.get("position_by_id") or raw.get("ticket") or 0)
        order_by_position.setdefault(position_id, []).append(raw)

    records = []
    for position_id, parts in grouped.items():
        if not parts["entries"]:
            continue
        first = parts["entries"][0]
        last_exit = parts["exits"][-1] if parts["exits"] else None
        related_orders = order_by_position.get(position_id, [])
        first_order = related_orders[0] if related_orders else {}
        last_order = related_orders[-1] if related_orders else {}
        opened = datetime.fromtimestamp(first["time"])
        closed = datetime.fromtimestamp(last_exit["time"]) if last_exit else None
        side = "BUY" if int(first.get("type", 0)) == 0 else "SELL"
        commission = sum(float(item.get("commission", 0.0) or 0.0) for item in parts["all"])
        swap = sum(float(item.get("swap", 0.0) or 0.0) for item in parts["all"])
        profit = sum(float(item.get("profit", 0.0) or 0.0) for item in parts["all"]) + commission + swap
        records.append({
            "ticket": position_id,
            "position_id": position_id,
            "order_ticket": int(first.get("order") or first_order.get("ticket") or position_id),
            "opened_at": opened.isoformat(timespec="seconds"),
            "closed_at": closed.isoformat(timespec="seconds") if closed else None,
            "symbol": str(first.get("symbol") or DEMO_SYMBOL),
            "side": side,
            "entry_price": float(first.get("price", 0.0)),
            "exit_price": float(last_exit.get("price", 0.0)) if last_exit else None,
            "volume": float(first.get("volume", 0.0)),
            "stop_loss": float(first_order.get("sl", 0.0) or 0.0),
            "take_profit": float(first_order.get("tp", 0.0) or 0.0),
            "profit_loss": round(profit, 2) if last_exit else None,
            "commission": commission,
            "swap": swap,
            "duration_seconds": int((closed - opened).total_seconds()) if closed else None,
            "status": "CLOSED" if last_exit else "OPEN",
            "entry_reason": str(first.get("comment") or first_order.get("comment") or "Signal confirmed"),
            "exit_reason": str(last_exit.get("comment") or last_order.get("comment") or "Position closed") if last_exit else None,
            "raw_deals": parts["all"],
            "raw_orders": related_orders,
        })
    return records
