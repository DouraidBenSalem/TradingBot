"""Moteur de suivi quotidien MT5 DEMO (obs-only, aucun side-effect sur la stratégie).

Ce module expose :
* Un enum typé des motifs de rejet NO_TRADE.
* Une fonction pure de catégorisation d'un signal (BUY/SELL/NO_TRADE) vers un motif.
* Un générateur de message explicatif en français à partir des vrais compteurs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


NO_TRADE_REASON_LABELS: Dict[str, str] = {
    "NO_TREND": "Tendance EMA non établie",
    "PULLBACK_NOT_CONFIRMED": "Pullback non confirmé (pas de rejet propre)",
    "MOMENTUM_NOT_CONFIRMED": "Confirmation momentum insuffisante (RSI / MACD)",
    "ATR_TOO_LOW": "Volatilité ATR trop faible pour un setup M1",
    "SPREAD_TOO_HIGH": "Spread / ATR trop élevé (coût d'exécution)",
    "OUTSIDE_TRADING_SESSION": "Hors sessions liquides Londres / New York",
    "BROKER_CONSTRAINT": "Contrainte broker (SL/TP, filling mode, AutoTrading...)",
    "POSITION_ALREADY_OPEN": "Une position DEMO est déjà ouverte",
    "DAILY_NET_TARGET_REACHED": "Résultat net journalier positif atteint",
    "MAX_TRADES_PER_DAY": "Limite de 3 trades journaliers atteinte",
    "SIGNAL_CONFLICT": "Conflit interne non catégorisé",
}


REASONS_IN_ORDER = [
    "OUTSIDE_TRADING_SESSION",
    "ATR_TOO_LOW",
    "SPREAD_TOO_HIGH",
    "NO_TREND",
    "PULLBACK_NOT_CONFIRMED",
    "MOMENTUM_NOT_CONFIRMED",
    "POSITION_ALREADY_OPEN",
    "DAILY_NET_TARGET_REACHED",
    "MAX_TRADES_PER_DAY",
    "BROKER_CONSTRAINT",
    "SIGNAL_CONFLICT",
]


def categorize_signal_rejection(
    signal_value: int,
    score: int,
    signal_details: Dict[str, Any],
    execution_reason: Optional[str] = None,
    execution_submitted: Optional[bool] = None,
) -> Optional[str]:
    """Catégoriser pourquoi un cycle n'a pas débouché sur un trade exécuté.

    Retourne ``None`` s'il n'y a pas de rejet (signal BUY/SELL exécuté avec succès).
    Sinon retourne une clé de ``NO_TRADE_REASON_LABELS``.
    """
    signal_value = int(signal_value or 0)

    if signal_value in (1, 2) and execution_submitted is True:
        return None

    if execution_reason:
        reason_lc = (execution_reason or "").lower()
        if "daily_net_target_reached" in reason_lc:
            return "DAILY_NET_TARGET_REACHED"
        if "max_trades_per_day" in reason_lc or "daily trade limit reached" in reason_lc:
            return "MAX_TRADES_PER_DAY"
        if "position_already_open" in reason_lc:
            return "POSITION_ALREADY_OPEN"
        if "duplicate blocked" in reason_lc:
            return "POSITION_ALREADY_OPEN"
        if "sl/tp violates broker" in reason_lc or "stop-distance" in reason_lc:
            return "BROKER_CONSTRAINT"
        if "no supported filling mode" in reason_lc:
            return "BROKER_CONSTRAINT"
        if "autotrading is disabled" in reason_lc:
            return "BROKER_CONSTRAINT"
        if "order_send rejected" in reason_lc or "retcode" in reason_lc:
            return "BROKER_CONSTRAINT"
        if "mt5_demo_execution_enabled" in reason_lc:
            return "BROKER_CONSTRAINT"
        if "account does not allow expert" in reason_lc:
            return "BROKER_CONSTRAINT"
        if "invalid atr" in reason_lc:
            return "ATR_TOO_LOW"
        if "spread/atr" in reason_lc:
            return "SPREAD_TOO_HIGH"

    details = signal_details or {}

    session_val = str(details.get("session") or "")
    if "outside liquid" in session_val.lower():
        return "OUTSIDE_TRADING_SESSION"

    volatility = str(details.get("volatility") or "")
    if "atr too low" in volatility.lower():
        return "ATR_TOO_LOW"
    if "atr unusually high" in volatility.lower():
        return "ATR_TOO_LOW"

    cost = str(details.get("cost") or "")
    if "spread/atr too high" in cost.lower():
        return "SPREAD_TOO_HIGH"

    volume = str(details.get("volume") or "")
    if "relative volume too low" in volume.lower():
        return "MOMENTUM_NOT_CONFIRMED"

    trend = str(details.get("trend") or "")
    if trend:
        if "insufficient" in trend.lower() or "emas too close" in trend.lower():
            return "NO_TREND"

    pullback = str(details.get("pullback") or "")
    if pullback and "no clean rejected pullback" in pullback.lower():
        return "PULLBACK_NOT_CONFIRMED"

    candle = str(details.get("candle") or "")
    rsi = str(details.get("rsi") or "")
    macd = str(details.get("macd") or "")
    if signal_value == 0:
        if candle and "not directional enough" in candle.lower():
            return "MOMENTUM_NOT_CONFIRMED"
        if rsi and "outside confirmation" in rsi.lower():
            return "MOMENTUM_NOT_CONFIRMED"
        if macd and "does not confirm" in macd.lower():
            return "MOMENTUM_NOT_CONFIRMED"

    if signal_value in (1, 2) and score < 7:
        return "MOMENTUM_NOT_CONFIRMED"

    return "SIGNAL_CONFLICT"


def top_reason(rejected_breakdown: Dict[str, int]) -> Optional[str]:
    if not rejected_breakdown:
        return None
    ordered = sorted(
        rejected_breakdown.items(),
        key=lambda kv: (-int(kv[1]), REASONS_IN_ORDER.index(kv[0]) if kv[0] in REASONS_IN_ORDER else 999),
    )
    for key, count in ordered:
        if int(count) > 0:
            return key
    return None


def build_no_trade_message(
    total_signals: int,
    buy_signals: int,
    sell_signals: int,
    rejected_signals: int,
    rejected_breakdown: Dict[str, int],
    no_trade_reason: Optional[str] = None,
) -> str:
    """Générer une phrase française compréhensible qui explique l'absence de trade.

    Ne jamais inventer de motif : n'évoque que des compteurs > 0.
    """
    total = max(0, int(total_signals or 0))
    buy = max(0, int(buy_signals or 0))
    sell = max(0, int(sell_signals or 0))
    rejected = max(0, int(rejected_signals or 0))

    if total == 0:
        return "Aucun cycle d'évaluation effectué aujourd'hui."

    parts = [f"Aucun trade aujourd'hui. {total} signal{'' if total <= 1 else 's'} évalué{'' if total <= 1 else 's'}"]
    if buy or sell:
        parts.append(f"dont {buy} BUY et {sell} SELL")

    breakdown_items: list[tuple[str, int]] = []
    for key in REASONS_IN_ORDER:
        val = int((rejected_breakdown or {}).get(key) or 0)
        if val > 0:
            breakdown_items.append((key, val))

    if rejected > 0 and not breakdown_items and no_trade_reason:
        breakdown_items.append((no_trade_reason, rejected))

    if breakdown_items:
        formatted = []
        for key, count in breakdown_items:
            label = NO_TRADE_REASON_LABELS.get(key, key.replace("_", " ").capitalize())
            formatted.append(f"{count} pour « {label} »")
        if len(formatted) == 1:
            parts.append(f"{rejected} rejeté{'' if rejected <= 1 else 's'} : {formatted[0]}")
        else:
            parts.append(
                f"{rejected} rejetés : "
                + ", ".join(formatted[:-1])
                + " et "
                + formatted[-1]
            )
    elif rejected > 0:
        parts.append(f"{rejected} rejeté{'' if rejected <= 1 else 's'} sans motif précisé")

    message = ", ".join(parts) + "."
    return message


def build_trade_message(
    total_signals: int,
    trades_opened: int,
    trades_won: int,
    trades_lost: int,
    pnl: float,
) -> str:
    total = max(0, int(total_signals or 0))
    opened = max(0, int(trades_opened or 0))
    won = max(0, int(trades_won or 0))
    lost = max(0, int(trades_lost or 0))
    pnl_val = float(pnl or 0.0)
    if opened == 0:
        return build_no_trade_message(total_signals, 0, 0, total, {}, None)
    parts = [
        f"Session active : {total} signal{'' if total <= 1 else 's'} évalué{'' if total <= 1 else 's'}",
        f"{opened} trade{'' if opened <= 1 else 's'} ouvert{'' if opened <= 1 else 's'}",
    ]
    if won or lost:
        parts.append(f"{won} gagnant{'' if won <= 1 else 's'} / {lost} perdant{'' if lost <= 1 else 's'}")
    parts.append(f"P&L du jour : {pnl_val:+.2f}")
    return ", ".join(parts) + "."
