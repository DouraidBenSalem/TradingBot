import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import Json, RealDictCursor, RealDictRow

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

_MONITORING_TABLES_READY = False


def get_connection():

    connection = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "trading_bot"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD")
    )

    return connection


def _add_column_if_missing(cursor, table: str, col: str, definition: str):
    """Ajouter une colonne si elle n'existe pas (idempotent)."""
    cursor.execute(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = split_part(%s, '.', 1)
          AND table_name   = split_part(%s, '.', 2)
          AND column_name  = %s
        """,
        (table, table, col),
    )
    if not cursor.fetchone():
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {definition}")


def ensure_trade_history_table(connection=None):
    """Créer ou migrer la table trade_history (transactions + justifications structurées).

    Idempotent :
      - CREATE TABLE IF NOT EXISTS pour les bases vierges.
      - Ajout automatique des colonnes manquantes sur les tables existantes.
      - Rattrapage des valeurs NULL pour les colonnes NOT NULL avant application
        de la contrainte.
      - Création des index (IF NOT EXISTS).
    """
    own_conn = connection is None
    if own_conn:
        connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS public.trade_history (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cols = [
                ("backtest_run_id",    "BIGINT"),
                ("trade_sequence",     "INTEGER"),
                ("source",             "VARCHAR(32) NOT NULL DEFAULT 'backtest'"),
                ("decision_type",      "VARCHAR(16)"),
                ("entry_time",         "TIMESTAMP"),
                ("exit_time",          "TIMESTAMP"),
                ("entry_date",         "DATE"),
                ("entry_hour",         "VARCHAR(12)"),
                ("exit_hour",          "VARCHAR(12)"),
                ("duration_seconds",   "INTEGER"),
                ("duration_minutes",   "DOUBLE PRECISION"),
                ("duration_text",      "VARCHAR(64)"),
                ("side",               "VARCHAR(8)"),
                ("entry_price",        "DOUBLE PRECISION"),
                ("exit_price",         "DOUBLE PRECISION"),
                ("sl_price",           "DOUBLE PRECISION"),
                ("tp_price",           "DOUBLE PRECISION"),
                ("sl_dollar",          "DOUBLE PRECISION"),
                ("tp_dollar",          "DOUBLE PRECISION"),
                ("volume",             "DOUBLE PRECISION"),
                ("lot",                "DOUBLE PRECISION"),
                ("ema_9",              "DOUBLE PRECISION"),
                ("ema_21",             "DOUBLE PRECISION"),
                ("ema_50",             "DOUBLE PRECISION"),
                ("rsi",                "DOUBLE PRECISION"),
                ("macd",               "DOUBLE PRECISION"),
                ("macd_hist",          "DOUBLE PRECISION"),
                ("atr",                "DOUBLE PRECISION"),
                ("score",              "INTEGER"),
                ("score_breakdown",    "JSONB"),
                ("signal_details",     "JSONB"),
                ("price_pnl",          "DOUBLE PRECISION"),
                ("commission",         "DOUBLE PRECISION"),
                ("slippage",           "DOUBLE PRECISION"),
                ("net_pnl",            "DOUBLE PRECISION"),
                ("balance_before",     "DOUBLE PRECISION"),
                ("balance_after",      "DOUBLE PRECISION"),
                ("outcome",            "VARCHAR(16)"),
                ("exit_reason_code",   "VARCHAR(32)"),
                ("exit_reason",        "TEXT"),
                ("justification_text", "TEXT"),
                ("justification_struct","JSONB"),
                ("raw_trade",          "JSONB"),
            ]
            for col, definition in cols:
                _add_column_if_missing(cursor, "public.trade_history", col, definition)

            cursor.execute(
                """
                UPDATE public.trade_history
                SET decision_type = COALESCE(UPPER(side), 'NOT_TRADE')
                WHERE decision_type IS NULL
                """
            )
            cursor.execute(
                """
                UPDATE public.trade_history
                SET entry_time = COALESCE(entry_date::TIMESTAMP, created_at)
                WHERE entry_time IS NULL
                """
            )

            cursor.execute(
                "ALTER TABLE public.trade_history ALTER COLUMN decision_type SET NOT NULL"
            )
            cursor.execute(
                "ALTER TABLE public.trade_history ALTER COLUMN entry_time SET NOT NULL"
            )

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS trade_history_entry_time_idx ON public.trade_history (entry_time DESC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS trade_history_side_idx ON public.trade_history (side)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS trade_history_outcome_idx ON public.trade_history (outcome)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS trade_history_decision_type_idx ON public.trade_history (decision_type)"
            )
            cursor.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS trade_history_run_sequence_unique_idx
                ON public.trade_history (backtest_run_id, trade_sequence)
                WHERE backtest_run_id IS NOT NULL AND trade_sequence IS NOT NULL
                """
            )
        connection.commit()
    finally:
        if own_conn:
            connection.close()


def insert_trades_batch(trades: List[Dict[str, Any]], source: str = "backtest",
                        backtest_run_id: Optional[int] = None, connection=None) -> List[int]:
    """Insérer un lot de trades dans trade_history et retourner la liste des IDs insérés."""
    if not trades:
        return []
    ensure_trade_history_table(connection)
    own_conn = connection is None
    if own_conn:
        connection = get_connection()
    inserted_ids: List[int] = []
    try:
        with connection.cursor() as cursor:
            for sequence, t in enumerate(trades):
                side = t.get("side")
                decision_type = (side or "NOT_TRADE").upper()
                trade_sequence = (
                    t.get("trade_sequence", sequence)
                    if backtest_run_id is not None
                    else t.get("trade_sequence")
                )
                cursor.execute(
                    """
                    INSERT INTO public.trade_history (
                        backtest_run_id, trade_sequence, source, decision_type,
                        entry_time, exit_time, entry_date, entry_hour, exit_hour,
                        duration_seconds, duration_minutes, duration_text,
                        side, entry_price, exit_price, sl_price, tp_price,
                        sl_dollar, tp_dollar, volume, lot,
                        ema_9, ema_21, ema_50, rsi, macd, macd_hist, atr,
                        score, score_breakdown, signal_details,
                        price_pnl, commission, slippage, net_pnl,
                        balance_before, balance_after, outcome,
                        exit_reason_code, exit_reason,
                        justification_text, justification_struct, raw_trade
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s,
                        %s, %s, %s
                    )
                    ON CONFLICT (backtest_run_id, trade_sequence)
                    WHERE backtest_run_id IS NOT NULL AND trade_sequence IS NOT NULL
                    DO UPDATE SET
                        source = EXCLUDED.source,
                        decision_type = EXCLUDED.decision_type,
                        entry_time = EXCLUDED.entry_time,
                        exit_time = EXCLUDED.exit_time,
                        entry_date = EXCLUDED.entry_date,
                        entry_hour = EXCLUDED.entry_hour,
                        exit_hour = EXCLUDED.exit_hour,
                        duration_seconds = EXCLUDED.duration_seconds,
                        duration_minutes = EXCLUDED.duration_minutes,
                        duration_text = EXCLUDED.duration_text,
                        side = EXCLUDED.side,
                        entry_price = EXCLUDED.entry_price,
                        exit_price = EXCLUDED.exit_price,
                        sl_price = EXCLUDED.sl_price,
                        tp_price = EXCLUDED.tp_price,
                        sl_dollar = EXCLUDED.sl_dollar,
                        tp_dollar = EXCLUDED.tp_dollar,
                        volume = EXCLUDED.volume,
                        lot = EXCLUDED.lot,
                        ema_9 = EXCLUDED.ema_9,
                        ema_21 = EXCLUDED.ema_21,
                        ema_50 = EXCLUDED.ema_50,
                        rsi = EXCLUDED.rsi,
                        macd = EXCLUDED.macd,
                        macd_hist = EXCLUDED.macd_hist,
                        atr = EXCLUDED.atr,
                        score = EXCLUDED.score,
                        score_breakdown = EXCLUDED.score_breakdown,
                        signal_details = EXCLUDED.signal_details,
                        price_pnl = EXCLUDED.price_pnl,
                        commission = EXCLUDED.commission,
                        slippage = EXCLUDED.slippage,
                        net_pnl = EXCLUDED.net_pnl,
                        balance_before = EXCLUDED.balance_before,
                        balance_after = EXCLUDED.balance_after,
                        outcome = EXCLUDED.outcome,
                        exit_reason_code = EXCLUDED.exit_reason_code,
                        exit_reason = EXCLUDED.exit_reason,
                        justification_text = EXCLUDED.justification_text,
                        justification_struct = EXCLUDED.justification_struct,
                        raw_trade = EXCLUDED.raw_trade
                    RETURNING id
                    """,
                    (
                        backtest_run_id, trade_sequence, source, decision_type,
                        t.get("entry_time") or t.get("time"),
                        t.get("exit_time"),
                        t.get("entry_date"),
                        t.get("entry_hour"),
                        t.get("exit_hour"),
                        t.get("duration_seconds"),
                        t.get("duration_minutes"),
                        t.get("duration_text"),
                        side,
                        t.get("entry"),
                        t.get("exit"),
                        t.get("sl"),
                        t.get("tp"),
                        t.get("sl_dollar"),
                        t.get("tp_dollar"),
                        t.get("volume"),
                        t.get("lot"),
                        t.get("ema_9"),
                        t.get("ema_21"),
                        t.get("ema_50"),
                        t.get("rsi"),
                        t.get("macd"),
                        t.get("macd_hist"),
                        t.get("atr"),
                        t.get("score"),
                        Json(t.get("score_breakdown")) if t.get("score_breakdown") else None,
                        Json(t.get("signal_details")) if t.get("signal_details") else None,
                        t.get("price_pnl"),
                        t.get("commission"),
                        t.get("slippage"),
                        t.get("net_pnl"),
                        t.get("balance_before"),
                        t.get("balance"),
                        t.get("outcome"),
                        t.get("exit_reason_code"),
                        t.get("exit_reason"),
                        t.get("why_trade"),
                        Json(_build_justification_struct(t)) if t else None,
                        Json(t),
                    ),
                )
                row = cursor.fetchone()
                if row:
                    inserted_ids.append(int(row[0]))
        connection.commit()
    finally:
        if own_conn:
            connection.close()
    return inserted_ids


def _build_justification_struct(t: Dict[str, Any]) -> Dict[str, Any]:
    """Construire la justification structurée à partir des données du trade."""
    sb = t.get("score_breakdown") or {}
    sd = t.get("signal_details") or {}
    side = t.get("side") or "NOT_TRADE"
    return {
        "decision": side,
        "score": t.get("score"),
        "score_required": 7,
        "score_total_max": sb.get("_total", {}).get("max", 8),
        "score_components": {
            "trend": sb.get("trend"),
            "pullback": sb.get("pullback"),
            "atr": sb.get("atr"),
            "rsi": sb.get("rsi"),
            "macd": sb.get("macd"),
            "candle": sb.get("candle"),
        },
        "signals": {
            "trend": sd.get("trend"),
            "pullback": sd.get("pullback"),
            "volatility": sd.get("volatility"),
            "rsi": sd.get("rsi"),
            "macd": sd.get("macd"),
            "candle": sd.get("candle"),
        },
        "indicators_snapshot": {
            "ema_9": t.get("ema_9"),
            "ema_21": t.get("ema_21"),
            "ema_50": t.get("ema_50"),
            "rsi_14": t.get("rsi"),
            "macd": t.get("macd"),
            "macd_signal": t.get("macd_signal"),
            "macd_hist": t.get("macd_hist"),
            "atr_14": t.get("atr"),
            "volume_ratio": t.get("volume_ratio"),
            "session": t.get("session"),
        },
        "risk_reward": {
            "sl": t.get("sl"),
            "tp": t.get("tp"),
            "sl_dollar": t.get("sl_dollar"),
            "tp_dollar": t.get("tp_dollar"),
            "ratio_rr": (round(t["tp_dollar"] / t["sl_dollar"], 2)
                         if t.get("sl_dollar") and t["sl_dollar"] > 0 and t.get("tp_dollar") else None),
        },
        "why_text": t.get("why_trade"),
    }


def list_trade_history(limit: int = 500, offset: int = 0,
                       side_filter: Optional[str] = None,
                       outcome_filter: Optional[str] = None,
                       decision_type: Optional[str] = None,
                       connection=None) -> List[RealDictRow]:
    """Récupérer l'historique des trades (avec filtres optionnels)."""
    ensure_trade_history_table(connection)
    own_conn = connection is None
    if own_conn:
        connection = get_connection()
    clauses = []
    params: List[Any] = []
    if side_filter:
        clauses.append("side = %s")
        params.append(side_filter.upper())
    if outcome_filter:
        clauses.append("outcome = %s")
        params.append(outcome_filter.upper())
    if decision_type:
        clauses.append("decision_type = %s")
        params.append(decision_type.upper())
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    query = f"""
        SELECT * FROM public.trade_history
        {where}
        ORDER BY entry_time DESC, id DESC
        LIMIT %s OFFSET %s
    """
    params.extend([limit, offset])
    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, tuple(params))
            return cursor.fetchall()
    finally:
        if own_conn:
            connection.close()


def count_trade_history(side_filter: Optional[str] = None,
                        outcome_filter: Optional[str] = None,
                        decision_type: Optional[str] = None,
                        connection=None) -> int:
    """Compter le nombre de trades (avec filtres optionnels)."""
    ensure_trade_history_table(connection)
    own_conn = connection is None
    if own_conn:
        connection = get_connection()
    clauses = []
    params: List[Any] = []
    if side_filter:
        clauses.append("side = %s")
        params.append(side_filter.upper())
    if outcome_filter:
        clauses.append("outcome = %s")
        params.append(outcome_filter.upper())
    if decision_type:
        clauses.append("decision_type = %s")
        params.append(decision_type.upper())
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    query = f"SELECT COUNT(*) AS c FROM public.trade_history {where}"
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, tuple(params))
            row = cursor.fetchone()
            return int(row[0]) if row else 0
    finally:
        if own_conn:
            connection.close()


def get_trade_by_id(trade_id: int, connection=None) -> Optional[RealDictRow]:
    """Récupérer un trade par son ID (incluant justification structurée)."""
    ensure_trade_history_table(connection)
    own_conn = connection is None
    if own_conn:
        connection = get_connection()
    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT * FROM public.trade_history WHERE id = %s",
                (trade_id,),
            )
            return cursor.fetchone()
    finally:
        if own_conn:
            connection.close()


def delete_trade_by_id(trade_id: int, connection=None) -> bool:
    """Supprimer un trade par son ID. Retourne True si la suppression a eu lieu."""
    ensure_trade_history_table(connection)
    own_conn = connection is None
    if own_conn:
        connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM public.trade_history WHERE id = %s RETURNING id",
                (trade_id,),
            )
            row = cursor.fetchone()
        connection.commit()
        return row is not None
    finally:
        if own_conn:
            connection.close()


def create_tables():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS eurusd_m1 (
            time TIMESTAMP PRIMARY KEY,
            open DOUBLE PRECISION NOT NULL,
            high DOUBLE PRECISION NOT NULL,
            low DOUBLE PRECISION NOT NULL,
            close DOUBLE PRECISION NOT NULL,
            tick_volume BIGINT,
            spread INTEGER,
            real_volume BIGINT
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS eurusd_m5 (
            time TIMESTAMP PRIMARY KEY,
            open DOUBLE PRECISION NOT NULL,
            high DOUBLE PRECISION NOT NULL,
            low DOUBLE PRECISION NOT NULL,
            close DOUBLE PRECISION NOT NULL,
            tick_volume BIGINT,
            spread INTEGER,
            real_volume BIGINT
        );
    """)

    connection.commit()

    cursor.close()
    connection.close()

    print("Tables PostgreSQL créées.")

def insert_market_data(df, table_name):

    connection = get_connection()

    cursor = connection.cursor()

    query = f"""
        INSERT INTO {table_name}
        (
            time,
            open,
            high,
            low,
            close,
            tick_volume,
            spread,
            real_volume
        )
        VALUES (
            %s, %s, %s, %s,
            %s, %s, %s, %s
        )
        ON CONFLICT (time)
        DO NOTHING;
    """

    for _, row in df.iterrows():

        cursor.execute(
            query,
            (
                row["time"],
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["tick_volume"],
                row["spread"],
                row["real_volume"]
            )
        )

    connection.commit()

    cursor.close()
    connection.close()

    if table_name == "eurusd_m1" and not df.empty:
        from app.data.feature_engineering import sync_features

        synced = sync_features(df["time"].tolist())
        print(f"{synced} lignes synchronisées dans eurusd_features")

    print(
        f"{len(df)} lignes traitées pour {table_name}"
    )


def get_last_timestamp(table_name):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        f"SELECT MAX(time) FROM {table_name}"
    )

    result = cursor.fetchone()

    cursor.close()
    connection.close()

    return result[0] if result else None


def get_missing_ranges(table_name, interval_minutes=1):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        f"""
        WITH ordered AS (
            SELECT
                time,
                LAG(time) OVER (ORDER BY time) AS previous_time
            FROM {table_name}
        )
        SELECT
            previous_time + (%s * INTERVAL '1 minute'),
            time - (%s * INTERVAL '1 minute')
        FROM ordered
        WHERE previous_time IS NOT NULL
          AND time > previous_time + (%s * INTERVAL '1 minute')
                    AND NOT EXISTS (
                            SELECT 1
                            FROM generate_series(
                                    previous_time::date,
                                    time::date,
                                    INTERVAL '1 day'
                            ) AS calendar_day
                            WHERE EXTRACT(ISODOW FROM calendar_day) IN (6, 7)
                    )
        ORDER BY previous_time
        """,
        (interval_minutes, interval_minutes, interval_minutes)
    )

    missing_ranges = cursor.fetchall()

    cursor.close()
    connection.close()

    return missing_ranges


def ensure_daily_sessions_table(connection=None):
    own_conn = connection is None
    if own_conn:
        connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS public.mt5_demo_daily_sessions (
                    id BIGSERIAL PRIMARY KEY,
                    session_date DATE NOT NULL,
                    opened_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    closed_at TIMESTAMP,
                    duration_seconds INTEGER,
                    strategy VARCHAR(64) NOT NULL DEFAULT 'ema_pullback_wider_stop',
                    symbol VARCHAR(16) NOT NULL DEFAULT 'EURUSD',
                    lot DOUBLE PRECISION NOT NULL DEFAULT 0.07,
                    account_login BIGINT,
                    account_server VARCHAR(128),
                    account_is_demo BOOLEAN NOT NULL DEFAULT FALSE,
                    status VARCHAR(16) NOT NULL DEFAULT 'RUNNING',
                    total_signals INTEGER NOT NULL DEFAULT 0,
                    buy_signals INTEGER NOT NULL DEFAULT 0,
                    sell_signals INTEGER NOT NULL DEFAULT 0,
                    rejected_signals INTEGER NOT NULL DEFAULT 0,
                    trades_opened INTEGER NOT NULL DEFAULT 0,
                    trades_won INTEGER NOT NULL DEFAULT 0,
                    trades_lost INTEGER NOT NULL DEFAULT 0,
                    trades_cancelled INTEGER NOT NULL DEFAULT 0,
                    pnl DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                    no_trade_reason VARCHAR(64),
                    rejected_breakdown JSONB,
                    last_signal JSONB,
                    message TEXT,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cols = [
                ("session_date",       "DATE"),
                ("opened_at",          "TIMESTAMP"),
                ("closed_at",          "TIMESTAMP"),
                ("duration_seconds",   "INTEGER"),
                ("strategy",           "VARCHAR(64)"),
                ("symbol",             "VARCHAR(16)"),
                ("lot",                "DOUBLE PRECISION"),
                ("account_login",      "BIGINT"),
                ("account_server",     "VARCHAR(128)"),
                ("account_is_demo",    "BOOLEAN"),
                ("status",             "VARCHAR(16)"),
                ("total_signals",      "INTEGER"),
                ("buy_signals",        "INTEGER"),
                ("sell_signals",       "INTEGER"),
                ("rejected_signals",   "INTEGER"),
                ("trades_opened",      "INTEGER"),
                ("trades_won",         "INTEGER"),
                ("trades_lost",        "INTEGER"),
                ("trades_cancelled",   "INTEGER"),
                ("pnl",                "DOUBLE PRECISION"),
                ("no_trade_reason",    "VARCHAR(64)"),
                ("rejected_breakdown", "JSONB"),
                ("last_signal",        "JSONB"),
                ("message",            "TEXT"),
                ("updated_at",         "TIMESTAMP"),
            ]
            for col, definition in cols:
                _add_column_if_missing(cursor, "public.mt5_demo_daily_sessions", col, definition)

            cursor.execute(
                """
                SELECT 1 FROM pg_indexes
                WHERE schemaname = 'public'
                  AND tablename = 'mt5_demo_daily_sessions'
                  AND indexname = 'mt5_demo_daily_sessions_unique_key'
                """
            )
            if not cursor.fetchone():
                try:
                    cursor.execute(
                        """
                        CREATE UNIQUE INDEX mt5_demo_daily_sessions_unique_key
                        ON public.mt5_demo_daily_sessions (session_date, COALESCE(account_login, 0), strategy, symbol)
                        """
                    )
                except Exception:
                    connection.rollback()
                    with connection.cursor() as c2:
                        c2.execute(
                            """
                            SELECT COUNT(*) = 0 FROM pg_indexes
                            WHERE schemaname = 'public'
                              AND tablename = 'mt5_demo_daily_sessions'
                              AND indexname = 'mt5_demo_daily_sessions_unique_key'
                            """
                        )
                        if c2.fetchone()[0]:
                            raise

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS mt5_demo_daily_sessions_date_idx ON public.mt5_demo_daily_sessions (session_date DESC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS mt5_demo_daily_sessions_status_idx ON public.mt5_demo_daily_sessions (status)"
            )

            cursor.execute(
                """
                UPDATE public.mt5_demo_daily_sessions
                SET strategy = 'ema_pullback_wider_stop' WHERE strategy IS NULL OR strategy = ''
                """
            )
            cursor.execute(
                "UPDATE public.mt5_demo_daily_sessions SET symbol = 'EURUSD' WHERE symbol IS NULL OR symbol = ''"
            )
            cursor.execute(
                "UPDATE public.mt5_demo_daily_sessions SET lot = 0.07 WHERE lot IS NULL"
            )
            cursor.execute(
                "UPDATE public.mt5_demo_daily_sessions SET status = 'RUNNING' WHERE status IS NULL OR status = ''"
            )
            cursor.execute(
                "UPDATE public.mt5_demo_daily_sessions SET account_is_demo = FALSE WHERE account_is_demo IS NULL"
            )

            for col, _ in cols:
                cursor.execute(
                    f"""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name   = 'mt5_demo_daily_sessions'
                      AND column_name  = %s
                    """,
                    (col,),
                )
                if not cursor.fetchone():
                    raise RuntimeError(f"Missing column after migration: {col}")
        connection.commit()
    finally:
        if own_conn:
            connection.close()


def upsert_daily_session(partial: Dict[str, Any], connection=None) -> int:
    ensure_daily_sessions_table(connection)
    own_conn = connection is None
    if own_conn:
        connection = get_connection()
    try:
        existing = get_daily_session_by_date(
            partial.get("session_date"),
            partial.get("account_login"),
            partial.get("strategy") or "ema_pullback_wider_stop",
            partial.get("symbol") or "EURUSD",
            connection=connection,
        )
        if existing:
            merged = dict(existing)
            for k, v in partial.items():
                if v is None:
                    continue
                if isinstance(merged.get(k), dict) and isinstance(v, dict):
                    merged[k] = {**merged[k], **v}
                else:
                    merged[k] = v
            merged["updated_at"] = datetime.now()
            with connection.cursor() as cursor:
                set_clauses = []
                params: List[Any] = []
                for key in (
                    "opened_at", "closed_at", "duration_seconds", "strategy", "symbol",
                    "lot", "account_login", "account_server", "account_is_demo",
                    "status", "total_signals", "buy_signals", "sell_signals",
                    "rejected_signals", "trades_opened", "trades_won", "trades_lost",
                    "trades_cancelled", "pnl", "no_trade_reason", "rejected_breakdown",
                    "last_signal", "message", "updated_at",
                ):
                    if key in merged:
                        val = merged[key]
                        if key in ("rejected_breakdown", "last_signal") and isinstance(val, dict):
                            val = Json(val)
                        set_clauses.append(f"{key} = %s")
                        params.append(val)
                params.append(int(existing["id"]))
                cursor.execute(
                    f"UPDATE public.mt5_demo_daily_sessions SET {', '.join(set_clauses)} WHERE id = %s",
                    tuple(params),
                )
            connection.commit()
            return int(existing["id"])

        session_date = partial.get("session_date")
        if isinstance(session_date, datetime):
            session_date = session_date.date()
        if session_date is None:
            session_date = datetime.now().date()
        account_login = partial.get("account_login") or 0
        strategy = partial.get("strategy") or "ema_pullback_wider_stop"
        symbol = partial.get("symbol") or "EURUSD"
        opened_at = partial.get("opened_at") or datetime.now()
        account_is_demo = bool(partial.get("account_is_demo", False))
        lot = float(partial.get("lot") or 0.07)
        status = partial.get("status") or "RUNNING"
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO public.mt5_demo_daily_sessions
                (session_date, opened_at, strategy, symbol, lot, account_login,
                 account_server, account_is_demo, status,
                 total_signals, buy_signals, sell_signals, rejected_signals,
                 trades_opened, trades_won, trades_lost, trades_cancelled,
                 pnl, no_trade_reason, rejected_breakdown, last_signal, message, updated_at)
                VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s,
                 %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    session_date, opened_at, strategy, symbol, lot, account_login,
                    partial.get("account_server"), account_is_demo, status,
                    int(partial.get("total_signals") or 0),
                    int(partial.get("buy_signals") or 0),
                    int(partial.get("sell_signals") or 0),
                    int(partial.get("rejected_signals") or 0),
                    int(partial.get("trades_opened") or 0),
                    int(partial.get("trades_won") or 0),
                    int(partial.get("trades_lost") or 0),
                    int(partial.get("trades_cancelled") or 0),
                    float(partial.get("pnl") or 0.0),
                    partial.get("no_trade_reason"),
                    Json(partial["rejected_breakdown"]) if isinstance(partial.get("rejected_breakdown"), dict) else None,
                    Json(partial["last_signal"]) if isinstance(partial.get("last_signal"), dict) else None,
                    partial.get("message"),
                    datetime.now(),
                ),
            )
            row = cursor.fetchone()
        connection.commit()
        return int(row[0]) if row else -1
    finally:
        if own_conn:
            connection.close()


def get_daily_session_by_date(session_date=None, account_login=None,
                              strategy: str = "ema_pullback_wider_stop",
                              symbol: str = "EURUSD",
                              connection=None) -> Optional[RealDictRow]:
    ensure_daily_sessions_table(connection)
    if session_date is None:
        session_date = datetime.now().date()
    elif isinstance(session_date, datetime):
        session_date = session_date.date()
    own_conn = connection is None
    if own_conn:
        connection = get_connection()
    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT * FROM public.mt5_demo_daily_sessions
                WHERE session_date = %s
                  AND COALESCE(account_login, 0) = %s
                  AND strategy = %s
                  AND symbol = %s
                ORDER BY id DESC LIMIT 1
                """,
                (session_date, account_login or 0, strategy, symbol),
            )
            return cursor.fetchone()
    finally:
        if own_conn:
            connection.close()


def list_daily_sessions(limit: int = 30, connection=None) -> List[RealDictRow]:
    ensure_daily_sessions_table(connection)
    own_conn = connection is None
    if own_conn:
        connection = get_connection()
    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT * FROM public.mt5_demo_daily_sessions
                ORDER BY session_date DESC, id DESC
                LIMIT %s
                """,
                (max(1, int(limit)),),
            )
            return cursor.fetchall()
    finally:
        if own_conn:
            connection.close()


def close_daily_session(session_id: int, overrides: Optional[Dict[str, Any]] = None,
                        connection=None) -> bool:
    ensure_daily_sessions_table(connection)
    overrides = overrides or {}
    own_conn = connection is None
    if own_conn:
        connection = get_connection()
    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT * FROM public.mt5_demo_daily_sessions WHERE id = %s",
                (int(session_id),),
            )
            row = cursor.fetchone()
        if not row:
            return False
        merged = dict(row)
        for k, v in overrides.items():
            if v is not None:
                merged[k] = v
        closed_at = merged.get("closed_at") or datetime.now()
        opened_at = merged.get("opened_at") or closed_at
        duration_seconds = merged.get("duration_seconds")
        if duration_seconds is None:
            try:
                duration_seconds = int((closed_at - opened_at).total_seconds())
            except Exception:
                duration_seconds = 0
        trades_opened = int(merged.get("trades_opened") or 0)
        current_status = merged.get("status") or "RUNNING"
        if current_status in ("RUNNING",):
            status = "TRADE" if trades_opened > 0 else "NO_TRADE"
        else:
            status = current_status
        no_trade_reason = merged.get("no_trade_reason")
        if status == "TRADE":
            no_trade_reason = None
        elif status == "NO_TRADE" and not no_trade_reason:
            no_trade_reason = "SIGNAL_CONFLICT"
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE public.mt5_demo_daily_sessions
                SET closed_at = %s,
                    duration_seconds = %s,
                    trades_won = %s,
                    trades_lost = %s,
                    trades_cancelled = %s,
                    pnl = %s,
                    status = %s,
                    no_trade_reason = %s,
                    message = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (
                    closed_at,
                    duration_seconds,
                    int(merged.get("trades_won") or 0),
                    int(merged.get("trades_lost") or 0),
                    int(merged.get("trades_cancelled") or 0),
                    float(merged.get("pnl") or 0.0),
                    status,
                    no_trade_reason,
                    merged.get("message"),
                    int(session_id),
                ),
            )
        connection.commit()
        return True
    finally:
        if own_conn:
            connection.close()


# =====================================================================
# Monitoring persistant du bot et de MT5
# =====================================================================

def ensure_monitoring_tables(connection=None):
    """Create the append-only monitoring tables used by the dashboards.

    These tables deliberately live next to ``trade_history`` instead of
    overloading it: a rejected signal is a decision, not a trade.  Keeping
    those concepts separate also makes the migration safe for databases that
    still have a historical ``trade_history.side NOT NULL`` constraint.
    """
    global _MONITORING_TABLES_READY
    if _MONITORING_TABLES_READY:
        return
    own_conn = connection is None
    if own_conn:
        connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS public.bot_decisions (
                    id BIGSERIAL PRIMARY KEY,
                    occurred_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    symbol VARCHAR(32) NOT NULL,
                    timeframe VARCHAR(16) NOT NULL,
                    current_price DOUBLE PRECISION,
                    signal VARCHAR(32) NOT NULL,
                    signal_score INTEGER,
                    conditions JSONB NOT NULL DEFAULT '{}'::jsonb,
                    decision VARCHAR(64) NOT NULL,
                    action VARCHAR(16) NOT NULL,
                    refusal_reason TEXT,
                    entry_reason TEXT,
                    stop_loss DOUBLE PRECISION,
                    take_profit DOUBLE PRECISION,
                    lot_size DOUBLE PRECISION,
                    trade_ticket BIGINT,
                    trade_result DOUBLE PRECISION,
                    status VARCHAR(24) NOT NULL DEFAULT 'RECORDED',
                    diagnostics JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS bot_decisions_candle_unique_idx
                ON public.bot_decisions (occurred_at, symbol, timeframe)
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS bot_decisions_occurred_idx ON public.bot_decisions (occurred_at DESC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS bot_decisions_action_idx ON public.bot_decisions (action)"
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS public.bot_activity_log (
                    id BIGSERIAL PRIMARY KEY,
                    occurred_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    event_type VARCHAR(64) NOT NULL,
                    severity VARCHAR(16) NOT NULL DEFAULT 'INFO',
                    component VARCHAR(32) NOT NULL DEFAULT 'BOT',
                    message TEXT NOT NULL,
                    symbol VARCHAR(32),
                    ticket BIGINT,
                    details JSONB NOT NULL DEFAULT '{}'::jsonb
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS bot_activity_occurred_idx ON public.bot_activity_log (occurred_at DESC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS bot_activity_type_idx ON public.bot_activity_log (event_type)"
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS public.mt5_connection_events (
                    id BIGSERIAL PRIMARY KEY,
                    occurred_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    status VARCHAR(32) NOT NULL,
                    terminal_open BOOLEAN NOT NULL DEFAULT FALSE,
                    connected BOOLEAN NOT NULL DEFAULT FALSE,
                    account_available BOOLEAN NOT NULL DEFAULT FALSE,
                    account_login BIGINT,
                    account_server VARCHAR(128),
                    error_message TEXT,
                    details JSONB NOT NULL DEFAULT '{}'::jsonb
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS mt5_connection_occurred_idx ON public.mt5_connection_events (occurred_at DESC)"
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS public.mt5_trade_history (
                    id BIGSERIAL PRIMARY KEY,
                    ticket BIGINT NOT NULL,
                    position_id BIGINT,
                    order_ticket BIGINT,
                    opened_at TIMESTAMP NOT NULL,
                    closed_at TIMESTAMP,
                    symbol VARCHAR(32) NOT NULL,
                    side VARCHAR(8) NOT NULL,
                    entry_price DOUBLE PRECISION,
                    exit_price DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    stop_loss DOUBLE PRECISION,
                    take_profit DOUBLE PRECISION,
                    profit_loss DOUBLE PRECISION,
                    commission DOUBLE PRECISION NOT NULL DEFAULT 0,
                    swap DOUBLE PRECISION NOT NULL DEFAULT 0,
                    duration_seconds INTEGER,
                    status VARCHAR(16) NOT NULL DEFAULT 'OPEN',
                    entry_reason TEXT,
                    exit_reason TEXT,
                    decision_id BIGINT,
                    raw_trade JSONB NOT NULL DEFAULT '{}'::jsonb,
                    deleted_at TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            _add_column_if_missing(
                cursor,
                "public.mt5_trade_history",
                "deleted_at",
                "TIMESTAMP",
            )
            cursor.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS mt5_trade_ticket_unique_idx ON public.mt5_trade_history (ticket)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS mt5_trade_opened_idx ON public.mt5_trade_history (opened_at DESC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS mt5_trade_status_idx ON public.mt5_trade_history (status)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS mt5_trade_symbol_idx ON public.mt5_trade_history (symbol)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS mt5_trade_deleted_idx ON public.mt5_trade_history (deleted_at)"
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS public.bot_metric_snapshots (
                    id BIGSERIAL PRIMARY KEY,
                    captured_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    account_login BIGINT,
                    balance DOUBLE PRECISION,
                    equity DOUBLE PRECISION,
                    free_margin DOUBLE PRECISION,
                    margin DOUBLE PRECISION,
                    floating_profit DOUBLE PRECISION,
                    open_trades INTEGER NOT NULL DEFAULT 0,
                    details JSONB NOT NULL DEFAULT '{}'::jsonb
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS bot_metrics_captured_idx ON public.bot_metric_snapshots (captured_at DESC)"
            )
        connection.commit()
        _MONITORING_TABLES_READY = True
    finally:
        if own_conn:
            connection.close()


def delete_daily_sessions(session_ids, connection=None) -> List[int]:
    """Delete one or more persisted daily summaries in one transaction."""
    ensure_daily_sessions_table(connection)
    ids = sorted({int(value) for value in (session_ids or []) if int(value) > 0})
    if not ids:
        return []
    own_conn = connection is None
    if own_conn:
        connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM public.mt5_demo_daily_sessions
                WHERE id = ANY(%s)
                  AND COALESCE(status, '') <> 'RUNNING'
                RETURNING id
                """,
                (ids,),
            )
            deleted = [int(row[0]) for row in cursor.fetchall()]
        connection.commit()
        return deleted
    finally:
        if own_conn:
            connection.close()


def insert_bot_decision(decision: Dict[str, Any], connection=None) -> int:
    """Persist one fully explained bot decision per completed candle."""
    ensure_monitoring_tables(connection)
    own_conn = connection is None
    if own_conn:
        connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO public.bot_decisions (
                    occurred_at, symbol, timeframe, current_price, signal,
                    signal_score, conditions, decision, action, refusal_reason,
                    entry_reason, stop_loss, take_profit, lot_size, trade_ticket,
                    trade_result, status, diagnostics
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s
                )
                ON CONFLICT (occurred_at, symbol, timeframe) DO UPDATE SET
                    current_price = EXCLUDED.current_price,
                    signal = EXCLUDED.signal,
                    signal_score = EXCLUDED.signal_score,
                    conditions = EXCLUDED.conditions,
                    decision = EXCLUDED.decision,
                    action = EXCLUDED.action,
                    refusal_reason = EXCLUDED.refusal_reason,
                    entry_reason = EXCLUDED.entry_reason,
                    stop_loss = EXCLUDED.stop_loss,
                    take_profit = EXCLUDED.take_profit,
                    lot_size = EXCLUDED.lot_size,
                    trade_ticket = EXCLUDED.trade_ticket,
                    trade_result = COALESCE(EXCLUDED.trade_result, public.bot_decisions.trade_result),
                    status = EXCLUDED.status,
                    diagnostics = EXCLUDED.diagnostics
                RETURNING id
                """,
                (
                    decision.get("occurred_at") or decision.get("time") or datetime.now(),
                    decision.get("symbol") or "EURUSD",
                    decision.get("timeframe") or "M1",
                    decision.get("current_price"),
                    decision.get("signal") or "NO_SIGNAL",
                    decision.get("signal_score"),
                    Json(decision.get("conditions") or {}),
                    decision.get("decision") or "NO TRADE",
                    (decision.get("action") or "NO_TRADE").upper().replace(" ", "_"),
                    decision.get("refusal_reason"),
                    decision.get("entry_reason"),
                    decision.get("stop_loss"),
                    decision.get("take_profit"),
                    decision.get("lot_size"),
                    decision.get("trade_ticket"),
                    decision.get("trade_result"),
                    decision.get("status") or "RECORDED",
                    Json(decision.get("diagnostics") or {}),
                ),
            )
            row = cursor.fetchone()
        connection.commit()
        return int(row[0])
    finally:
        if own_conn:
            connection.close()


def record_bot_event(event_type: str, message: str, severity: str = "INFO",
                     component: str = "BOT", symbol: Optional[str] = None,
                     ticket: Optional[int] = None, details: Optional[Dict[str, Any]] = None,
                     occurred_at=None, connection=None) -> int:
    """Append an operational event to the durable activity journal."""
    ensure_monitoring_tables(connection)
    own_conn = connection is None
    if own_conn:
        connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO public.bot_activity_log
                    (occurred_at, event_type, severity, component, message, symbol, ticket, details)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    occurred_at or datetime.now(), event_type, severity, component,
                    message, symbol, ticket, Json(details or {}),
                ),
            )
            row = cursor.fetchone()
        connection.commit()
        return int(row[0])
    finally:
        if own_conn:
            connection.close()


def record_mt5_snapshot(snapshot: Dict[str, Any], connection=None) -> None:
    """Persist MT5 state transitions and throttled account metric samples."""
    ensure_monitoring_tables(connection)
    own_conn = connection is None
    if own_conn:
        connection = get_connection()
    try:
        status = snapshot.get("connection_status") or (
            "CONNECTED" if snapshot.get("connected") else
            "CONNECTION_ERROR" if snapshot.get("connection_error") else "DISCONNECTED"
        )
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT * FROM public.mt5_connection_events ORDER BY occurred_at DESC, id DESC LIMIT 1"
            )
            previous = cursor.fetchone()
            changed = not previous or any((
                previous.get("status") != status,
                bool(previous.get("terminal_open")) != bool(snapshot.get("terminal_open")),
                bool(previous.get("connected")) != bool(snapshot.get("connected")),
                bool(previous.get("account_available")) != bool(snapshot.get("account_available")),
                (previous.get("error_message") or None) != (snapshot.get("connection_error") or None),
            ))
            if changed:
                cursor.execute(
                    """
                    INSERT INTO public.mt5_connection_events
                        (status, terminal_open, connected, account_available,
                         account_login, account_server, error_message, details)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        status, bool(snapshot.get("terminal_open")), bool(snapshot.get("connected")),
                        bool(snapshot.get("account_available")), snapshot.get("account_login"),
                        snapshot.get("account_server"), snapshot.get("connection_error"), Json(snapshot),
                    ),
                )
                event_type = {
                    "CONNECTED": "MT5_CONNECTED",
                    "DISCONNECTED": "MT5_DISCONNECTED",
                    "CONNECTING": "MT5_CONNECTING",
                    "RECONNECTING": "MT5_RECONNECTING",
                }.get(status, "MT5_CONNECTION_ERROR")
                message = snapshot.get("connection_error") or status.replace("_", " ").title()
                terminal_changed = (not previous and bool(snapshot.get("terminal_open"))) or (
                    previous and bool(previous.get("terminal_open")) != bool(snapshot.get("terminal_open"))
                )
                if terminal_changed:
                    cursor.execute(
                        """
                        INSERT INTO public.bot_activity_log
                            (event_type, severity, component, message, details)
                        VALUES (%s, 'INFO', 'MT5', %s, %s)
                        """,
                        (
                            "MT5_OPENED" if snapshot.get("terminal_open") else "MT5_CLOSED",
                            "MT5 terminal opened" if snapshot.get("terminal_open") else "MT5 terminal closed",
                            Json(snapshot),
                        ),
                    )
                cursor.execute(
                    """
                    INSERT INTO public.bot_activity_log
                        (event_type, severity, component, message, details)
                    VALUES (%s, %s, 'MT5', %s, %s)
                    """,
                    (event_type, "ERROR" if status == "CONNECTION_ERROR" else "INFO", message, Json(snapshot)),
                )

            if snapshot.get("connected") and snapshot.get("account_available"):
                cursor.execute(
                    "SELECT captured_at FROM public.bot_metric_snapshots ORDER BY captured_at DESC, id DESC LIMIT 1"
                )
                latest = cursor.fetchone()
                should_insert = not latest or (datetime.now() - latest["captured_at"]).total_seconds() >= 60
                if should_insert:
                    cursor.execute(
                        """
                        INSERT INTO public.bot_metric_snapshots
                            (account_login, balance, equity, free_margin, margin,
                             floating_profit, open_trades, details)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            snapshot.get("account_login"), snapshot.get("balance"), snapshot.get("equity"),
                            snapshot.get("free_margin"), snapshot.get("margin"), snapshot.get("floating_profit"),
                            len(snapshot.get("positions") or []), Json(snapshot),
                        ),
                    )
        connection.commit()
    finally:
        if own_conn:
            connection.close()


def upsert_mt5_trade(trade: Dict[str, Any], connection=None) -> int:
    """Insert or reconcile one MT5 position without duplicating its ticket."""
    ensure_monitoring_tables(connection)
    own_conn = connection is None
    if own_conn:
        connection = get_connection()
    ticket = int(trade.get("ticket") or trade.get("position_id") or trade.get("order_ticket") or 0)
    if not ticket:
        raise ValueError("An MT5 trade ticket or position_id is required")
    try:
        with connection.cursor() as cursor:
            if trade.get("position_id") or trade.get("order_ticket"):
                cursor.execute(
                    """
                    SELECT ticket FROM public.mt5_trade_history
                    WHERE (%s IS NOT NULL AND position_id = %s)
                       OR (%s IS NOT NULL AND order_ticket = %s)
                    ORDER BY id DESC LIMIT 1
                    """,
                    (
                        trade.get("position_id"), trade.get("position_id"),
                        trade.get("order_ticket"), trade.get("order_ticket"),
                    ),
                )
                existing = cursor.fetchone()
                if existing:
                    ticket = int(existing[0])
            cursor.execute(
                """
                INSERT INTO public.mt5_trade_history (
                    ticket, position_id, order_ticket, opened_at, closed_at, symbol,
                    side, entry_price, exit_price, volume, stop_loss, take_profit,
                    profit_loss, commission, swap, duration_seconds, status,
                    entry_reason, exit_reason, decision_id, raw_trade
                ) VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
                ON CONFLICT (ticket) DO UPDATE SET
                    position_id = COALESCE(EXCLUDED.position_id, public.mt5_trade_history.position_id),
                    order_ticket = COALESCE(EXCLUDED.order_ticket, public.mt5_trade_history.order_ticket),
                    opened_at = LEAST(EXCLUDED.opened_at, public.mt5_trade_history.opened_at),
                    closed_at = COALESCE(EXCLUDED.closed_at, public.mt5_trade_history.closed_at),
                    symbol = EXCLUDED.symbol,
                    side = EXCLUDED.side,
                    entry_price = COALESCE(EXCLUDED.entry_price, public.mt5_trade_history.entry_price),
                    exit_price = COALESCE(EXCLUDED.exit_price, public.mt5_trade_history.exit_price),
                    volume = COALESCE(EXCLUDED.volume, public.mt5_trade_history.volume),
                    stop_loss = COALESCE(NULLIF(EXCLUDED.stop_loss, 0), public.mt5_trade_history.stop_loss),
                    take_profit = COALESCE(NULLIF(EXCLUDED.take_profit, 0), public.mt5_trade_history.take_profit),
                    profit_loss = COALESCE(EXCLUDED.profit_loss, public.mt5_trade_history.profit_loss),
                    commission = EXCLUDED.commission,
                    swap = EXCLUDED.swap,
                    duration_seconds = COALESCE(EXCLUDED.duration_seconds, public.mt5_trade_history.duration_seconds),
                    status = EXCLUDED.status,
                    entry_reason = COALESCE(EXCLUDED.entry_reason, public.mt5_trade_history.entry_reason),
                    exit_reason = COALESCE(EXCLUDED.exit_reason, public.mt5_trade_history.exit_reason),
                    decision_id = COALESCE(EXCLUDED.decision_id, public.mt5_trade_history.decision_id),
                    raw_trade = EXCLUDED.raw_trade,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
                """,
                (
                    ticket, trade.get("position_id"), trade.get("order_ticket"),
                    trade.get("opened_at") or trade.get("entry_time") or datetime.now(), trade.get("closed_at"),
                    trade.get("symbol") or "EURUSD", (trade.get("side") or "BUY").upper(),
                    trade.get("entry_price") or trade.get("entry"), trade.get("exit_price") or trade.get("exit"),
                    trade.get("volume") or trade.get("lot"), trade.get("stop_loss") or trade.get("sl"),
                    trade.get("take_profit") or trade.get("tp"), trade.get("profit_loss") if trade.get("profit_loss") is not None else trade.get("net_pnl"),
                    float(trade.get("commission") or 0), float(trade.get("swap") or 0),
                    trade.get("duration_seconds"), (trade.get("status") or "OPEN").upper(),
                    trade.get("entry_reason"), trade.get("exit_reason"), trade.get("decision_id"), Json(trade),
                ),
            )
            row = cursor.fetchone()
            if row and (trade.get("status") or "").upper() == "CLOSED":
                cursor.execute(
                    """
                    UPDATE public.bot_decisions
                    SET trade_result = %s, status = 'CLOSED'
                    WHERE id = (
                        SELECT decision_id FROM public.mt5_trade_history WHERE id = %s
                    )
                       OR trade_ticket = ANY(%s)
                    """,
                    (
                        trade.get("profit_loss") if trade.get("profit_loss") is not None else trade.get("net_pnl"),
                        int(row[0]),
                        [value for value in (ticket, trade.get("position_id"), trade.get("order_ticket")) if value],
                    ),
                )
        connection.commit()
        return int(row[0])
    finally:
        if own_conn:
            connection.close()


def _monitoring_filters(symbol=None, start_date=None, end_date=None, side=None,
                        performance=None, status=None):
    clauses: List[str] = ["deleted_at IS NULL"]
    params: List[Any] = []
    if symbol:
        clauses.append("symbol = %s")
        params.append(symbol)
    if start_date:
        clauses.append("opened_at >= %s")
        params.append(start_date)
    if end_date:
        clauses.append("opened_at < (%s::date + INTERVAL '1 day')")
        params.append(end_date)
    if side:
        clauses.append("side = %s")
        params.append(side.upper())
    if status:
        clauses.append("status = %s")
        params.append(status.upper())
    if performance == "PROFITABLE":
        clauses.append("profit_loss > 0")
    elif performance == "LOSS":
        clauses.append("profit_loss < 0")
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    return where, params


def list_mt5_trades(limit: int = 1000, offset: int = 0, symbol=None,
                    start_date=None, end_date=None, side=None, performance=None,
                    status=None, connection=None) -> List[RealDictRow]:
    ensure_monitoring_tables(connection)
    own_conn = connection is None
    if own_conn:
        connection = get_connection()
    where, params = _monitoring_filters(symbol, start_date, end_date, side, performance, status)
    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                f"SELECT * FROM public.mt5_trade_history {where} ORDER BY opened_at DESC, id DESC LIMIT %s OFFSET %s",
                tuple(params + [limit, offset]),
            )
            return cursor.fetchall()
    finally:
        if own_conn:
            connection.close()


def delete_mt5_trades(trade_ids, connection=None) -> List[int]:
    """Hide MT5 history rows permanently, including after broker re-sync."""
    ensure_monitoring_tables(connection)
    ids = sorted({int(value) for value in (trade_ids or []) if int(value) > 0})
    if not ids:
        return []
    own_conn = connection is None
    if own_conn:
        connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE public.mt5_trade_history
                SET deleted_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ANY(%s)
                  AND deleted_at IS NULL
                  AND COALESCE(status, '') <> 'OPEN'
                RETURNING id
                """,
                (ids,),
            )
            deleted = [int(row[0]) for row in cursor.fetchall()]
        connection.commit()
        return deleted
    finally:
        if own_conn:
            connection.close()


def list_bot_decisions(limit: int = 250, action=None, symbol=None, connection=None) -> List[RealDictRow]:
    ensure_monitoring_tables(connection)
    own_conn = connection is None
    if own_conn:
        connection = get_connection()
    clauses, params = [], []
    if action:
        clauses.append("action = %s")
        params.append(action.upper().replace(" ", "_"))
    if symbol:
        clauses.append("symbol = %s")
        params.append(symbol)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                f"SELECT * FROM public.bot_decisions {where} ORDER BY occurred_at DESC, id DESC LIMIT %s",
                tuple(params + [limit]),
            )
            return cursor.fetchall()
    finally:
        if own_conn:
            connection.close()


def list_bot_events(limit: int = 250, severity=None, event_type=None, connection=None) -> List[RealDictRow]:
    ensure_monitoring_tables(connection)
    own_conn = connection is None
    if own_conn:
        connection = get_connection()
    clauses, params = [], []
    if severity:
        clauses.append("severity = %s")
        params.append(severity.upper())
    if event_type:
        clauses.append("event_type = %s")
        params.append(event_type.upper())
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                f"SELECT * FROM public.bot_activity_log {where} ORDER BY occurred_at DESC, id DESC LIMIT %s",
                tuple(params + [limit]),
            )
            return cursor.fetchall()
    finally:
        if own_conn:
            connection.close()


def list_metric_snapshots(limit: int = 1440, connection=None) -> List[RealDictRow]:
    ensure_monitoring_tables(connection)
    own_conn = connection is None
    if own_conn:
        connection = get_connection()
    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT * FROM public.bot_metric_snapshots ORDER BY captured_at DESC, id DESC LIMIT %s",
                (limit,),
            )
            return cursor.fetchall()
    finally:
        if own_conn:
            connection.close()


def list_mt5_connection_events(limit: int = 100, connection=None) -> List[RealDictRow]:
    ensure_monitoring_tables(connection)
    own_conn = connection is None
    if own_conn:
        connection = get_connection()
    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT * FROM public.mt5_connection_events ORDER BY occurred_at DESC, id DESC LIMIT %s",
                (limit,),
            )
            return cursor.fetchall()
    finally:
        if own_conn:
            connection.close()
