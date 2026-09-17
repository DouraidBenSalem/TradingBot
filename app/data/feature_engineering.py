import pandas as pd
from psycopg2.extras import execute_values

from ml_trading.technical_indicators import add_technical_indicators


FEATURE_COLUMNS = [
    "time", "open", "high", "low", "close", "tick_volume", "spread",
    "return_1m", "return_5m", "return_15m", "ema_9", "ema_21", "ema_50",
    "rsi_14", "atr_14", "macd", "macd_signal", "macd_hist", "volatility_20",
    "high_low_range", "body_size", "upper_wick", "lower_wick",
    "candle_direction", "volume_ma_20", "volume_ratio", "hour", "minute",
    "day_of_week", "session", "ema9_distance", "ema21_distance",
    "ema50_distance", "trend",
]


INSERT_SQL = """
INSERT INTO public.eurusd_features (
    time, open, high, low, close, tick_volume, spread,
    return_1m, return_5m, return_15m, ema_9, ema_21, ema_50,
    rsi_14, atr_14, macd, macd_signal, macd_hist, volatility_20,
    high_low_range, body_size, upper_wick, lower_wick,
    candle_direction, volume_ma_20, volume_ratio, hour, minute,
    day_of_week, session, ema9_distance, ema21_distance, ema50_distance, trend
) VALUES %s
ON CONFLICT (time) DO UPDATE SET
    open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
    close = EXCLUDED.close, tick_volume = EXCLUDED.tick_volume,
    spread = EXCLUDED.spread, return_1m = EXCLUDED.return_1m,
    return_5m = EXCLUDED.return_5m, return_15m = EXCLUDED.return_15m,
    ema_9 = EXCLUDED.ema_9, ema_21 = EXCLUDED.ema_21, ema_50 = EXCLUDED.ema_50,
    rsi_14 = EXCLUDED.rsi_14, atr_14 = EXCLUDED.atr_14,
    macd = EXCLUDED.macd, macd_signal = EXCLUDED.macd_signal,
    macd_hist = EXCLUDED.macd_hist, volatility_20 = EXCLUDED.volatility_20,
    high_low_range = EXCLUDED.high_low_range, body_size = EXCLUDED.body_size,
    upper_wick = EXCLUDED.upper_wick, lower_wick = EXCLUDED.lower_wick,
    candle_direction = EXCLUDED.candle_direction,
    volume_ma_20 = EXCLUDED.volume_ma_20, volume_ratio = EXCLUDED.volume_ratio,
    hour = EXCLUDED.hour, minute = EXCLUDED.minute,
    day_of_week = EXCLUDED.day_of_week, session = EXCLUDED.session,
    ema9_distance = EXCLUDED.ema9_distance, ema21_distance = EXCLUDED.ema21_distance,
    ema50_distance = EXCLUDED.ema50_distance, trend = EXCLUDED.trend
"""


def _ensure_features_table(connection):
    with connection.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS public.eurusd_features (
                time TIMESTAMP PRIMARY KEY,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                tick_volume BIGINT,
                spread DOUBLE PRECISION,
                return_1m DOUBLE PRECISION,
                return_5m DOUBLE PRECISION,
                return_15m DOUBLE PRECISION,
                ema_9 DOUBLE PRECISION,
                ema_21 DOUBLE PRECISION,
                ema_50 DOUBLE PRECISION,
                rsi_14 DOUBLE PRECISION,
                atr_14 DOUBLE PRECISION,
                macd DOUBLE PRECISION,
                macd_signal DOUBLE PRECISION,
                macd_hist DOUBLE PRECISION,
                volatility_20 DOUBLE PRECISION,
                high_low_range DOUBLE PRECISION,
                body_size DOUBLE PRECISION,
                upper_wick DOUBLE PRECISION,
                lower_wick DOUBLE PRECISION,
                candle_direction INTEGER,
                volume_ma_20 DOUBLE PRECISION,
                volume_ratio DOUBLE PRECISION,
                hour INTEGER,
                minute INTEGER,
                day_of_week INTEGER,
                session TEXT NOT NULL,
                ema9_distance DOUBLE PRECISION,
                ema21_distance DOUBLE PRECISION,
                ema50_distance DOUBLE PRECISION,
                trend INTEGER
            )
        """)
    connection.commit()


def _calculate_features(source):
    frame = source.copy()
    frame["time"] = pd.to_datetime(frame["time"])
    frame = frame.sort_values("time").reset_index(drop=True)
    frame["return_1m"] = frame["close"].pct_change()
    frame["return_5m"] = frame["close"].pct_change(5)
    frame["return_15m"] = frame["close"].pct_change(15)
    frame = add_technical_indicators(frame)
    frame["volatility_20"] = frame["return_1m"].rolling(20).std()
    frame["high_low_range"] = frame["high"] - frame["low"]
    frame["body_size"] = (frame["close"] - frame["open"]).abs()
    frame["upper_wick"] = frame["high"] - frame[["open", "close"]].max(axis=1)
    frame["lower_wick"] = frame[["open", "close"]].min(axis=1) - frame["low"]
    frame["candle_direction"] = (frame["close"] > frame["open"]).astype(int) - (frame["close"] < frame["open"]).astype(int)
    frame["volume_ma_20"] = frame["tick_volume"].rolling(20).mean()
    frame["volume_ratio"] = frame["tick_volume"] / frame["volume_ma_20"].replace(0, pd.NA)
    timestamps = pd.to_datetime(frame["time"])
    frame["hour"] = timestamps.dt.hour
    frame["minute"] = timestamps.dt.minute
    frame["day_of_week"] = timestamps.dt.dayofweek
    # The overlap is assigned to New York so the single categorical field is
    # unambiguous.  Both labels remain eligible execution sessions.
    frame["session"] = timestamps.dt.hour.map(
        lambda hour: "new_york" if 13 <= hour < 22 else "london" if 8 <= hour < 13 else "other"
    )
    frame["ema9_distance"] = (frame["close"] - frame["ema_9"]) / frame["ema_9"]
    frame["ema21_distance"] = (frame["close"] - frame["ema_21"]) / frame["ema_21"]
    frame["ema50_distance"] = (frame["close"] - frame["ema_50"]) / frame["ema_50"]
    frame["trend"] = 0
    frame.loc[(frame["ema_9"] > frame["ema_21"]) & (frame["ema_21"] > frame["ema_50"]), "trend"] = 1
    frame.loc[(frame["ema_9"] < frame["ema_21"]) & (frame["ema_21"] < frame["ema_50"]), "trend"] = -1
    # Keep the warm-up rows as NaN rather than importing future information.
    # Signal generation rejects incomplete rows, which is the only causal
    # treatment of an indicator warm-up period.
    frame = frame.ffill()
    return frame[FEATURE_COLUMNS]


def sync_features(times=None):
    from Database.postgresql import get_connection

    connection = get_connection()
    try:
        _ensure_features_table(connection)
        with connection.cursor() as cursor:
            # EMA and MACD are recursive.  Recomputing a short trailing
            # window resets them and makes live features differ from the
            # historical backtest.  Read the causal history so each updated
            # timestamp is calculated from exactly the same preceding bars.
            cursor.execute(
                """
                SELECT time, open, high, low, close, tick_volume, spread
                FROM public.eurusd_m1
                ORDER BY time
                """
            )
            source = cursor.fetchall()
            columns = [column.name for column in cursor.description]

        if not source:
            return 0

        features = _calculate_features(pd.DataFrame(source, columns=columns))
        if times:
            wanted = set(pd.to_datetime(list(times)))
            features = features[features["time"].isin(wanted)]

        rows = [tuple(row) for row in features.itertuples(index=False, name=None)]
        if not rows:
            return 0
        with connection.cursor() as cursor:
            execute_values(cursor, INSERT_SQL, rows, page_size=500)
        connection.commit()
        return len(rows)
    finally:
        connection.close()
