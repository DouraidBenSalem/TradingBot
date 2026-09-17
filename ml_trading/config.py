"""Paramètres partagés de la stratégie technique EUR/USD."""

SOURCE_TABLE = "public.eurusd_m1"
MAX_TRADES_PER_DAY = 3
HORIZON_BARS = 30
SL_ATR_MULTIPLIER = 1.0
TP_ATR_MULTIPLIER = 2.0

FEATURES_TABLE = "public.eurusd_features"

FEATURES_REQUIRED_COLUMNS = [
    "time", "open", "high", "low", "close", "tick_volume", "spread",
    "return_1m", "return_5m", "return_15m", "ema_9", "ema_21", "ema_50",
    "rsi_14", "atr_14", "macd", "macd_signal", "macd_hist", "volatility_20",
    "high_low_range", "body_size", "upper_wick", "lower_wick",
    "candle_direction", "volume_ma_20", "volume_ratio", "hour", "minute",
    "day_of_week", "session", "ema9_distance", "ema21_distance",
    "ema50_distance", "trend",
]

FEATURES_VALIDATION = {
    "max_null_ratio": 0.01,
    "max_freshness_minutes": 1440,
    "value_ranges": {
        "rsi_14": (0.0, 100.0),
        "atr_14": (1e-7, 1.0),
        "volatility_20": (0.0, 1.0),
        "candle_direction": (-1, 1),
        "trend": (-1, 1),
        "hour": (0, 23),
        "minute": (0, 59),
        "day_of_week": (0, 6),
        "return_1m": (-0.1, 0.1),
        "return_5m": (-0.2, 0.2),
        "return_15m": (-0.3, 0.3),
    },
}
