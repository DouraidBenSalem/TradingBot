"""Module d'indicateurs techniques pour la stratégie EUR/USD M1."""

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """Calcul des indicateurs techniques sur des données OHLCV."""

    @staticmethod
    def ema(data, period):
        """Calculer l'EMA (Exponential Moving Average)."""
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(data, period=14):
        """Calculer le RSI (Relative Strength Index)."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """Calculer le MACD (Moving Average Convergence Divergence)."""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def atr(high, low, close, period=14):
        """Calculer l'ATR (Average True Range)."""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    @staticmethod
    def session_indicator(hour, minute):
        """
        Déterminer la session de trading (Londres/New York).
        Retourne 1 si Londres, 2 si New York, 0 sinon.
        """
        # New York is intentionally checked first in the overlap.  ``session``
        # is a single categorical value, while both sessions remain tradable.
        if 13 <= hour < 22:
            return 2  # New York
        if 8 <= hour < 13:
            return 1  # Londres
        return 0

    @staticmethod
    def is_high_volatility(atr, close, atr_threshold=0.0008):
        """
        Vérifier si l'ATR indique une volatilité suffisante.
        atr_threshold : ATR/Close ratio minimum (par défaut 0.08% de volatilité)
        """
        if pd.isna(atr) or pd.isna(close) or close == 0 or atr <= 0:
            return False
        volatility_ratio = atr / close
        return volatility_ratio >= atr_threshold

    @staticmethod
    def is_pullback_to_ema(close, ema_9, ema_21, direction="both"):
        """
        Déterminer si le prix effectue un pullback vers la zone EMA 9/21.
        
        direction : 'both', 'up', 'down'
        - 'up' : pullback en tendance haussière (prix doit toucher zone EMA 9/21 par le bas)
        - 'down' : pullback en tendance baissière (prix doit toucher zone EMA 9/21 par le haut)
        - 'both' : retourne True si prix est dans la zone 20% entre EMA 9 et EMA 21
        """
        if pd.isna(close) or pd.isna(ema_9) or pd.isna(ema_21):
            return False
        
        ema_min = min(ema_9, ema_21)
        ema_max = max(ema_9, ema_21)
        ema_range = abs(ema_9 - ema_21)
        
        # Zone de pullback : ±20% autour de la zone EMA 9/21
        lower_bound = ema_min - ema_range * 0.2
        upper_bound = ema_max + ema_range * 0.2
        
        return lower_bound <= close <= upper_bound

    @staticmethod
    def candle_confirmation(
        open_price,
        close,
        high,
        low,
        direction="up",
        min_body_ratio=0.5,
        min_close_position=None,
    ):
        """
        Vérifier la confirmation de bougie.
        
        direction : 'up' (bullish) ou 'down' (bearish)
        - Bullish : close > open et corps de bougie suffisamment grand.
        - Bearish : open > close et corps de bougie suffisamment grand.

        ``min_close_position`` est optionnel pour conserver la compatibilite
        avec les anciens appels. S'il est defini, la cloture doit aussi etre
        situee dans la partie haute (BUY) ou basse (SELL) de la bougie. Cela
        permet de distinguer un rejet du pullback d'une simple bougie verte ou
        rouge peu convaincante.
        """
        body = abs(close - open_price)
        range_val = high - low
        
        if range_val == 0:
            return False
        
        body_ratio = body / range_val
        
        close_position = (close - low) / range_val
        if direction == "up":
            return (
                close > open_price
                and body_ratio >= min_body_ratio
                and (min_close_position is None or close_position >= min_close_position)
            )
        elif direction == "down":
            return (
                open_price > close
                and body_ratio >= min_body_ratio
                and (min_close_position is None or close_position <= (1 - min_close_position))
            )
        return False

    @staticmethod
    def momentum_confirms_direction(rsi, macd_hist, direction="up"):
        """
        Vérifier la confirmation du momentum.
        
        direction : 'up' ou 'down'
        """
        if direction == "up":
            return rsi > 50 and macd_hist > 0
        elif direction == "down":
            return rsi < 50 and macd_hist < 0
        return False


def add_technical_indicators(data):
    """
    Ajouter tous les indicateurs techniques au DataFrame.
    
    Args:
        data : DataFrame avec colonnes 'high', 'low', 'close', 'time'
    
    Returns:
        DataFrame enrichi avec tous les indicateurs
    """
    df = data.copy()
    
    # EMAs
    df["ema_9"] = TechnicalIndicators.ema(df["close"], 9)
    df["ema_21"] = TechnicalIndicators.ema(df["close"], 21)
    df["ema_50"] = TechnicalIndicators.ema(df["close"], 50)
    
    # RSI
    df["rsi_14"] = TechnicalIndicators.rsi(df["close"], 14)
    
    # MACD
    df["macd"], df["macd_signal"], df["macd_hist"] = TechnicalIndicators.macd(df["close"])
    
    # ATR
    df["atr_14"] = TechnicalIndicators.atr(df["high"], df["low"], df["close"], 14)
    
    # Session (basée sur l'heure du timestamp)
    df["hour"] = pd.to_datetime(df["time"]).dt.hour
    df["minute"] = pd.to_datetime(df["time"]).dt.minute
    df["session"] = df.apply(
        lambda row: TechnicalIndicators.session_indicator(row["hour"], row["minute"]),
        axis=1
    )
    
    # Never back-fill indicator warm-up values: doing so would copy a value
    # calculated with future candles into an earlier row.  A strategy must
    # simply abstain until it has enough completed history.
    df = df.ffill()
    
    return df
