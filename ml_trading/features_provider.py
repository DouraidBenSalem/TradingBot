"""Module d'accès sécurisé et validé aux indicateurs pré-calculés de eurusd_features.

Ce module fournit une couche d'abstraction entre la stratégie EUR/USD et la table
`public.eurusd_features` qui stocke l'ensemble des indicateurs techniques pré-calculés.

Aucun calcul d'indicateur n'est effectué ici : les valeurs sont lues directement
depuis la table, avec des contrôles de cohérence, de fraîcheur et de complétude.


Architecture de connexion — Stratégie EUR/USD ↔ Table eurusd_features
======================================================================

Flux de données (sans recalcul d'indicateurs dans la stratégie) :

  ┌────────────────────────────────┐
  │  app/data/feature_engineering  │  ← Génère et maintient la table
  │  .sync_features()              │    eurusd_features à partir d'OHLC
  └──────────────┬─────────────────┘    (retours, EMA, RSI, MACD, ATR, …)
                 │ écrit
                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  PostgreSQL : public.eurusd_features                             │
  │  ─────────────────────────────────────────────────────────────  │
  │  • Colonnes OHLCV+ : time, open, high, low, close, tick_volume,  │
  │    spread, return_1m/5m/15m, ema_9/21/50, rsi_14, atr_14,       │
  │    macd, macd_signal, macd_hist, volatility_20, …, trend        │
  │  • Clé primaire : time (1 bougie = 1 ligne, 34 indicateurs)      │
  └────────────────────────────┬────────────────────────────────────┘
                               │ lecture directe
                               ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  ml_trading/features_provider.py  ← CE MODULE                    │
  │  ─────────────────────────────────────────────────────────────  │
  │  • fetch_features_range(start, end)       → backtest             │
  │  • fetch_latest_features(N)               → bot temps réel       │
  │  • validate_features() / health_check()   → contrôles qualité    │
  │  • _map_session() : 'london'→1 / 'new_york'→2 / 'other'→0       │
  └──────────────┬───────────────────────────────────────┬───────────┘
                 │ utilisé par                            │ utilisé par
                 ▼                                        ▼
  ┌────────────────────────────────┐    ┌────────────────────────────────┐
  │  backtest_technical.py         │    │  technical_trading_bot.py      │
  │  ────────────────────────────  │    │  ────────────────────────────  │
  │  • Plus de _read_source() +    │    │  • Plus de _get_latest_data()  │
  │    add_technical_indicators()  │    │    depuis eurusd_m1            │
  │  • fetch_features_range() →    │    │  • Plus de add_technical_      │
  │    DataFrame prêt pour la      │    │    indicators()                │
  │    stratégie (score/signaux)   │    │  • fetch_latest_features() →   │
  └──────────────┬─────────────────┘    │    DataFrame prêt              │
                 │                       └──────────────┬─────────────────┘
                 │                                      │
                 ▼                                      ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  ml_trading/technical_strategy.py : TrendFollowingStrategy        │
  │  ──────────────────────────────────────────────────────────────  │
  │  • Reçoit directement les colonnes pré-calculées :                │
  │    ema_9, ema_21, ema_50, rsi_14, macd_hist, atr_14, hour,       │
  │    session (0/1/2), open, high, low, close, spread, macd         │
  │  • AUCUN appel à TechnicalIndicators.ema() / .rsi() / .macd()…   │
  │    → la stratégie n'effectue plus aucun calcul d'indicateur      │
  └──────────────────────────────────────────────────────────────────┘

Règles de validation (cf. FEATURES_VALIDATION dans config.py) :
  • max_null_ratio       : tolérance max de valeurs NULL (<1% par défaut)
  • max_freshness_minutes: fraîcheur max pour usage temps réel (24h)
  • value_ranges         : plages réalistes par indicateur
                           ex. RSI ∈ [0,100], ATR > 0, hour ∈ [0,23]…
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from Database.postgresql import get_connection
from ml_trading.config import FEATURES_TABLE, FEATURES_REQUIRED_COLUMNS, FEATURES_VALIDATION


SESSION_TEXT_TO_INT = {
    "london": 1,
    "new_york": 2,
    "other": 0,
}


@dataclass
class ValidationReport:
    """Rapport de validation des données récupérées."""
    total_rows: int
    rows_without_nulls: int
    null_ratio: float
    latest_timestamp: Optional[datetime]
    freshness_minutes: Optional[float]
    value_range_violations: dict
    duplicate_timestamps: int
    is_valid: bool

    def to_dict(self):
        return {
            "total_rows": self.total_rows,
            "rows_without_nulls": self.rows_without_nulls,
            "null_ratio": round(self.null_ratio, 4),
            "latest_timestamp": str(self.latest_timestamp) if self.latest_timestamp else None,
            "freshness_minutes": round(self.freshness_minutes, 1) if self.freshness_minutes else None,
            "value_range_violations": self.value_range_violations,
            "duplicate_timestamps": self.duplicate_timestamps,
            "is_valid": self.is_valid,
        }


def _select_columns() -> str:
    return ", ".join(FEATURES_REQUIRED_COLUMNS)


def _map_session(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit la colonne `session` texte -> entier compatible stratégie."""
    if "session" in df.columns and not df.empty:
        if pd.api.types.is_string_dtype(df["session"]) or df["session"].dtype == object:
            df["session"] = (
                df["session"].astype("string").str.lower().map(SESSION_TEXT_TO_INT).fillna(0).astype(int)
            )
    return df


def _validate_ranges(df: pd.DataFrame) -> dict:
    """Vérifie que les valeurs des indicateurs sont dans des plages réalistes."""
    violations = {}
    bounds = FEATURES_VALIDATION.get("value_ranges", {})
    for column, (min_val, max_val) in bounds.items():
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        bad = (series < min_val) | (series > max_val)
        n_bad = int(bad.sum())
        if n_bad > 0:
            violations[column] = {
                "count": n_bad,
                "min_allowed": min_val,
                "max_allowed": max_val,
                "actual_min": float(series.min()) if series.notna().any() else None,
                "actual_max": float(series.max()) if series.notna().any() else None,
            }
    return violations


def validate_features(df: pd.DataFrame, freshness_check: bool = True) -> ValidationReport:
    """Valide l'intégralité d'un DataFrame de features récupéré depuis la table.

    Args:
        df: DataFrame contenant les colonnes de eurusd_features.
        freshness_check: Si True, vérifie que les données sont récentes.

    Returns:
        ValidationReport avec les statistiques de contrôle.
    """
    total_rows = len(df)
    rows_without_nulls = int(df.dropna(subset=FEATURES_REQUIRED_COLUMNS).shape[0]) if total_rows > 0 else 0
    null_ratio = (1.0 - (rows_without_nulls / total_rows)) if total_rows > 0 else 0.0

    latest_timestamp = None
    freshness_minutes = None
    if total_rows > 0 and "time" in df.columns:
        latest_timestamp = pd.to_datetime(df["time"]).max().to_pydatetime()
        if freshness_check:
            now = datetime.now()
            freshness_minutes = (now - latest_timestamp).total_seconds() / 60.0

    value_range_violations = _validate_ranges(df) if total_rows > 0 else {}

    duplicate_timestamps = 0
    if total_rows > 0 and "time" in df.columns:
        duplicate_timestamps = int(df["time"].duplicated().sum())

    is_valid = (
        total_rows > 0
        and null_ratio <= FEATURES_VALIDATION.get("max_null_ratio", 0.01)
        and not value_range_violations
        and duplicate_timestamps == 0
    )

    if freshness_check and freshness_minutes is not None:
        max_freshness = FEATURES_VALIDATION.get("max_freshness_minutes", 1440)
        is_valid = is_valid and (freshness_minutes <= max_freshness)

    return ValidationReport(
        total_rows=total_rows,
        rows_without_nulls=rows_without_nulls,
        null_ratio=null_ratio,
        latest_timestamp=latest_timestamp,
        freshness_minutes=freshness_minutes,
        value_range_violations=value_range_violations,
        duplicate_timestamps=duplicate_timestamps,
        is_valid=is_valid,
    )


def fetch_features_range(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    validate: bool = True,
    raise_on_invalid: bool = False,
    freshness_check: bool = True,
) -> tuple[pd.DataFrame, Optional[ValidationReport]]:
    """Récupère les features pré-calculées pour une plage de dates.

    Les indicateurs sont chargés directement depuis la table `eurusd_features`
    sans aucun recalcul côté application.

    Args:
        start_time: Début de la plage (inclusif). Si None, pas de limite basse.
        end_time: Fin de la plage (inclusif). Si None, pas de limite haute.
        validate: Si True, exécute la validation après chargement.
        raise_on_invalid: Si True, lève une exception si la validation échoue.
        freshness_check: Si True (défaut), vérifie que les données sont
            récentes. Mettre False pour les backtests sur données historiques.

    Returns:
        Tuple (DataFrame avec toutes les features, ValidationReport ou None)
    """
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {_select_columns()}
                FROM {FEATURES_TABLE}
                WHERE (%s IS NULL OR time >= %s)
                  AND (%s IS NULL OR time <= %s)
                ORDER BY time ASC
                """,
                (start_time, start_time, end_time, end_time),
            )
            rows = cursor.fetchall()
            columns = [column.name for column in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
    finally:
        connection.close()

    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
        numeric_cols = [c for c in FEATURES_REQUIRED_COLUMNS if c not in ("time", "session")]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = _map_session(df)

    report = None
    if validate:
        report = validate_features(df, freshness_check=freshness_check)
        if raise_on_invalid and not report.is_valid:
            raise ValueError(
                f"Validation des features échouée: {report.to_dict()}"
            )

    return df, report


def fetch_latest_features(
    lookback_candles: int = 100,
    validate: bool = True,
    raise_on_invalid: bool = False,
    freshness_check: bool = True,
) -> tuple[pd.DataFrame, Optional[ValidationReport]]:
    """Récupère les N dernières bougies avec leurs features pré-calculées.

    Destiné au fonctionnement temps réel du bot de trading.

    Args:
        lookback_candles: Nombre de bougies à remonter dans le passé.
        validate: Si True, exécute la validation après chargement.
        raise_on_invalid: Si True, lève une exception si la validation échoue.
        freshness_check: Si True (défaut), vérifie la fraîcheur des données.
            Mettre False pour les tests / usages historiques.

    Returns:
        Tuple (DataFrame avec les N dernières bougies, ValidationReport ou None)
    """
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {_select_columns()}
                FROM {FEATURES_TABLE}
                ORDER BY time DESC
                LIMIT %s
                """,
                (lookback_candles,),
            )
            rows = cursor.fetchall()
            columns = [column.name for column in cursor.description]
        df = pd.DataFrame(list(reversed(rows)), columns=columns)
    finally:
        connection.close()

    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
        numeric_cols = [c for c in FEATURES_REQUIRED_COLUMNS if c not in ("time", "session")]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = _map_session(df)

    report = None
    if validate:
        report = validate_features(df, freshness_check=freshness_check)
        if raise_on_invalid and not report.is_valid:
            raise ValueError(
                f"Validation des features échouée: {report.to_dict()}"
            )

    return df, report


def get_latest_feature_timestamp() -> Optional[datetime]:
    """Retourne le timestamp le plus récent présent dans la table de features."""
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT MAX(time) FROM {FEATURES_TABLE}")
            result = cursor.fetchone()[0]
        return result
    finally:
        connection.close()


def health_check(freshness_check: bool = True) -> dict:
    """Diagnostic rapide de l'état de la table de features.

    Args:
        freshness_check: Si True, le flag `ok` prend en compte la fraîcheur
            des données. Mettre False pour un diagnostic historique.

    Returns:
        Dict avec les métriques de santé (nombre de lignes, fraîcheur, validation).
    """
    latest_ts = get_latest_feature_timestamp()
    freshness_minutes = None
    if latest_ts is not None:
        freshness_minutes = (datetime.now() - latest_ts).total_seconds() / 60.0

    df, report = fetch_latest_features(
        lookback_candles=100,
        validate=True,
        raise_on_invalid=False,
        freshness_check=freshness_check,
    )

    return {
        "features_table": FEATURES_TABLE,
        "latest_timestamp": str(latest_ts) if latest_ts else None,
        "freshness_minutes": round(freshness_minutes, 1) if freshness_minutes is not None else None,
        "sample_size": len(df),
        "validation": report.to_dict() if report else None,
        "ok": report is not None and report.is_valid,
    }
