import time
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5

from app.data.mt5_data import (
    connect_mt5,
    disconnect_mt5,
    get_market_data,
    get_market_data_range
)

from Database.postgresql import (
    insert_market_data,
    get_last_timestamp,
    get_missing_ranges
)


SYMBOL = "EURUSD.m"
TIMEFRAME = mt5.TIMEFRAME_M1
TABLE_NAME = "eurusd_m1"
CHUNK_DAYS = 1
TIMEFRAME_MINUTES = 1


def detect_gap():

    last_db_time = get_last_timestamp(TABLE_NAME)
    market_data = get_market_data(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        bars=2
    )

    if len(market_data) < 2:
        raise RuntimeError(
            "MT5 n'a pas fourni assez de bougies clôturées pour détecter les gaps."
        )

    latest_closed_time = market_data.iloc[-2]["time"]

    print(f"Dernière donnée en base : {last_db_time}")
    print(f"Dernière bougie clôturée MT5 : {latest_closed_time}")

    if last_db_time is None:

        print("Aucune donnée en base. Démarrage du backfill complet.")
        return None, latest_closed_time

    db_time = last_db_time
    if db_time.tzinfo is None:
        db_time = db_time.replace(tzinfo=latest_closed_time.tzinfo)

    if db_time >= latest_closed_time:

        print("Aucun gap détecté.")
        return last_db_time, last_db_time

    print(f"Gap détecté : {last_db_time} -> {latest_closed_time}")

    return db_time, latest_closed_time


def _backfill_range(start, end):

    current_start = start

    total_inserted = 0

    while current_start < end:

        current_end = min(
            current_start + timedelta(days=CHUNK_DAYS),
            end
        )

        print(f"Backfill : {current_start} -> {current_end}")

        request_start_value = current_start - timedelta(minutes=5)
        request_end_value = current_end + timedelta(minutes=5)
        request_start = request_start_value.replace(tzinfo=timezone.utc) if request_start_value.tzinfo is None else request_start_value.astimezone(timezone.utc)
        request_end = request_end_value.replace(tzinfo=timezone.utc) if request_end_value.tzinfo is None else request_end_value.astimezone(timezone.utc)

        df = get_market_data_range(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            datetime_from=request_start,
            datetime_to=request_end
        )

        if not df.empty:
            start_naive = current_start.replace(tzinfo=None)
            end_naive = current_end.replace(tzinfo=None)
            df = df[(df["time"] >= start_naive) & (df["time"] < end_naive)]
            insert_market_data(df, TABLE_NAME)
            total_inserted += len(df)
            print(f"Bougies MT5 retenues : {len(df)}")
        else:
            print("Aucune bougie retournée par MT5 pour cette plage.")

        current_start = current_end
        time.sleep(0.5)

    return total_inserted


def _backfill_history(end):

    current_end = end

    while True:

        current_start = current_end - timedelta(days=CHUNK_DAYS)

        print(f"Backfill historique : {current_start} -> {current_end}")

        df = get_market_data_range(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            datetime_from=current_start,
            datetime_to=current_end
        )

        if df.empty:
            break

        insert_market_data(df, TABLE_NAME)
        current_end = current_start
        time.sleep(0.5)


def recover_missing_gaps(detected_gaps=None):

    if detected_gaps is None:
        missing_ranges = get_missing_ranges(
            TABLE_NAME,
            interval_minutes=TIMEFRAME_MINUTES
        )
        ranges = [
            (missing_start, missing_end + timedelta(minutes=1))
            for missing_start, missing_end in missing_ranges
        ]
    else:
        ranges = [
            (
                gap["start"] + timedelta(minutes=TIMEFRAME_MINUTES),
                gap["end"] + timedelta(minutes=TIMEFRAME_MINUTES)
            )
            for gap in detected_gaps
        ]

    inserted_count = 0
    for missing_start, missing_end in ranges:
        print(
            f"Gap à récupérer : {missing_start} -> {missing_end}"
        )
        inserted_count += _backfill_range(
            missing_start,
            missing_end
        )

    remaining_gaps = get_missing_ranges(
        TABLE_NAME,
        interval_minutes=TIMEFRAME_MINUTES
    )
    if remaining_gaps:
        print(f"Attention : {len(remaining_gaps)} gap(s) restent après le backfill.")
        for gap_start, gap_end in remaining_gaps:
            print(f"Gap non récupéré : {gap_start} -> {gap_end}")
    else:
        print("Vérification : tous les gaps intraday ont été récupérés.")

    print(f"Total de bougies insérées : {inserted_count}")

    return len(ranges)


def backfill(manage_connection=True, detected_gaps=None):

    if manage_connection:
        connect_mt5()

    try:

        start, end = detect_gap()

        if start is None:
            _backfill_history(end)
        else:
            missing_gap_count = recover_missing_gaps(detected_gaps)

            if start < end:
                _backfill_range(
                    start + timedelta(minutes=TIMEFRAME_MINUTES),
                    end + timedelta(minutes=TIMEFRAME_MINUTES)
                )

            if missing_gap_count or start < end:
                print("Backfill terminé.")
            else:
                print("Aucun backfill nécessaire.")

    finally:

        if manage_connection:
            disconnect_mt5()


if __name__ == "__main__":

    backfill()
