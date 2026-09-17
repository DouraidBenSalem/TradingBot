import time
import threading
from datetime import datetime

import MetaTrader5 as mt5

from app.data.mt5_data import (
    connect_mt5,
    disconnect_mt5,
    get_market_data
)

from Database.postgresql import (
    get_last_timestamp,
    insert_market_data,
    record_bot_event,
    record_mt5_snapshot,
)

from app.data.backfill import backfill
from ml_trading.pipeline import initialize, on_new_candle


SYMBOL = "EURUSD.m"

TIMEFRAME = mt5.TIMEFRAME_M1

TABLE_NAME = "eurusd_m1"
MAX_RECONNECT_DELAY_SECONDS = 30


def _connect_with_retry():
    """Keep the collector alive while MT5 is closed or unavailable."""
    delay = 5
    previous_error = None
    while True:
        try:
            connect_mt5()
            try:
                record_bot_event(
                    "MT5_CONNECTED",
                    "MT5 connected; real-time collection started.",
                    component="MT5",
                )
            except Exception:
                pass
            return
        except Exception as error:
            message = str(error)
            if message != previous_error:
                print(f"MT5 indisponible : {message}. Nouvelle tentative dans {delay}s.")
                try:
                    record_mt5_snapshot({
                        "connection_status": "RECONNECTING",
                        "terminal_open": False,
                        "connected": False,
                        "account_available": False,
                        "connection_error": message,
                    })
                    record_bot_event(
                        "MT5_CONNECTION_ERROR",
                        message,
                        severity="ERROR",
                        component="MT5",
                    )
                except Exception:
                    pass
                previous_error = message
            time.sleep(delay)
            delay = min(delay * 2, MAX_RECONNECT_DELAY_SECONDS)


def collect_realtime(detected_gaps=None, run_initialization=True):

    _connect_with_retry()

    print("================================")
    print("REAL-TIME COLLECTOR")
    print("================================")

    try:

        last_saved_time = get_last_timestamp(TABLE_NAME)
        if run_initialization:
            initialize()

        if last_saved_time is None:
            print("Aucune donnée en base. Lancement du backfill complet...")
        else:
            print("Recherche des gaps à récupérer...")

        backfill(
            manage_connection=False,
            detected_gaps=detected_gaps
        )
        last_saved_time = get_last_timestamp(TABLE_NAME)

        print(
            f"Dernière bougie connue : "
            f"{last_saved_time}"
        )

        while True:

            df = get_market_data(
                symbol=SYMBOL,
                timeframe=TIMEFRAME,
                bars=3
            )

            if len(df) < 2:

                time.sleep(1)

                continue

            # Avant-dernière bougie = dernière bougie clôturée
            closed_candle = df.iloc[-2]

            candle_time = closed_candle["time"]

            # Vérifier si elle est nouvelle
            if (
                last_saved_time is None
                or candle_time > last_saved_time
            ):

                insert_market_data(
                    df.iloc[[-2]],
                    TABLE_NAME
                )

                print(
                    f"[{datetime.now()}] "
                    f"Nouvelle bougie M1 : "
                    f"{candle_time}"
                )

                print(
                    f"Close : "
                    f"{closed_candle['close']}"
                )

                print(
                    f"Spread : "
                    f"{closed_candle['spread']}"
                )

                last_saved_time = candle_time
                on_new_candle(candle_time)

            # Vérification toutes les secondes
            time.sleep(1)

    except KeyboardInterrupt:

        print(
            "\nArrêt du collector..."
        )

    finally:

        disconnect_mt5()


def start_realtime_collector():
    thread = threading.Thread(
        target=collect_realtime,
        kwargs={"run_initialization": False},
        name="realtime-collector",
        daemon=True,
    )
    thread.start()
    return thread
