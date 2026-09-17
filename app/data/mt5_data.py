import os
from datetime import timezone
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

from dotenv import load_dotenv


load_dotenv()

MT5_CONNECT_TIMEOUT_MS = int(os.getenv("MT5_CONNECT_TIMEOUT_MS", "5000"))


def _terminal_path():
    configured = os.getenv("MT5_PATH")
    if configured and Path(configured).is_file():
        return configured
    program_files = os.getenv("ProgramFiles")
    if not program_files:
        return None
    matches = sorted(Path(program_files).glob("*MetaTrader 5*/terminal64.exe"))
    return str(matches[0]) if len(matches) == 1 else None


def connect_mt5():

    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    if not login or not password or not server:
        raise RuntimeError(
            "Les informations MT5 sont absentes du fichier .env"
        )

    login = int(login)

    # Connexion au compte MT5
    terminal_path = _terminal_path()
    initialize_args = (terminal_path,) if terminal_path else ()
    connected = mt5.initialize(
        *initialize_args,
        login=login,
        password=password,
        server=server,
        timeout=MT5_CONNECT_TIMEOUT_MS,
    )

    if not connected:

        error = mt5.last_error()

        raise RuntimeError(
            f"Impossible de se connecter à MT5 : {error}"
        )

    print("================================")
    print("MT5 connecté avec succès")
    print("================================")

    # Informations du compte
    account = mt5.account_info()

    if account is not None:

        print(f"Login  : {account.login}")
        print(f"Serveur: {account.server}")
        print(f"Balance: {account.balance}")
        print(f"Equity : {account.equity}")


def disconnect_mt5():

    mt5.shutdown()

    print("MT5 déconnecté")


def find_eurusd_symbol():

    symbols = mt5.symbols_get()

    if symbols is None:
        raise RuntimeError(
            "Impossible de récupérer les symboles MT5"
        )

    matches = []

    for symbol in symbols:

        name = symbol.name.upper()

        if "EURUSD" in name:

            matches.append(symbol.name)

    return matches


def get_market_data(
    symbol,
    timeframe,
    bars=5000
):

    # Vérifier que le symbole existe
    symbol_info = mt5.symbol_info(symbol)

    if symbol_info is None:

        raise RuntimeError(
            f"Symbole introuvable : {symbol}"
        )

    # Activer le symbole
    if not symbol_info.visible:

        success = mt5.symbol_select(
            symbol,
            True
        )

        if not success:

            raise RuntimeError(
                f"Impossible d'activer {symbol}"
            )

    # Récupération des bougies
    rates = mt5.copy_rates_from_pos(
        symbol,
        timeframe,
        0,
        bars
    )

    if rates is None:

        raise RuntimeError(
            f"Erreur récupération données : "
            f"{mt5.last_error()}"
        )

    # Conversion en DataFrame
    df = pd.DataFrame(rates)

    # Conversion timestamp
    df["time"] = pd.to_datetime(
        df["time"],
        unit="s"
    )

    return df


def get_market_data_range(
    symbol,
    timeframe,
    datetime_from,
    datetime_to
):

    if datetime_from.tzinfo is None:
        datetime_from = datetime_from.replace(tzinfo=timezone.utc)
    else:
        datetime_from = datetime_from.astimezone(timezone.utc)

    if datetime_to.tzinfo is None:
        datetime_to = datetime_to.replace(tzinfo=timezone.utc)
    else:
        datetime_to = datetime_to.astimezone(timezone.utc)

    rates = mt5.copy_rates_range(
        symbol,
        timeframe,
        datetime_from,
        datetime_to
    )

    if rates is None:

        raise RuntimeError(
            f"Erreur récupération historique : "
            f"{mt5.last_error()}"
        )

    df = pd.DataFrame(rates)

    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)

    return df
