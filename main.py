"""Point d'entrée unique du bot EUR/USD M1."""

import sys
from pathlib import Path


# L'interpréteur embarqué contient les dépendances compilées, tandis que
# quelques dépendances Python pures sont fournies dans ``_pydeps``.
_LOCAL_DEPENDENCIES = Path(__file__).resolve().parent / "_pydeps"
if _LOCAL_DEPENDENCIES.is_dir():
    sys.path.append(str(_LOCAL_DEPENDENCIES))


from ml_trading.dashboard import run as run_dashboard
from ml_trading.pipeline import initialize
from app.data.realtime_collector import start_realtime_collector

if __name__ == "__main__":
    print("Synchronisation EUR/USD, stratégie technique et monitoring...")
    initialize()
    start_realtime_collector()
    print("Dashboard disponible sur http://127.0.0.1:5000")
    run_dashboard(host="127.0.0.1", port=5000)
