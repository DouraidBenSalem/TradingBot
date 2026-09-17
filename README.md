# TradingBot

TradingBot est un système de trading technique pour EUR/USD en timeframe M1. Il collecte les cotations MetaTrader 5, stocke les données dans PostgreSQL, calcule des indicateurs techniques, génère des signaux et expose un tableau de bord Flask.

La stratégie est entièrement technique : le dossier `ml_trading` n'utilise pas de modèle de machine learning.

## Fonctionnalités

- Collecte temps réel des bougies M1 depuis MetaTrader 5
- Backfill automatique des données manquantes
- Calcul de 34 indicateurs et variables techniques
- Surveillance de la qualité des données
- Signaux `BUY`, `SELL` ou `NO_TRADE`
- Gestion du risque, des sessions quotidiennes et du nombre maximal de trades
- Exécution possible sur un compte MetaTrader 5 DÉMO uniquement
- Backtest avec commissions, slippage et limites de perte
- Validation walk-forward avec déploiement refusé par défaut
- Tableau de bord web et API de supervision

## Prérequis

- Windows pour l'exécution MetaTrader 5
- Python 3.12 ou supérieur
- PostgreSQL
- MetaTrader 5 installé et connecté à un compte DÉMO
- Les packages Python listés ci-dessous

## Installation

Créer un environnement virtuel :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Installer les dépendances :

```powershell
python -m pip install MetaTrader5 pandas python-dotenv psycopg2-binary Flask numpy
```

Le projet contient localement `_pydeps/` pour un environnement de développement existant, mais ce dossier est ignoré par Git et ne doit pas servir de configuration portable.

## Configuration

Créer un fichier `.env` à la racine. Ne jamais le committer ni partager ses identifiants.

```dotenv
# MetaTrader 5
MT5_LOGIN=
MT5_PASSWORD=
MT5_SERVER=
MT5_PATH=
MT5_DEMO_MODE=true
MT5_DEMO_EXECUTION_ENABLED=false
MT5_EURUSD_SYMBOL=EURUSD.m
MT5_DEMO_MAGIC=26091401
MT5_DEMO_LOT=0.07
MT5_DEMO_DEVIATION_POINTS=10
MT5_CONNECT_TIMEOUT_MS=5000
MT5_MAX_TRADES_PER_DAY=3

# PostgreSQL
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_bot
DB_USER=postgres
DB_PASSWORD=

# Backtest
BACKTEST_INITIAL_BALANCE=100
BACKTEST_COMMISSION_PER_LOT=7
BACKTEST_SLIPPAGE_POINTS=1
BACKTEST_FIXED_LOT=0.07
BACKTEST_DAILY_LOSS_LIMIT=10
```

`MT5_DEMO_EXECUTION_ENABLED` doit être explicitement défini à `true` pour autoriser l'envoi d'ordres. Le bot reste bloqué en `NO_TRADE` tant qu'aucune stratégie n'a passé la validation et obtenu `deployment_approved: true` dans `reports/scalping_validation_report.json`.

## Démarrage

Initialiser explicitement les tables PostgreSQL, optionnellement :

```powershell
python Database/postgresql.py
```

Lancer le collecteur, le pipeline et le tableau de bord :

```powershell
python main.py
```

Le tableau de bord est disponible sur :

```text
http://127.0.0.1:5000
```

## Commandes utiles

Backfill des historiques MetaTrader 5 :

```powershell
python app/data/backfill.py
```

Générer le rapport de qualité des données :

```powershell
python app/data/dataquality.py --output data_quality_report.html --table eurusd_m1
```

Lancer la validation walk-forward des stratégies :

```powershell
python ml_trading/scalping_validation.py
```

Cette commande génère :

- `reports/scalping_validation_report.json`
- `reports/scalping_validation_report.md`

Le backtest est lancé depuis l'interface `/backtest` ou via les endpoints API dédiés.

## Tableau de bord

Principales routes :

| Route | Description |
|---|---|
| `/` | Vue principale du bot |
| `/api/status` | État courant du bot |
| `/api/monitoring` | Métriques et activité |
| `/scalping-validation` | Rapport de validation |
| `/strategy-risk` | Risques et limites de la stratégie |
| `/audit` | Journal d'audit du pipeline |
| `/mt5-demo` | État du compte et des sessions DÉMO |
| `/api/trade-history` | Historique des trades |
| `/data-quality` | Rapport de qualité des données |


## Fichiers générés

Les fichiers suivants sont créés pendant l'exécution et ignorés par Git :

- `runtime/pipeline_state.json`
- `runtime/trading_state.json`
- `runtime/trade_log.json`
- `reports/technical_backtest_report.html`
- `reports/scalping_validation_report.json`
- `reports/scalping_validation_report.md`
- `data_quality_report.html`
- `*.log`

