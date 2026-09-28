# Comparaison EUR/USD M1 sur les donnees actuelles

Periode : 2026-08-19 14:00:00+00:00 -> 2026-09-24 19:49:00+00:00 ; 37,715 bougies.

Huit variantes fixees avant calcul. Resultats exploratoires : historique deja examine.
Aucune strategie activee. Lot 0,07 ; capital 100 $ ; 3 trades maximum/jour ; perte realisee journaliere 10 $ ; arret apres journee nette positive.
Couts : spread historique bid/ask, 7 $/lot/cote de commission, 1 point/cote de slippage.

| Variante | Trades G/P | Net $ | Win % | PF | Expectancy $ | DD solde % | DD equity borne % | Trades/j | Validation $ | Recent $ | Couts accrus, memes trades $ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EMA actuelle (reference) | 20 (10/10) | +2.86 | 50.0 | 1.08 | +0.143 | 10.57 | 12.20 | 0.74 | -7.91 | -0.39 | -2.74 |
| EMA + tendance M5 cloturee | 17 (9/8) | +4.75 | 52.9 | 1.16 | +0.279 | 9.76 | 12.72 | 0.63 | -5.54 | -0.87 | -0.01 |
| EMA + RSI/MACD stricts | 11 (1/10) | -32.23 | 9.1 | 0.09 | -2.930 | 32.23 | 33.25 | 0.41 | -18.20 | -6.98 | -35.31 |
| EMA session 13-17 UTC | 8 (4/4) | -1.15 | 50.0 | 0.92 | -0.143 | 7.05 | 9.43 | 0.30 | -3.55 | -3.96 | -3.39 |
| EMA filtre cout/volatilite | 9 (5/4) | +1.93 | 55.6 | 1.13 | +0.214 | 12.15 | 15.20 | 0.33 | +3.57 | +3.92 | -0.59 |
| EMA SL 1.6 / TP 2.4 ATR | 19 (10/9) | +7.14 | 52.6 | 1.20 | +0.376 | 13.78 | 16.59 | 0.70 | -0.68 | -4.77 | +1.82 |
| Breakout session | 63 (13/50) | -103.27 | 20.6 | 0.21 | -1.639 | 103.27 | 103.27 | 2.33 | -13.27 | -30.51 | -120.91 |
| RSI retour a la moyenne | 31 (9/22) | -46.30 | 29.0 | 0.25 | -1.494 | 51.26 | 53.83 | 1.15 | -24.39 | -0.92 | -54.98 |

## Motifs de rejet

### EMA actuelle (reference)

Hypothese : Regles DEMO figees; indicateurs recalcules causalement, RSI corrige.

- train: echantillon 7 trades < 10
- validation: echantillon 6 trades < 10
- validation: profit net -7.91 $ <= 0
- validation: PF 0.45 <= 1,10
- validation: expectancy -1.319 $ <= 0
- periode recente: echantillon 7 trades < 10
- periode recente: profit net -0.39 $ <= 0
- periode recente: PF 0.98 <= 1,10
- periode recente: expectancy -0.055 $ <= 0
- historique complet: echantillon 20 trades < 60
- historique complet: PF 1.08 <= 1,10
- historique: 27 jours observes < 60
- aucun holdout neuf evalue apres fixation de cette strategie

### EMA + tendance M5 cloturee

Hypothese : Exiger une tendance EMA 9/21/50 M5 alignee sur le signal M1.

- train: echantillon 7 trades < 10
- validation: echantillon 5 trades < 10
- validation: profit net -5.54 $ <= 0
- validation: PF 0.49 <= 1,10
- validation: expectancy -1.107 $ <= 0
- validation avec couts stresses: profit net <= 0
- periode recente: echantillon 5 trades < 10
- periode recente: profit net -0.87 $ <= 0
- periode recente: PF 0.93 <= 1,10
- periode recente: expectancy -0.174 $ <= 0
- historique complet: echantillon 17 trades < 60
- historique: 27 jours observes < 60
- aucun holdout neuf evalue apres fixation de cette strategie

### EMA + RSI/MACD stricts

Hypothese : Exiger RSI et MACD ensemble et limiter les entrees RSI deja etendues.

- train: echantillon 2 trades < 10
- train: profit net -7.05 $ <= 0
- train: PF 0.00 <= 1,10
- train: expectancy -3.525 $ <= 0
- validation: echantillon 7 trades < 10
- validation: profit net -18.20 $ <= 0
- validation: PF 0.15 <= 1,10
- validation: expectancy -2.600 $ <= 0
- validation: borne prudente drawdown equity 20.09% > 15%
- periode recente: echantillon 2 trades < 10
- periode recente: profit net -6.98 $ <= 0
- periode recente: PF 0.00 <= 1,10
- periode recente: expectancy -3.490 $ <= 0
- historique complet: echantillon 11 trades < 60
- historique complet: profit net -32.23 $ <= 0
- historique complet: PF 0.09 <= 1,10
- historique complet: expectancy -2.930 $ <= 0
- historique complet: borne prudente drawdown equity 33.25% > 15%
- historique: 27 jours observes < 60
- periode recente avec couts stresses: profit net <= 0
- aucun holdout neuf evalue apres fixation de cette strategie

### EMA session 13-17 UTC

Hypothese : Limiter les entrees a une fenetre horaire fixe, sans objectif quotidien.

- train: echantillon 4 trades < 10
- validation: echantillon 3 trades < 10
- validation: profit net -3.55 $ <= 0
- validation: PF 0.52 <= 1,10
- validation: expectancy -1.183 $ <= 0
- periode recente: echantillon 1 trades < 10
- periode recente: profit net -3.96 $ <= 0
- periode recente: PF 0.00 <= 1,10
- periode recente: expectancy -3.955 $ <= 0
- historique complet: echantillon 8 trades < 60
- historique complet: profit net -1.15 $ <= 0
- historique complet: PF 0.92 <= 1,10
- historique complet: expectancy -0.143 $ <= 0
- historique: 27 jours observes < 60
- periode recente avec couts stresses: profit net <= 0
- aucun holdout neuf evalue apres fixation de cette strategie

### EMA filtre cout/volatilite

Hypothese : Reserver les entrees a un spread/ATR plus faible et une volatilite moderee.

- train: echantillon 5 trades < 10
- train: profit net -5.56 $ <= 0
- train: PF 0.51 <= 1,10
- train: expectancy -1.113 $ <= 0
- validation: echantillon 3 trades < 10
- validation avec couts stresses: profit net <= 0
- periode recente: echantillon 1 trades < 10
- historique complet: echantillon 9 trades < 60
- historique complet: borne prudente drawdown equity 15.20% > 15%
- historique: 27 jours observes < 60
- periode recente avec couts stresses: profit net <= 0
- aucun holdout neuf evalue apres fixation de cette strategie

### EMA SL 1.6 / TP 2.4 ATR

Hypothese : Variante voisine de distances SL/TP, a lot et limites de risque constants.

- train: echantillon 7 trades < 10
- validation: echantillon 6 trades < 10
- validation: profit net -0.68 $ <= 0
- validation: PF 0.94 <= 1,10
- validation: expectancy -0.114 $ <= 0
- periode recente: echantillon 6 trades < 10
- periode recente: profit net -4.77 $ <= 0
- periode recente: PF 0.72 <= 1,10
- periode recente: expectancy -0.795 $ <= 0
- periode recente: borne prudente drawdown equity 18.31% > 15%
- historique complet: echantillon 19 trades < 60
- historique complet: borne prudente drawdown equity 16.59% > 15%
- historique: 27 jours observes < 60
- aucun holdout neuf evalue apres fixation de cette strategie

### Breakout session

Hypothese : Comparer une famille de continuation par cassure du range precedent.

- train: profit net -62.56 $ <= 0
- train: PF 0.18 <= 1,10
- train: expectancy -1.788 $ <= 0
- train: borne prudente drawdown equity 63.84% > 15%
- validation: profit net -13.27 $ <= 0
- validation: PF 0.45 <= 1,10
- validation: expectancy -1.021 $ <= 0
- validation: borne prudente drawdown equity 15.76% > 15%
- validation avec couts stresses: profit net <= 0
- periode recente: profit net -30.51 $ <= 0
- periode recente: PF 0.13 <= 1,10
- periode recente: expectancy -1.695 $ <= 0
- periode recente: borne prudente drawdown equity 34.04% > 15%
- historique complet: profit net -103.27 $ <= 0
- historique complet: PF 0.21 <= 1,10
- historique complet: expectancy -1.639 $ <= 0
- historique complet: borne prudente drawdown equity 103.27% > 15%
- historique: 27 jours observes < 60
- periode recente avec couts stresses: profit net <= 0
- aucun holdout neuf evalue apres fixation de cette strategie

### RSI retour a la moyenne

Hypothese : Comparer une famille de rejet RSI en regime EMA non impulsif.

- train: profit net -20.99 $ <= 0
- train: PF 0.26 <= 1,10
- train: expectancy -1.399 $ <= 0
- train: borne prudente drawdown equity 26.65% > 15%
- validation: profit net -24.39 $ <= 0
- validation: PF 0.06 <= 1,10
- validation: expectancy -2.439 $ <= 0
- validation: borne prudente drawdown equity 25.64% > 15%
- validation avec couts stresses: profit net <= 0
- periode recente: echantillon 6 trades < 10
- periode recente: profit net -0.92 $ <= 0
- periode recente: PF 0.87 <= 1,10
- periode recente: expectancy -0.153 $ <= 0
- historique complet: echantillon 31 trades < 60
- historique complet: profit net -46.30 $ <= 0
- historique complet: PF 0.25 <= 1,10
- historique complet: expectancy -1.494 $ <= 0
- historique complet: borne prudente drawdown equity 53.83% > 15%
- historique: 27 jours observes < 60
- aucun holdout neuf evalue apres fixation de cette strategie

## Reexecution avec spread +25% et slippage 2 points/cote

Ce scenario modifie les entrees, les sorties et les limites journalieres. Son portefeuille differe du test initial.

| Variante | Trades initiaux | Trades stress | Net stress $ | Validation stress $ | Recent stress $ |
|---|---:|---:|---:|---:|---:|
| EMA actuelle (reference) | 20 | 8 | +19.84 | +7.20 | +2.45 |
| EMA + tendance M5 cloturee | 17 | 4 | +13.37 | +0.00 | +3.18 |
| EMA + RSI/MACD stricts | 11 | 1 | +3.74 | +3.74 | +0.00 |
| EMA session 13-17 UTC | 8 | 3 | +4.13 | +3.75 | -4.02 |
| EMA filtre cout/volatilite | 9 | 1 | -4.43 | +0.00 | -4.43 |
| EMA SL 1.6 / TP 2.4 ATR | 19 | 8 | +16.95 | +8.06 | +2.69 |
| Breakout session | 63 | 54 | -90.91 | -1.09 | -19.15 |
| RSI retour a la moyenne | 31 | 20 | -34.60 | -15.26 | +0.36 |
## Sensibilite a la sortie temporelle

Reference sans timeout (sorties gaps/fin de jour maintenues) : 20 trades, +2.86 $.

## Limites et reproductibilite

- Historique OHLC M1 bid; spread constant dans chaque bougie, ordre intrabougie inconnu.
- SL prioritaire si SL et TP touches; gaps: sortie derniere cloture observee, convention de recherche non reproductible en live sans prevoir les donnees manquantes.
- Cloture quotidienne/de gap et timeout simules: ces sorties ne sont pas implementees dans le bot DEMO.
- Aucune simulation de marge, execution au tick, swaps ni calendrier de nouvelles economiques.
- Drawdown equity: borne prudente intrabougie, pas une trajectoire au tick observee.
- RSI corrige et indicateurs reconstruits causalement en memoire; features DB et strategie active inchangees.
- Couts stresses: memes signaux de bougies; spread execution +25%, slippage 2 points/cote.
- Le stress de reexecution change les trades retenus et les arrets journaliers; une hausse du profit stress ne prouve pas la robustesse aux couts.
- Sensibilite a trades identiques: debit comptable supplementaire sans recalcul des executions; ce n'est pas un second backtest.
- Fuseau horaire des bougies interprete UTC selon le collecteur, mais correspondance avec le broker non confirmee.
- Aucun seuil minimum de trades par jour ni de win rate; taille minimale d'echantillon distincte.

Le fichier manifest.json fixe les variantes, criteres, couts, dates et empreintes du code avant les backtests.
raw.json.gz contient uniquement les donnees de marche du test ; results.json contient les transactions et chaque scenario.
Aucune ecriture PostgreSQL, modification de .env ou activation DEMO/REAL n'est effectuee.

Controle horaire : 152 lignes ont un horodatage posterieur a l'horloge UTC du rapport. Le fuseau source doit etre confirme avant toute interpretation des sessions ou utilisation en live.
