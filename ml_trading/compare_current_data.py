"""Read-only EUR/USD research. Never routes orders or updates deployment.

Run: python -m ml_trading.compare_current_data
Reproduce: python -m ml_trading.compare_current_data --snapshot <raw.json.gz>
The observed 2026-09-24 history is explicitly NOT a virgin holdout.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml_trading.research_execution import simulate
from ml_trading.research_signals import CANDIDATES, build_signals, prepare_features

ROOT = Path(__file__).resolve().parents[1]
# End of the data discussed with the user before this experiment was designed.
PREVIOUSLY_EXAMINED_UNTIL = pd.Timestamp("2026-09-24 19:49:00", tz="UTC")
RISK = dict(fixed_lot=0.07, initial_balance=100.0, daily_loss_limit=10.0,
            max_trades_per_day=3, commission_per_lot_side=7.0, slippage_points=1.0)
STRESS = dict(spread_multiplier=1.25, slippage_points=2.0)
CRITERIA = dict(min_train_trades=10, min_validation_trades=10, min_tail_trades=10,
                min_full_trades=60, min_observed_market_days=60,
                profit_factor_strictly_above=1.10, max_equity_drawdown_bound_pct=15.0,
                minimum_daily_frequency=None, minimum_win_rate=None)
RAW_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume", "spread"]


def json_text(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False, default=str)


def source_digest():
    paths = [Path(__file__), ROOT / "ml_trading/research_execution.py",
             ROOT / "ml_trading/research_signals.py", ROOT / "ml_trading/technical_strategy.py",
             ROOT / "ml_trading/technical_indicators.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def fetch_raw(snapshot=None):
    if snapshot:
        with gzip.open(snapshot, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        frame = pd.DataFrame(payload["rows"], columns=RAW_COLUMNS)
        metadata = payload["metadata"]
    else:
        # Only connection creation is imported; no live bot/collector imports.
        from Database.postgresql import get_connection
        connection = get_connection()
        try:
            connection.set_session(readonly=True, isolation_level="REPEATABLE READ")
            with connection.cursor() as cursor:
                cursor.execute("SELECT time,open,high,low,close,tick_volume,spread "
                               "FROM public.eurusd_m1 ORDER BY time")
                frame = pd.DataFrame(cursor.fetchall(), columns=RAW_COLUMNS)
                cursor.execute("SELECT MAX(source_end) FROM public.technical_backtest_runs")
                previously_backtested = cursor.fetchone()[0]
            metadata = {"source": "public.eurusd_m1", "database_access": "READ ONLY",
                        "previously_backtested_until": str(previously_backtested)}
        finally:
            connection.close()
    if frame.empty:
        raise ValueError("Aucune donnee EUR/USD M1 disponible")
    frame["time"] = pd.to_datetime(frame["time"])
    for column in RAW_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    numeric = frame[RAW_COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or frame["time"].isna().any():
        raise ValueError("OHLC/spread/volume manquants ou non finis: recherche arretee")
    if frame["time"].duplicated().any() or not frame["time"].is_monotonic_increasing:
        raise ValueError("Timestamps dupliques ou non tries")
    if ((frame["time"] != frame["time"].dt.floor("min"))).any():
        raise ValueError("Timestamps non alignes sur M1")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Prix non positifs")
    if ((frame["high"] < frame[["open", "close", "low"]].max(axis=1)) |
        (frame["low"] > frame[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("OHLC incoherents")
    if (frame["spread"] < 0).any() or (frame["tick_volume"] < 0).any():
        raise ValueError("Spread ou volume negatif")
    return frame, metadata


def split_masks(frame):
    dates = sorted(frame.loc[frame.time.dt.dayofweek < 5, "time"].dt.date.unique())
    if len(dates) < 12:
        raise ValueError("Au moins 12 jours observes requis pour les comparaisons chronologiques")
    validation_start = pd.Timestamp(dates[len(dates) // 2], tz=frame.time.dt.tz)
    tail_start = pd.Timestamp(dates[3 * len(dates) // 4], tz=frame.time.dt.tz)
    masks = {"full": np.ones(len(frame), dtype=bool),
             "train": (frame.time < validation_start).to_numpy(),
             "validation": ((frame.time >= validation_start) & (frame.time < tail_start)).to_numpy(),
             "diagnostic_tail": (frame.time >= tail_start).to_numpy()}
    return masks, {"validation_start": str(validation_start), "diagnostic_tail_start": str(tail_start)}


def period_failures(metrics, minimum):
    failures = []
    if metrics["total_trades"] < minimum:
        failures.append(f"echantillon {metrics['total_trades']} trades < {minimum}")
    if metrics["net_profit"] <= 0:
        failures.append(f"profit net {metrics['net_profit']:.2f} $ <= 0")
    pf = metrics["profit_factor"]
    if pf is not None and pf <= CRITERIA["profit_factor_strictly_above"]:
        failures.append(f"PF {pf:.2f} <= 1,10")
    if metrics["expectancy"] <= 0:
        failures.append(f"expectancy {metrics['expectancy']:.3f} $ <= 0")
    dd = metrics["max_equity_drawdown_bound_pct"]
    if dd > CRITERIA["max_equity_drawdown_bound_pct"]:
        failures.append(f"borne prudente drawdown equity {dd:.2f}% > 15%")
    return failures


def selection_key(result):
    """Must never reference diagnostic_tail, full performance or fresh holdout."""
    val = result["periods"]["validation"]["metrics"]
    pf = val["profit_factor"]
    return (not result["selection_failures"], val["net_profit"] > 0,
            min(pf if pf is not None else (10.0 if val["wins"] else 0.0), 10.0),
            val["expectancy"], -val["max_equity_drawdown_bound_pct"])


def matched_cost_sensitivity(trades):
    """Accounting sensitivity on IDENTICAL trades; not a new fill simulation.

    Debit 25% of the spread paid on its executable side plus one additional
    slippage point per side. Do not reconsider signals, stop hits or daily
    stopping rules. This prevents trade filtering from masking cost exposure.
    """
    pnls = []
    extra = 0.0
    for trade in trades:
        paid_spread = trade["entry_spread_points"] if trade["side"] == "BUY" else trade["exit_spread_points"]
        debit = (0.25 * paid_spread + 2.0) * 0.00001 * trade["lot"] * 100000
        extra += debit
        pnls.append(trade["net_pnl"] - debit)
    profit = sum(value for value in pnls if value > 0)
    loss = -sum(value for value in pnls if value < 0)
    return {"trade_count": len(trades), "extra_cost": extra, "net_profit": sum(pnls),
            "expectancy": sum(pnls) / len(pnls) if pnls else 0.0,
            "profit_factor": profit / loss if loss else None,
            "method": "identical trades, extra spread/slippage debit; no re-simulation of fills"}


def analyse(frame, outdir, metadata):
    masks, boundaries = split_masks(frame)
    end = frame.time.max()
    # A completed fresh trading date must be wholly after all previous tests.
    known = PREVIOUSLY_EXAMINED_UNTIL
    previous = pd.to_datetime(metadata.get("previously_backtested_until"), errors="coerce", utc=True)
    if pd.notna(previous):
        known = max(known, previous)
    # Also exclude previously exposed tails from earlier runs of this experiment.
    for old in (ROOT / "reports").glob("strategy_comparison_*/results.json"):
        old_report = json.loads(old.read_text(encoding="utf-8"))
        known = max(known, pd.to_datetime(old_report["data"]["end"], utc=True))
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "risk": RISK, "stress": STRESS, "criteria": CRITERIA,
                "candidates": CANDIDATES, "source_hashes": source_digest(),
                "split": boundaries, "previously_examined_until": str(known),
                "data_sha256": metadata["data_sha256"],
        "protocol": "8 fixed candidates; choose on train/validation only; tail descriptive; no deployment"}
    # Freeze and persist the entire plan BEFORE calculating any PnL.
    (outdir / "manifest.json").write_text(json_text(manifest), encoding="utf-8")
    print(f"PLAN FROZEN: {len(CANDIDATES)} variants; {boundaries}", flush=True)
    rows = []
    for candidate in CANDIDATES:
        signals = build_signals(frame, candidate)
        options = {**RISK, **candidate["simulation"],
                   "max_spread_to_atr": candidate["strategy"].get("max_spread_to_atr", 0.35)}
        periods = {}
        # Train/validation calculations happen before any historical-tail results.
        for key in ("train", "validation"):
            mask = masks[key]
            periods[key] = simulate(frame.loc[mask].reset_index(drop=True), signals[mask], **options)
        validation_stress = simulate(frame.loc[masks["validation"]].reset_index(drop=True),
                                     signals[masks["validation"]], **{**options, **STRESS})
        reasons = [f"train: {msg}" for msg in period_failures(periods["train"]["metrics"], CRITERIA["min_train_trades"])]
        reasons += [f"validation: {msg}" for msg in period_failures(periods["validation"]["metrics"], CRITERIA["min_validation_trades"])]
        if validation_stress["metrics"]["net_profit"] <= 0:
            reasons.append("validation avec couts stresses: profit net <= 0")
        rows.append({**candidate, "periods": periods, "validation_stress": validation_stress,
                     "selection_failures": reasons, "signal_count": int(np.count_nonzero(signals)),
                     "_signals": signals, "_options": options})
        print(f"Preselection {candidate['name']}: {len(reasons)} motifs de rejet", flush=True)
    ranked = sorted(rows, key=selection_key, reverse=True)
    selected = next((row for row in ranked if not row["selection_failures"]), None)
    (outdir / "selection_frozen.json").write_text(json_text({
        "selected": selected["name"] if selected else None,
        "ranking_train_validation_only": [r["name"] for r in ranked],
        "manifest_sha256": hashlib.sha256((outdir / "manifest.json").read_bytes()).hexdigest(),
    }), encoding="utf-8")
    for row in rows:
        signals, options = row.pop("_signals"), row.pop("_options")
        for key in ("full", "diagnostic_tail"):
            mask = masks[key]
            row["periods"][key] = simulate(frame.loc[mask].reset_index(drop=True), signals[mask], **options)
        row["full_stress"] = simulate(frame, signals, **{**options, **STRESS})
        mask = masks["diagnostic_tail"]
        row["tail_stress"] = simulate(frame.loc[mask].reset_index(drop=True), signals[mask], **{**options, **STRESS})
        row["matched_cost_sensitivity"] = {
            name: matched_cost_sensitivity(row["periods"][name]["trades"])
            for name in ("full", "validation", "diagnostic_tail")}
        failures = list(row["selection_failures"])
        failures += [f"periode recente: {msg}" for msg in period_failures(
            row["periods"]["diagnostic_tail"]["metrics"], CRITERIA["min_tail_trades"])]
        full = row["periods"]["full"]["metrics"]
        failures += [f"historique complet: {msg}" for msg in period_failures(full, CRITERIA["min_full_trades"])]
        if full["observed_market_days"] < CRITERIA["min_observed_market_days"]:
            failures.append(f"historique: {full['observed_market_days']} jours observes < {CRITERIA['min_observed_market_days']}")
        if row["tail_stress"]["metrics"]["net_profit"] <= 0:
            failures.append("periode recente avec couts stresses: profit net <= 0")
        # Existing samples have already been consulted. Do not manufacture OOS proof.
        failures.append("aucun holdout neuf evalue apres fixation de cette strategie")
        row["rejection_reasons"] = failures
        row["deployment_approved"] = False
        print(f"TERMINE {row['name']}: {full['total_trades']} trades, net {full['net_profit']:.2f}", flush=True)
    # GTC diagnostic on current reference only: no parameter search or selection.
    baseline = rows[0]
    baseline_gtc = simulate(frame, build_signals(frame, baseline),
                            **{**RISK, **baseline["simulation"], "max_holding_minutes": None,
                               "max_spread_to_atr": baseline["strategy"]["max_spread_to_atr"]})
    return {"manifest": manifest, "data": {**metadata, "rows": len(frame),
            "start": str(frame.time.min()), "end": str(end),
            "timestamps_interpretation": "Horodatages stockes interpretes UTC; decalage broker non verifie independamment",
            "rows_after_report_clock": int((frame.time > pd.Timestamp(manifest["created_at_utc"])).sum()),
            "missing_minute_intervals": int((frame.time.diff() > pd.Timedelta(minutes=1)).sum()),
            "zero_spread_rows": int((frame.spread == 0).sum())},
            "selected_on_train_validation": selected["name"] if selected else None,
            "ranking_train_validation_only": [r["name"] for r in ranked],
            "fresh_holdout": {"status": "not_evaluated", "previously_examined_until": str(known),
                              "reason": "Le bloc recent est un diagnostic deja expose, pas une validation independante."},
            "deployment_approved": False, "results": rows, "baseline_without_time_exit": baseline_gtc,
            "limitations": ["Historique OHLC M1 bid; spread constant dans chaque bougie, ordre intrabougie inconnu.",
                "SL prioritaire si SL et TP touches; gaps: sortie derniere cloture observee, convention de recherche non reproductible en live sans prevoir les donnees manquantes.",
                "Cloture quotidienne/de gap et timeout simules: ces sorties ne sont pas implementees dans le bot DEMO.",
                "Aucune simulation de marge, execution au tick, swaps ni calendrier de nouvelles economiques.",
                "Drawdown equity: borne prudente intrabougie, pas une trajectoire au tick observee.",
                "RSI corrige et indicateurs reconstruits causalement en memoire; features DB et strategie active inchangees.",
                "Couts stresses: memes signaux de bougies; spread execution +25%, slippage 2 points/cote.",
                "Le stress de reexecution change les trades retenus et les arrets journaliers; une hausse du profit stress ne prouve pas la robustesse aux couts.",
                "Sensibilite a trades identiques: debit comptable supplementaire sans recalcul des executions; ce n'est pas un second backtest.",
                "Fuseau horaire des bougies interprete UTC selon le collecteur, mais correspondance avec le broker non confirmee.",
                "Aucun seuil minimum de trades par jour ni de win rate; taille minimale d'echantillon distincte."]}


def report_markdown(report):
    data = report["data"]
    lines = ["# Comparaison EUR/USD M1 sur les donnees actuelles", "",
             f"Periode : {data['start']} -> {data['end']} ; {data['rows']:,} bougies.", "",
             "Huit variantes fixees avant calcul. Resultats exploratoires : historique deja examine.",
             "Aucune strategie activee. Lot 0,07 ; capital 100 $ ; 3 trades maximum/jour ; perte realisee journaliere 10 $ ; arret apres journee nette positive.",
             "Couts : spread historique bid/ask, 7 $/lot/cote de commission, 1 point/cote de slippage.", "",
             "| Variante | Trades G/P | Net $ | Win % | PF | Expectancy $ | DD solde % | DD equity borne % | Trades/j | Validation $ | Recent $ | Couts accrus, memes trades $ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["results"]:
        m = row["periods"]["full"]["metrics"]
        pf = f"{m['profit_factor']:.2f}" if m["profit_factor"] is not None else "sans perte"
        lines.append(f"| {row['label']} | {m['total_trades']} ({m['wins']}/{m['losses']}) | {m['net_profit']:+.2f} | {m['win_rate']:.1f} | {pf} | {m['expectancy']:+.3f} | {m['max_drawdown_pct']:.2f} | {m['max_equity_drawdown_bound_pct']:.2f} | {m['trades_per_market_day']:.2f} | {row['periods']['validation']['metrics']['net_profit']:+.2f} | {row['periods']['diagnostic_tail']['metrics']['net_profit']:+.2f} | {row['matched_cost_sensitivity']['full']['net_profit']:+.2f} |")
    lines += ["", "## Motifs de rejet", ""]
    for row in report["results"]:
        lines += [f"### {row['label']}", "", f"Hypothese : {row['rationale']}", "",
                  *[f"- {reason}" for reason in row["rejection_reasons"]], ""]
    lines += ["## Reexecution avec spread +25% et slippage 2 points/cote", "",
              "Ce scenario modifie les entrees, les sorties et les limites journalieres. Son portefeuille differe du test initial.", "",
              "| Variante | Trades initiaux | Trades stress | Net stress $ | Validation stress $ | Recent stress $ |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in report["results"]:
        stress = row["full_stress"]["metrics"]
        lines.append(f"| {row['label']} | {row['periods']['full']['metrics']['total_trades']} | {stress['total_trades']} | {stress['net_profit']:+.2f} | {row['validation_stress']['metrics']['net_profit']:+.2f} | {row['tail_stress']['metrics']['net_profit']:+.2f} |")
    gtc = report["baseline_without_time_exit"]["metrics"]
    lines += ["## Sensibilite a la sortie temporelle", "",
              f"Reference sans timeout (sorties gaps/fin de jour maintenues) : {gtc['total_trades']} trades, {gtc['net_profit']:+.2f} $.", "",
              "## Limites et reproductibilite", "", *[f"- {msg}" for msg in report["limitations"]], "",
              "Le fichier manifest.json fixe les variantes, criteres, couts, dates et empreintes du code avant les backtests.",
              "raw.json.gz contient uniquement les donnees de marche du test ; results.json contient les transactions et chaque scenario.",
              "Aucune ecriture PostgreSQL, modification de .env ou activation DEMO/REAL n'est effectuee."]
    if data.get("rows_after_report_clock"):
        lines += ["", f"Controle horaire : {data['rows_after_report_clock']} lignes ont un horodatage posterieur a l'horloge UTC du rapport. Le fuseau source doit etre confirme avant toute interpretation des sessions ou utilisation en live."]
    return "\n".join(lines) + "\n"


def report_html(markdown):
    """Small escaped renderer for the report's headings, tables and lists."""
    body, table_open, list_open = [], False, False
    for line in markdown.splitlines():
        if not line.startswith("|") and table_open:
            body.append("</tbody></table></div>")
            table_open = False
        if not line.startswith("- ") and list_open:
            body.append("</ul>")
            list_open = False
        if line.startswith("|"):
            if set(line.replace("|", "").replace(":", "").replace("-", "").strip()) == set():
                continue
            cells = [html.escape(cell.strip()) for cell in line.strip("|").split("|")]
            if not table_open:
                body.append('<div class="table"><table><thead><tr>' + ''.join(f'<th>{c}</th>' for c in cells) + '</tr></thead><tbody>')
                table_open = True
            else:
                body.append('<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            body.append(f'<h{level}>{html.escape(line[level:].strip())}</h{level}>')
        elif line.startswith("- "):
            if not list_open:
                body.append("<ul>")
                list_open = True
            body.append('<li>' + html.escape(line[2:]) + '</li>')
        elif line:
            body.append('<p>' + html.escape(line) + '</p>')
    if table_open:
        body.append("</tbody></table></div>")
    if list_open:
        body.append("</ul>")
    return ('<!doctype html><html lang="fr"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Comparaison EUR/USD M1</title><style>'
            'body{background:#f5f7fa;color:#172536;margin:2rem;font:15px/1.6 system-ui}'
            'main{max-width:1500px;margin:auto;background:white;padding:2rem;border-radius:12px}'
            'h1,h2{color:#16375b}h2{margin-top:2rem}.table{overflow-x:auto}'
            'table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px 12px;border-bottom:1px solid #dde4ec;text-align:right;white-space:nowrap}'
            'th{background:#173859;color:white}td:first-child,th:first-child{text-align:left}'
            'tr:nth-child(even){background:#f4f7fa}</style></head><body><main>'
            + '\n'.join(body) + '</main></body></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw, metadata = fetch_raw(args.snapshot)
    serialized = raw.assign(time=raw.time.astype(str)).to_json(orient="values", double_precision=15)
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    if args.snapshot and metadata.get("data_sha256") != digest:
        raise ValueError("Empreinte du snapshot differente; donnees modifiees ou serialisation incompatible")
    metadata["data_sha256"] = digest
    outdir = args.output or ROOT / "reports" / ("strategy_comparison_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    outdir.mkdir(parents=True, exist_ok=False)
    with gzip.open(outdir / "raw.json.gz", "wt", encoding="utf-8") as stream:
        json.dump({"metadata": metadata, "rows": json.loads(serialized)}, stream)
    report = analyse(prepare_features(raw), outdir, metadata)
    (outdir / "results.json").write_text(json_text(report), encoding="utf-8")
    markdown = report_markdown(report)
    (outdir / "comparison.md").write_text(markdown, encoding="utf-8")
    (outdir / "comparison.html").write_text(report_html(markdown), encoding="utf-8")
    print(f"REPORT: {outdir}", flush=True)


if __name__ == "__main__":
    main()
