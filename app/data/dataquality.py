import argparse
import html
from datetime import datetime
from pathlib import Path

import pandas as pd

from Database.postgresql import get_connection


DEFAULT_TABLE = "eurusd_m1"
DEFAULT_OUTPUT = "data_quality_report.html"
TIMEFRAME_MINUTES = 1
RECENT_ROWS_LIMIT = 100
FEATURE_COLUMNS = [
	"return_1m", "return_5m", "return_15m", "ema_9", "ema_21", "ema_50",
	"rsi_14", "atr_14", "macd", "macd_signal", "macd_hist", "volatility_20",
	"high_low_range", "body_size", "upper_wick", "lower_wick",
	"candle_direction", "volume_ma_20", "volume_ratio", "hour", "minute",
	"day_of_week", "session", "ema9_distance", "ema21_distance",
	"ema50_distance", "trend",
]


def _format_value(value):
	if value is None:
		return "-"
	if isinstance(value, datetime):
		return value.strftime("%Y-%m-%d %H:%M:%S")
	if isinstance(value, float):
		return f"{value:.5f}"
	return str(value)


def _cell(value):
	return f"<td>{html.escape(_format_value(value))}</td>"


def _table(headers, rows):
	header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
	rows_html = "".join(
		"<tr>" + "".join(_cell(value) for value in row) + "</tr>"
		for row in rows
	)
	empty_html = (
		f'<tr><td class="empty" colspan="{len(headers)}">Aucune donnée</td></tr>'
		if not rows
		else ""
	)
	return f"<table><thead><tr>{header_html}</tr></thead><tbody>{rows_html}{empty_html}</tbody></table>"


def _bullet_list(items):
	if not items:
		return '<p class="good-note">Aucun point bloquant détecté.</p>'
	return "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in items) + "</ul>"


def _fetch_report_data(table_name):
	connection = get_connection()
	cursor = connection.cursor()
	try:
		cursor.execute(
			f"""
			SELECT
				COUNT(*), MIN(time), MAX(time),
				COUNT(*) - COUNT(DISTINCT time),
				COUNT(*) FILTER (WHERE open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL),
				COUNT(*) FILTER (WHERE high < low OR high < open OR high < close OR low > open OR low > close),
				COUNT(*) FILTER (WHERE tick_volume < 0 OR real_volume < 0),
				COUNT(*) FILTER (WHERE spread < 0)
			FROM {table_name}
			"""
		)
		summary = cursor.fetchone()

		cursor.execute(
			f"""
			WITH ordered AS (
				SELECT time, LAG(time) OVER (ORDER BY time) AS previous_time
				FROM {table_name}
			)
			SELECT previous_time, time,
				   FLOOR(EXTRACT(EPOCH FROM (time - previous_time)) / 60)::BIGINT - 1
			FROM ordered
			WHERE previous_time IS NOT NULL
			  AND time > previous_time + (%s * INTERVAL '1 minute')
							AND NOT EXISTS (
									SELECT 1
									FROM generate_series(
											previous_time::date,
											time::date,
											INTERVAL '1 day'
									) AS calendar_day
									WHERE EXTRACT(ISODOW FROM calendar_day) IN (6, 7)
							)
			ORDER BY previous_time
			""",
			(TIMEFRAME_MINUTES,)
		)
		gaps = cursor.fetchall()

		cursor.execute(
			f"""
			SELECT time, open, high, low, close, tick_volume, spread, real_volume
			FROM {table_name}
			ORDER BY time DESC
			LIMIT %s
			""",
			(RECENT_ROWS_LIMIT,)
		)
		recent_rows = cursor.fetchall()
		return summary, gaps, recent_rows
	finally:
		cursor.close()
		connection.close()


def _fetch_features_report_data():
	connection = get_connection()
	try:
		with connection.cursor() as cursor:
			cursor.execute("SELECT * FROM public.eurusd_features ORDER BY time ASC")
			rows = cursor.fetchall()
			columns = [column.name for column in cursor.description]
		features = pd.DataFrame(rows, columns=columns)
	finally:
		connection.close()

	if features.empty:
		return features, [], [], []

	numeric_columns = [column for column in FEATURE_COLUMNS if column not in {"session"}]
	stats = []
	for column in FEATURE_COLUMNS:
		missing = int(features[column].isna().sum())
		if column in numeric_columns:
			values = pd.to_numeric(features[column], errors="coerce")
			stats.append((column, missing, values.min(), values.max(), values.mean()))
		else:
			stats.append((column, missing, "-", "-", "-"))
	sessions = features["session"].value_counts().rename_axis("Session").reset_index(name="Lignes")
	trends = features["trend"].value_counts().sort_index().rename_axis("Trend").reset_index(name="Lignes")
	return features, stats, sessions.values.tolist(), trends.values.tolist()


def collect_quality(table_name=DEFAULT_TABLE):
	"""Return the compact data-quality payload used by the dashboard."""
	summary, gaps, recent_rows = _fetch_report_data(table_name)
	(
		row_count,
		first_time,
		last_time,
		duplicate_count,
		null_count,
		ohlc_error_count,
		negative_volume_count,
		negative_spread_count,
	) = summary
	missing_candles = sum(row[2] for row in gaps)
	issue_count = null_count + ohlc_error_count + negative_volume_count + negative_spread_count
	return {
		"status": "OK" if not gaps and not duplicate_count and not issue_count else "WARNING",
		"total_candles": row_count,
		"first_time": first_time,
		"last_time": last_time,
		"gap_1m": len(gaps),
		"missing_candles": missing_candles,
		"recent_candles": recent_rows,
		"duplicate_count": duplicate_count,
		"issue_count": issue_count,
	}


def build_report_html(table_name=DEFAULT_TABLE):
	summary, gaps, recent_rows = _fetch_report_data(table_name)
	(
		row_count,
		first_time,
		last_time,
		duplicate_count,
		null_count,
		ohlc_error_count,
		negative_volume_count,
		negative_spread_count,
	) = summary
	features, feature_stats, session_rows, trend_rows = _fetch_features_report_data()
	feature_nan_count = int(features[FEATURE_COLUMNS].isna().sum().sum()) if not features.empty else 0
	feature_last_time = features["time"].max() if not features.empty else None
	missing_candles = sum(row[2] for row in gaps)
	issue_count = null_count + ohlc_error_count + negative_volume_count + negative_spread_count
	feature_lag = last_time is not None and (feature_last_time is None or feature_last_time < last_time)
	blocking_issues = bool(
		gaps or duplicate_count or issue_count or feature_nan_count or features.empty or feature_lag
	)
	status_class = "warning" if blocking_issues else "ok"
	status_text = "Données saines" if status_class == "ok" else "Points à vérifier"
	actions = []
	if gaps:
		actions.append(f"Récupérer {missing_candles} bougie(s) manquante(s) sur {len(gaps)} gap(s) avec le backfill.")
	if feature_lag:
		actions.append("Relancer feature_engineering : les features ne couvrent pas encore la dernière bougie source.")
	if feature_nan_count:
		actions.append(f"Traiter les {feature_nan_count} valeur(s) NaN restantes avant le modèle.")
	if duplicate_count:
		actions.append(f"Supprimer ou examiner {duplicate_count} doublon(s) de time dans la source.")
	if issue_count:
		actions.append("Examiner les anomalies OHLC, volume et spread avant le backtest.")
	if not actions:
		actions.append("Dataset prêt pour la stratégie, le backtest et la préparation ML.")

	gap_rows = [
		(start, end, missing_count)
		for start, end, missing_count in gaps
	]
	issue_rows = [
		("Valeurs OHLC nulles", null_count),
		("OHLC incohérents", ohlc_error_count),
		("Volumes négatifs", negative_volume_count),
		("Spreads négatifs", negative_spread_count),
	]
	generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	feature_summary = [
		("Lignes dans eurusd_features", len(features)),
		("Première date features", features["time"].min() if not features.empty else None),
		("Dernière date features", feature_last_time),
		("Différence source/features", row_count - len(features)),
		("NaN features", feature_nan_count),
	]
	feature_recent_rows = (
		features.tail(RECENT_ROWS_LIMIT)[["time", "close"] + FEATURE_COLUMNS].values.tolist()
		if not features.empty
		else []
	)

	return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qualité des données - {html.escape(table_name)}</title>
  <style>
	:root {{ --ink:#17202a; --muted:#65727e; --line:#dce3e8; --paper:#f5f7f8; --accent:#176b87; --ok:#247a54; --warn:#a35b10; }}
	* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:var(--paper); font:15px/1.5 system-ui, sans-serif; }}
	main {{ max-width:1280px; margin:0 auto; padding:32px 20px 48px; }}
	header {{ display:flex; justify-content:space-between; gap:20px; align-items:end; margin-bottom:26px; }}
	h1 {{ margin:0; font:700 clamp(28px,4vw,44px)/1.05 Georgia, serif; }} h2 {{ margin:0 0 14px; font-size:20px; }}
	.subtitle, .meta {{ color:var(--muted); }} .status {{ padding:10px 14px; border-radius:6px; font-weight:700; background:#e1f1e9; color:var(--ok); }}
	.status.warning {{ background:#fff0dc; color:var(--warn); }} .grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:24px; }}
	.metric, section {{ background:white; border:1px solid var(--line); border-radius:6px; }} .metric {{ padding:16px; }} .metric strong {{ display:block; font-size:25px; margin-top:4px; }}
	section {{ padding:20px; margin-top:18px; overflow:auto; }} table {{ width:100%; border-collapse:collapse; min-width:620px; }} th, td {{ padding:10px 12px; text-align:left; border-bottom:1px solid var(--line); white-space:nowrap; }} th {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
	.empty {{ text-align:center; color:var(--muted); }} @media (max-width:720px) {{ header {{ display:block; }} .status {{ display:inline-block; margin-top:15px; }} .grid {{ grid-template-columns:repeat(2,1fr); }} }}
	.decision {{ border-left:4px solid var(--warn); }} .decision.ok {{ border-left-color:var(--ok); }} .decision h2 {{ margin-bottom:8px; }} .decision ul {{ margin:8px 0 0; padding-left:22px; }} .good-note {{ color:var(--ok); font-weight:600; }}
  </style>
</head>
<body><main>
  <header><div><p class="subtitle">Rapport de contrôle PostgreSQL</p><h1>{html.escape(table_name)}</h1><p class="meta">Généré le {generated_at}</p></div><div class="status {status_class}">{status_text}</div></header>
  <div class="grid">
	<div class="metric">Bougies<strong>{_format_value(row_count)}</strong></div>
	<div class="metric">Première date<strong>{_format_value(first_time)}</strong></div>
	<div class="metric">Dernière date<strong>{_format_value(last_time)}</strong></div>
	<div class="metric">Gaps détectés<strong>{len(gaps)}</strong></div>
  </div>
	<section class="decision {status_class}"><h2>Décision et actions prioritaires</h2>{_bullet_list(actions)}</section>
	<section><h2>Anomalies source</h2>{_table(["Contrôle", "Nombre"], issue_rows + [("Doublons de time", duplicate_count)])}</section>
  <section><h2>Gaps de bougies M1</h2>{_table(["Début du gap", "Fin du gap", "Bougies manquantes"], gap_rows)}</section>
  <section><h2>Dernières bougies</h2>{_table(["Date", "Open", "High", "Low", "Close", "Tick volume", "Spread", "Real volume"], recent_rows)}</section>
	<section><h2>Dataset features</h2>{_table(["Contrôle", "Valeur"], feature_summary)}</section>
	<section><h2>Contrôle des features</h2>{_table(["Feature", "NaN", "Minimum", "Maximum", "Moyenne"], feature_stats)}</section>
	<section><h2>Répartition sessions</h2>{_table(["Session", "Lignes"], session_rows)}</section>
	<section><h2>Répartition tendances</h2>{_table(["Trend", "Lignes"], trend_rows)}</section>
	<section><h2>Dernières features</h2>{_table(["Date", "Close"] + FEATURE_COLUMNS, feature_recent_rows)}</section>
</main></body></html>"""


def generate_report(output_path=DEFAULT_OUTPUT, table_name=DEFAULT_TABLE):
	report = build_report_html(table_name)
	output = Path(output_path)
	output.write_text(report, encoding="utf-8")
	return output


def main():
	parser = argparse.ArgumentParser(description="Génère un rapport HTML de qualité des données.")
	parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="Fichier HTML de sortie")
	parser.add_argument("--table", default=DEFAULT_TABLE, help="Table PostgreSQL à analyser")
	args = parser.parse_args()
	output = generate_report(args.output, args.table)
	print(f"Rapport généré : {output.resolve()}")


if __name__ == "__main__":
	main()
