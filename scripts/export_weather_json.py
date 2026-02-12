"""Export Polymarket weather data from Parquet to JSON for use in external projects.

Reads weather market data, price history, and trades from Parquet files
and exports them as JSON files that can be consumed by other tools
(e.g. Rust projects using serde).

Usage:
    uv run scripts/export_weather_json.py [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

WEATHER_DIR = Path("data/polymarket/weather")
DEFAULT_OUTPUT_DIR = WEATHER_DIR / "export"


def _parse_json_string(val: str) -> list:
    """Parse a JSON string field, handling Python-style repr strings."""
    if not val or val == "[]":
        return []
    try:
        return json.loads(val.replace("'", '"'))
    except (json.JSONDecodeError, TypeError):
        return []


def _serialize_value(val):
    """Convert values to JSON-serializable types."""
    if isinstance(val, datetime):
        return val.isoformat() + ("Z" if val.tzinfo is None else "")
    if isinstance(val, pd.Timestamp):
        return val.isoformat() + ("Z" if val.tzinfo is None else "")
    if pd.isna(val):
        return None
    return val


def export_markets(output_dir: Path) -> int:
    """Export weather markets to JSON."""
    markets_dir = WEATHER_DIR / "markets"
    parquet_files = list(markets_dir.glob("markets_*.parquet"))
    if not parquet_files:
        print("No market data found. Run the Polymarket weather indexer first.")
        return 0

    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

    records = []
    for _, row in df.iterrows():
        records.append({
            "market_id": row.get("condition_id", row.get("id", "")),
            "question": row.get("question", ""),
            "slug": row.get("slug", ""),
            "outcomes": _parse_json_string(str(row.get("outcomes", "[]"))),
            "outcome_prices": [
                float(p) for p in _parse_json_string(str(row.get("outcome_prices", "[]")))
            ],
            "token_ids": _parse_json_string(str(row.get("clob_token_ids", "[]"))),
            "volume": float(row.get("volume", 0) or 0),
            "liquidity": float(row.get("liquidity", 0) or 0),
            "active": bool(row.get("active", False)),
            "closed": bool(row.get("closed", False)),
            "end_date": _serialize_value(row.get("end_date")),
            "created_at": _serialize_value(row.get("created_at")),
        })

    path = output_dir / "weather_markets.json"
    path.write_text(json.dumps(records, indent=2, default=str))
    print(f"Exported {len(records)} markets to {path}")
    return len(records)


def export_prices(output_dir: Path) -> int:
    """Export weather price history to JSON."""
    prices_dir = WEATHER_DIR / "prices"
    parquet_files = list(prices_dir.glob("prices_*.parquet"))
    if not parquet_files:
        print("No price data found.")
        return 0

    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

    records = []
    for _, row in df.iterrows():
        ts = int(row.get("timestamp", 0))
        records.append({
            "date": datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") if ts else None,
            "timestamp": ts,
            "open": float(row.get("open", 0)),
            "high": float(row.get("high", 0)),
            "low": float(row.get("low", 0)),
            "close": float(row.get("close", 0)),
            "volume": float(row.get("volume", 0) or 0),
            "market_id": row.get("condition_id", ""),
            "token_id": row.get("token_id", ""),
        })

    path = output_dir / "weather_prices.json"
    path.write_text(json.dumps(records, indent=2, default=str))
    print(f"Exported {len(records)} price points to {path}")
    return len(records)


def export_trades(output_dir: Path) -> int:
    """Export weather trades to JSON."""
    trades_dir = WEATHER_DIR / "trades"
    parquet_files = list(trades_dir.glob("trades_*.parquet"))
    if not parquet_files:
        print("No trade data found.")
        return 0

    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

    records = []
    for _, row in df.iterrows():
        records.append({
            "condition_id": row.get("condition_id", ""),
            "asset": row.get("asset", ""),
            "side": row.get("side", ""),
            "size": float(row.get("size", 0) or 0),
            "price": float(row.get("price", 0) or 0),
            "timestamp": int(row.get("timestamp", 0) or 0),
            "outcome": row.get("outcome", ""),
            "outcome_index": int(row.get("outcome_index", 0) or 0),
            "transaction_hash": row.get("transaction_hash", ""),
        })

    path = output_dir / "weather_trades.json"
    path.write_text(json.dumps(records, indent=2, default=str))
    print(f"Exported {len(records)} trades to {path}")
    return len(records)


def main():
    parser = argparse.ArgumentParser(description="Export Polymarket weather data to JSON")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for JSON files",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting weather data to {output_dir}/\n")

    total_markets = export_markets(output_dir)
    total_prices = export_prices(output_dir)
    total_trades = export_trades(output_dir)

    print(f"\nExport complete: {total_markets} markets, {total_prices} prices, {total_trades} trades")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()
