"""Indexer for Polymarket weather prediction market data."""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.common.indexer import Indexer
from src.indexers.polymarket.client import PolymarketClient

WEATHER_DIR = Path("data/polymarket/weather")
MARKETS_DIR = WEATHER_DIR / "markets"

# Keywords that identify weather prediction markets in Polymarket questions/slugs.
# Checked against lowercase question and slug text.
WEATHER_KEYWORDS = [
    "temperature",
    "high temp",
    "low temp",
    "degrees fahrenheit",
    "degrees celsius",
    "rain ",
    "rainfall",
    "precipitation",
    "inches of rain",
    "snow",
    "snowfall",
    "inches of snow",
    "hurricane",
    "tropical storm",
    "cyclone",
    "tornado",
    "severe weather",
    "arctic ice",
    "heat wave",
    "heatwave",
    "cold front",
    "wind speed",
    "weather forecast",
    "national weather service",
    "hottest day",
    "coldest day",
    "record high",
    "record low",
    "wind chill",
    "dew point",
]

# Slug fragments that strongly indicate weather markets.
WEATHER_SLUG_FRAGMENTS = [
    "temperature",
    "rain",
    "snow",
    "hurricane",
    "tornado",
    "weather",
    "arctic-ice",
    "heat-wave",
    "heatwave",
]


def _is_weather_market(question: str, slug: str) -> bool:
    """Check if a Polymarket market is weather-related based on question and slug."""
    text = f"{question} {slug}".lower()
    if any(kw in text for kw in WEATHER_KEYWORDS):
        return True
    slug_lower = slug.lower()
    return any(frag in slug_lower for frag in WEATHER_SLUG_FRAGMENTS)


class PolymarketWeatherIndexer(Indexer):
    """Fetches and stores Polymarket weather prediction market data.

    Scans all Polymarket markets and filters for weather-related ones based
    on keyword matching in question text and slug. Stores market metadata
    including condition_id and clob_token_ids for cross-referencing with
    blockchain trade data.
    """

    def __init__(self):
        super().__init__(
            name="polymarket_weather",
            description="Fetches weather prediction markets from Polymarket",
        )

    def run(self) -> None:
        MARKETS_DIR.mkdir(parents=True, exist_ok=True)

        client = PolymarketClient()

        total = 0
        weather_count = 0
        all_weather: list[dict] = []
        chunk_size = 10000
        chunks_saved = 0

        print("Scanning Polymarket markets for weather predictions...")

        for markets, next_offset in client.iter_markets(limit=500):
            if markets:
                total += len(markets)
                weather = [
                    m for m in markets
                    if _is_weather_market(m.question, m.slug)
                ]

                if weather:
                    weather_count += len(weather)
                    fetched_at = datetime.utcnow()
                    for m in weather:
                        record = asdict(m)
                        record["_fetched_at"] = fetched_at
                        all_weather.append(record)
                    print(
                        f"Scanned {total} markets, "
                        f"found {len(weather)} weather (total weather: {weather_count})"
                    )

                # Save in chunks
                while len(all_weather) >= chunk_size:
                    chunk = all_weather[:chunk_size]
                    chunk_start = chunks_saved * chunk_size
                    chunk_path = MARKETS_DIR / f"markets_{chunk_start}_{chunk_start + chunk_size}.parquet"
                    pd.DataFrame(chunk).to_parquet(chunk_path)
                    all_weather = all_weather[chunk_size:]
                    chunks_saved += 1

            if next_offset < 0:
                break

        # Save remaining
        if all_weather:
            chunk_start = chunks_saved * chunk_size
            chunk_path = MARKETS_DIR / f"markets_{chunk_start}_{chunk_start + len(all_weather)}.parquet"
            pd.DataFrame(all_weather).to_parquet(chunk_path)

        client.close()
        print(
            f"\nScan complete: {total} markets scanned, "
            f"{weather_count} weather markets stored"
        )
