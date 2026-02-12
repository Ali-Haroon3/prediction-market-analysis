"""Indexer for Polymarket weather prediction market data."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.common.indexer import Indexer
from src.indexers.polymarket.client import PolymarketClient
from src.indexers.polymarket.models import Market

WEATHER_DIR = Path("data/polymarket/weather")
MARKETS_DIR = WEATHER_DIR / "markets"
TRADES_DIR = WEATHER_DIR / "trades"
PRICES_DIR = WEATHER_DIR / "prices"

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


def _parse_token_ids(clob_token_ids: str) -> list[str]:
    """Parse CLOB token IDs from the JSON string stored in market data."""
    try:
        ids = json.loads(clob_token_ids.replace("'", '"'))
        if isinstance(ids, list):
            return [str(t) for t in ids if t]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


class PolymarketWeatherIndexer(Indexer):
    """Fetches and stores Polymarket weather prediction market data.

    Three-phase indexer:
    1. Scans all markets, filters for weather, stores metadata.
    2. Fetches OHLCV price history for each weather market via CLOB API.
    3. Fetches per-market trades via CLOB API.
    """

    def __init__(self, max_workers: int = 5):
        super().__init__(
            name="polymarket_weather",
            description="Fetches weather prediction markets, price history, and trades from Polymarket",
        )
        self._max_workers = max_workers

    def run(self) -> None:
        MARKETS_DIR.mkdir(parents=True, exist_ok=True)
        TRADES_DIR.mkdir(parents=True, exist_ok=True)
        PRICES_DIR.mkdir(parents=True, exist_ok=True)

        # Phase 1: Fetch weather markets
        print("Phase 1: Fetching weather markets...")
        weather_markets = self._fetch_weather_markets()

        if not weather_markets:
            print("No weather markets found.")
            return

        # Collect all token IDs from weather markets
        token_map: dict[str, str] = {}  # token_id -> condition_id
        for m in weather_markets:
            token_ids = _parse_token_ids(m.clob_token_ids)
            for tid in token_ids:
                token_map[tid] = m.condition_id

        if not token_map:
            print("No CLOB token IDs found for weather markets. Skipping price/trade fetch.")
            return

        print(f"\nFound {len(token_map)} outcome tokens across {len(weather_markets)} weather markets.")

        # Phase 2: Fetch price history
        print(f"\nPhase 2: Fetching price history for {len(token_map)} tokens...")
        self._fetch_price_history(token_map)

        # Phase 3: Fetch per-market trades
        print(f"\nPhase 3: Fetching trades for {len(token_map)} tokens...")
        self._fetch_trades(token_map)

    def _fetch_weather_markets(self) -> list[Market]:
        """Fetch all markets, filter for weather, and store them."""
        client = PolymarketClient()

        total = 0
        weather_count = 0
        weather_markets: list[Market] = []
        all_weather_records: list[dict] = []
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
                    weather_markets.extend(weather)
                    fetched_at = datetime.utcnow()
                    for m in weather:
                        record = asdict(m)
                        record["_fetched_at"] = fetched_at
                        all_weather_records.append(record)
                    print(
                        f"Scanned {total} markets, "
                        f"found {len(weather)} weather (total weather: {weather_count})"
                    )

                while len(all_weather_records) >= chunk_size:
                    chunk = all_weather_records[:chunk_size]
                    chunk_start = chunks_saved * chunk_size
                    path = MARKETS_DIR / f"markets_{chunk_start}_{chunk_start + chunk_size}.parquet"
                    pd.DataFrame(chunk).to_parquet(path)
                    all_weather_records = all_weather_records[chunk_size:]
                    chunks_saved += 1

            if next_offset < 0:
                break

        if all_weather_records:
            chunk_start = chunks_saved * chunk_size
            path = MARKETS_DIR / f"markets_{chunk_start}_{chunk_start + len(all_weather_records)}.parquet"
            pd.DataFrame(all_weather_records).to_parquet(path)

        client.close()
        print(f"\nMarkets: {total} scanned, {weather_count} weather markets stored")
        return weather_markets

    def _fetch_price_history(self, token_map: dict[str, str]) -> None:
        """Fetch OHLCV price history for each token via CLOB API."""
        all_prices: list[dict] = []

        def fetch_one(token_id: str) -> list[dict]:
            client = PolymarketClient()
            try:
                points = client.get_price_history(token_id, interval="all", fidelity=60)
                if not points:
                    return []
                fetched_at = datetime.utcnow()
                return [
                    {**asdict(p), "condition_id": token_map[token_id], "_fetched_at": fetched_at}
                    for p in points
                ]
            except Exception:
                return []
            finally:
                client.close()

        pbar = tqdm(total=len(token_map), desc="Fetching price history")
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(fetch_one, tid): tid
                for tid in token_map
            }
            for future in as_completed(futures):
                tid = futures[future]
                try:
                    records = future.result()
                    if records:
                        all_prices.extend(records)
                except Exception as e:
                    tqdm.write(f"Error fetching prices for {tid[:12]}...: {e}")
                pbar.update(1)
                pbar.set_postfix(records=len(all_prices))
        pbar.close()

        if all_prices:
            path = PRICES_DIR / "prices_0.parquet"
            pd.DataFrame(all_prices).to_parquet(path)

        print(f"Price history: {len(all_prices)} candles stored")

    def _fetch_trades(self, token_map: dict[str, str]) -> None:
        """Fetch per-market trades for each token via CLOB API."""
        batch_size = 10000
        all_trades: list[dict] = []
        total_saved = 0
        next_chunk_idx = 0

        def save_batch(trades_batch: list[dict]) -> int:
            nonlocal next_chunk_idx
            if not trades_batch:
                return 0
            path = TRADES_DIR / f"trades_{next_chunk_idx}_{next_chunk_idx + batch_size}.parquet"
            pd.DataFrame(trades_batch).to_parquet(path)
            next_chunk_idx += batch_size
            return len(trades_batch)

        def fetch_one(token_id: str) -> list[dict]:
            client = PolymarketClient()
            try:
                trades = client.get_all_market_trades(token_id)
                if not trades:
                    return []
                fetched_at = datetime.utcnow()
                return [
                    {**asdict(t), "_fetched_at": fetched_at}
                    for t in trades
                ]
            except Exception:
                return []
            finally:
                client.close()

        pbar = tqdm(total=len(token_map), desc="Fetching trades")
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(fetch_one, tid): tid
                for tid in token_map
            }
            for future in as_completed(futures):
                tid = futures[future]
                try:
                    records = future.result()
                    if records:
                        all_trades.extend(records)
                except Exception as e:
                    tqdm.write(f"Error fetching trades for {tid[:12]}...: {e}")

                pbar.update(1)
                pbar.set_postfix(buffer=len(all_trades), saved=total_saved)

                while len(all_trades) >= batch_size:
                    total_saved += save_batch(all_trades[:batch_size])
                    all_trades = all_trades[batch_size:]
        pbar.close()

        if all_trades:
            total_saved += save_batch(all_trades)

        print(f"Trades: {total_saved} trades stored")
