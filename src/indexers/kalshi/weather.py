"""Indexer for Kalshi weather prediction market data."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.common.indexer import Indexer
from src.common.storage import ParquetStorage
from src.indexers.kalshi.client import KalshiClient

WEATHER_DIR = Path("data/kalshi/weather")
MARKETS_DIR = WEATHER_DIR / "markets"
TRADES_DIR = WEATHER_DIR / "trades"

# Weather event ticker prefixes from the category system.
# More specific prefixes are listed first, with generic catch-alls at the end.
WEATHER_PREFIXES = [
    "HIGHNY", "HIGHCHI", "HIGHAUS", "HIGHMIA", "HIGHLAX",
    "HIGHDEN", "HIGHPHIL", "HIGHHOU", "HMONTH",
    "RAINNYC", "RAINNYCM",
    "SNOWNYM",
    "TORNADO", "HURCAT", "ARCTICICE",
    "WEATHER", "HIGH", "RAIN", "SNOW",
]


def _is_weather_market(event_ticker: str) -> bool:
    """Check if an event ticker belongs to a weather prediction market."""
    if not event_ticker:
        return False
    upper = event_ticker.upper()
    return any(upper.startswith(prefix) for prefix in WEATHER_PREFIXES)


class KalshiWeatherIndexer(Indexer):
    """Fetches and stores Kalshi weather prediction market data (markets + trades)."""

    def __init__(self, max_workers: int = 10):
        super().__init__(
            name="kalshi_weather",
            description="Fetches weather prediction markets and trades from Kalshi",
        )
        self._max_workers = max_workers

    def run(self) -> None:
        MARKETS_DIR.mkdir(parents=True, exist_ok=True)
        TRADES_DIR.mkdir(parents=True, exist_ok=True)

        # Phase 1: Fetch weather markets
        print("Phase 1: Fetching weather markets...")
        weather_tickers = self._fetch_weather_markets()

        if not weather_tickers:
            print("No weather markets found.")
            return

        # Phase 2: Fetch trades for weather markets
        print(f"\nPhase 2: Fetching trades for {len(weather_tickers)} weather markets...")
        self._fetch_weather_trades(weather_tickers)

    def _fetch_weather_markets(self) -> list[str]:
        """Fetch all markets from Kalshi, filter for weather, and store them.

        Returns:
            List of weather market tickers for trade fetching.
        """
        client = KalshiClient()
        storage = ParquetStorage(data_dir=MARKETS_DIR)

        total = 0
        weather_count = 0
        weather_tickers: list[str] = []

        for markets, next_cursor in client.iter_markets(limit=1000):
            if markets:
                weather = [m for m in markets if _is_weather_market(m.event_ticker)]
                total += len(markets)

                if weather:
                    weather_count += len(weather)
                    storage.append_markets(weather)
                    weather_tickers.extend(m.ticker for m in weather)
                    print(
                        f"Scanned {total} markets, "
                        f"found {len(weather)} weather (total weather: {weather_count})"
                    )

            if not next_cursor:
                break

        client.close()
        print(f"\nMarkets scan complete: {total} scanned, {weather_count} weather markets stored")
        return weather_tickers

    def _fetch_weather_trades(self, tickers: list[str]) -> None:
        """Fetch and store trades for the given weather market tickers."""
        batch_size = 10000

        all_trades: list[dict] = []
        total_saved = 0
        next_chunk_idx = 0

        # Resume from existing chunks if any
        existing_files = list(TRADES_DIR.glob("trades_*.parquet"))
        if existing_files:
            indices = []
            for f in existing_files:
                parts = f.stem.split("_")
                if len(parts) >= 2:
                    try:
                        indices.append(int(parts[1]))
                    except ValueError:
                        pass
            if indices:
                next_chunk_idx = max(indices) + batch_size

        def save_batch(trades_batch: list[dict]) -> int:
            nonlocal next_chunk_idx
            if not trades_batch:
                return 0
            path = TRADES_DIR / f"trades_{next_chunk_idx}_{next_chunk_idx + batch_size}.parquet"
            df = pd.DataFrame(trades_batch)
            df.to_parquet(path)
            next_chunk_idx += batch_size
            return len(trades_batch)

        def fetch_ticker_trades(ticker: str) -> list[dict]:
            client = KalshiClient()
            try:
                trades = client.get_market_trades(ticker, verbose=False)
                if not trades:
                    return []
                fetched_at = datetime.utcnow()
                return [{**asdict(t), "_fetched_at": fetched_at} for t in trades]
            finally:
                client.close()

        pbar = tqdm(total=len(tickers), desc="Fetching weather trades")
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(fetch_ticker_trades, ticker): ticker
                for ticker in tickers
            }

            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    trades_data = future.result()
                    if trades_data:
                        all_trades.extend(trades_data)

                    pbar.update(1)
                    pbar.set_postfix(
                        buffer=len(all_trades),
                        saved=total_saved,
                        last=ticker[-20:],
                    )

                    while len(all_trades) >= batch_size:
                        total_saved += save_batch(all_trades[:batch_size])
                        all_trades = all_trades[batch_size:]

                except Exception as e:
                    pbar.update(1)
                    tqdm.write(f"Error fetching {ticker}: {e}")

        pbar.close()

        if all_trades:
            total_saved += save_batch(all_trades)

        print(
            f"\nTrades complete: {len(tickers)} markets processed, "
            f"{total_saved} trades saved"
        )
