from collections.abc import Generator
from typing import Optional, Union

import httpx

from src.common.client import retry_request
from src.indexers.polymarket.models import Market, PricePoint, Trade

GAMMA_API_URL = "https://gamma-api.polymarket.com"
DATA_API_URL = "https://data-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"


class PolymarketClient:
    def __init__(
        self,
        gamma_url: str = GAMMA_API_URL,
        data_url: str = DATA_API_URL,
        clob_url: str = CLOB_API_URL,
    ):
        self.gamma_url = gamma_url
        self.data_url = data_url
        self.clob_url = clob_url
        self.client = httpx.Client(timeout=30.0)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.client.close()

    def close(self):
        self.client.close()

    @retry_request()
    def _get(self, url: str, params: Optional[dict] = None) -> Union[dict, list]:
        """Make a GET request with retry/backoff."""
        response = self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_markets(self, limit: int = 500, offset: int = 0, **kwargs) -> list[Market]:
        """Fetch markets from Gamma API."""
        params = {"limit": limit, "offset": offset, **kwargs}
        data = self._get(f"{self.gamma_url}/markets", params=params)
        if isinstance(data, list):
            return [Market.from_dict(m) for m in data]
        return [Market.from_dict(m) for m in data.get("markets", data)]

    def iter_markets(self, limit: int = 500, offset: int = 0) -> Generator[tuple[list[Market], int], None, None]:
        """Iterate through all markets using offset pagination.

        Yields:
            Tuple of (markets, next_offset) where next_offset is -1 when done.
        """
        current_offset = offset

        while True:
            markets = self.get_markets(limit=limit, offset=current_offset)

            if not markets:
                yield [], -1
                break

            next_offset = current_offset + len(markets)
            yield markets, next_offset

            if len(markets) < limit:
                break

            current_offset = next_offset

    def get_trades(self, limit: int = 500, offset: int = 0) -> list[Trade]:
        """Fetch trades from Data API.

        Note: The Polymarket data API does not support filtering by market.
        All trades are returned globally.

        Args:
            limit: Max trades to fetch (max 500)
            offset: Pagination offset
        """
        params = {"limit": min(limit, 500), "offset": offset}
        data = self._get(f"{self.data_url}/trades", params=params)
        if isinstance(data, list):
            return [Trade.from_dict(t) for t in data]
        return [Trade.from_dict(t) for t in data.get("trades", data)]

    def iter_trades(self, limit: int = 500, offset: int = 0) -> Generator[tuple[list[Trade], int], None, None]:
        """Iterate through all trades using offset pagination.

        Note: The Polymarket data API does not support filtering by market.

        Yields:
            Tuple of (trades, next_offset) where next_offset is -1 when done.
        """
        current_offset = offset

        while True:
            trades = self.get_trades(limit=limit, offset=current_offset)

            if not trades:
                yield [], -1
                break

            next_offset = current_offset + len(trades)
            yield trades, next_offset

            if len(trades) < limit:
                break

            current_offset = next_offset

    def get_market_trades(self, token_id: str, cursor: Optional[str] = None) -> tuple[list[Trade], Optional[str]]:
        """Fetch trades for a specific market token from the CLOB API.

        Args:
            token_id: The CLOB token ID for a specific outcome.
            cursor: Pagination cursor from a previous response.

        Returns:
            Tuple of (trades, next_cursor). next_cursor is None when no more pages.
        """
        params: dict = {"asset_id": token_id}
        if cursor:
            params["cursor"] = cursor

        data = self._get(f"{self.clob_url}/trades", params=params)

        trades: list[Trade] = []
        next_cursor = None

        if isinstance(data, dict):
            trades = [Trade.from_dict(t) for t in data.get("data", [])]
            next_cursor = data.get("next_cursor")
            if next_cursor == "LTE=":
                next_cursor = None
        elif isinstance(data, list):
            trades = [Trade.from_dict(t) for t in data]

        return trades, next_cursor

    def get_all_market_trades(self, token_id: str) -> list[Trade]:
        """Fetch all trades for a specific market token, handling pagination."""
        all_trades: list[Trade] = []
        cursor = None

        while True:
            trades, next_cursor = self.get_market_trades(token_id, cursor=cursor)
            if trades:
                all_trades.extend(trades)
            if not next_cursor:
                break
            cursor = next_cursor

        return all_trades

    def get_price_history(
        self,
        token_id: str,
        interval: str = "all",
        fidelity: int = 60,
    ) -> list[PricePoint]:
        """Fetch OHLCV price history for a market token from the CLOB API.

        Args:
            token_id: The CLOB token ID for a specific outcome.
            interval: Time range - "all", "1w", "1d", "6h", "1h".
            fidelity: Candle resolution in minutes (1, 5, 60, etc.).

        Returns:
            List of PricePoint candles.
        """
        params = {
            "market": token_id,
            "interval": interval,
            "fidelity": fidelity,
        }
        data = self._get(f"{self.clob_url}/prices-history", params=params)

        points: list[dict] = []
        if isinstance(data, dict):
            points = data.get("history", [])
        elif isinstance(data, list):
            points = data

        return [PricePoint.from_dict(p, token_id=token_id) for p in points]
