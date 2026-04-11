from __future__ import annotations
from typing import Any, Dict, Optional
import pandas as pd

class OpenTravelDataService:
    """
    Structured lookup service backed by pandas DataFrames loaded from OpenTravelData.

    This is NOT a RAG service.
    It is a local structured lookup layer for airports, airlines, and related travel data.
    """

    def __init__(
        self,
        airports_df: Optional[pd.DataFrame] = None,
        airlines_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.airports_df = airports_df
        self.airlines_df = airlines_df

    def _ensure_airports_loaded(self) -> None:
        if self.airports_df is None:
            raise ValueError("Airports DataFrame is not loaded.")

    def _ensure_airlines_loaded(self) -> None:
        if self.airlines_df is None:
            raise ValueError("Airlines DataFrame is not loaded.")

    def preview_airports_columns(self) -> list[str]:
        self._ensure_airports_loaded()
        return self.airports_df.columns.tolist()

    def preview_airlines_columns(self) -> list[str]:
        self._ensure_airlines_loaded()
        return self.airlines_df.columns.tolist()

    def get_airport_by_iata(self, code: str) -> list[Dict[str, Any]]:
        """
        Exact lookup by IATA code.
        """
        self._ensure_airports_loaded()
        code = code.strip().upper()

        df = self.airports_df.copy()
        candidate_cols = [c for c in df.columns if c.lower() in {"iata_code", "iata", "airport_code"}]

        if not candidate_cols:
            raise ValueError("No obvious IATA column found in airports DataFrame.")

        col = candidate_cols[0]
        results = df[df[col].astype(str).str.upper() == code]
        return results.to_dict(orient="records")

    def find_airports_by_city(self, city: str, limit: int = 10) -> list[Dict[str, Any]]:
        """
        Fuzzy contains-match on likely city-related columns.
        """
        self._ensure_airports_loaded()
        city = city.strip().lower()

        df = self.airports_df.copy()
        possible_city_cols = [c for c in df.columns if "city" in c.lower() or "municip" in c.lower()]

        if not possible_city_cols:
            raise ValueError("No obvious city column found in airports DataFrame.")

        mask = False
        for col in possible_city_cols:
            mask = mask | df[col].astype(str).str.lower().str.contains(city, na=False)

        results = df[mask].head(limit)
        return results.to_dict(orient="records")

    def find_airlines_by_name(self, name: str, limit: int = 10) -> list[Dict[str, Any]]:
        """
        Fuzzy contains-match on likely airline-name columns.
        """
        self._ensure_airlines_loaded()
        name = name.strip().lower()

        df = self.airlines_df.copy()
        possible_name_cols = [c for c in df.columns if "name" in c.lower()]

        if not possible_name_cols:
            raise ValueError("No obvious airline name column found in airlines DataFrame.")

        mask = False
        for col in possible_name_cols:
            mask = mask | df[col].astype(str).str.lower().str.contains(name, na=False)

        results = df[mask].head(limit)
        return results.to_dict(orient="records")