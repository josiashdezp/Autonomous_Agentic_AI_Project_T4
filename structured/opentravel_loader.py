from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import json
import math
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    """
    Lightweight dataset descriptor derived from structured/opentraveldata_memory_input.json.

    Notes:
    - OpenTravelData CSVs in this project are caret-delimited (^) rather than comma-delimited.
    - `path` is relative to the OpenTravelData base directory (default: data/opentraveldata).
    """

    name: str            # filename (e.g., optd_airlines.csv)
    path: str            # relative path (e.g., curated/regions/optd_region_0001_geonames.csv)
    columns: list[str]   # parsed header columns from the first line


class OpenTravelDataLoader:
    """
    Loads selected OpenTravelData CSV files from a local directory into pandas DataFrames.

    Expected folder structure:
        data/opentraveldata/
            <csv files here>

    This loader does not assume one exact filename convention beyond what you pass in.
    """

    DEFAULT_REGISTRY_PATH = Path("structured/opentraveldata_memory_input.json")

    # Fallback aliases (used only if registry JSON has no aliases).
    # You can still pass an explicit filename/path to any load_* method.
    DEFAULT_DATASET_ALIASES: dict[str, str] = {
        # Airlines
        "airlines": "optd_airlines.csv",
        "airline_best_known": "optd_airline_best_known_so_far.csv",
        "airline_alliance_membership": "optd_airline_alliance_membership.csv",

        # "Airports / cities / stations" live in the POR (points of reference) table.
        "por": "optd_por_public.csv",
        "airports": "optd_por_public.csv",
        "cities": "optd_por_public.csv",

        # Geography
        "countries": "optd_countries.csv",
        "regions": "optd_regions.csv",
        "region_details": "optd_region_details.csv",

        # Misc
        "aircraft": "optd_aircraft.csv",
        "airport_popularity": "ref_airport_popularity.csv",
        "airport_pagerank": "ref_airport_pageranked.csv",
    }

    def __init__(
        self,
        base_path: str | Path = "data/opentraveldata",
        registry_path: str | Path | None = None,
        dataset_aliases: Optional[dict[str, str]] = None,
    ) -> None:
        self.base_path = Path(base_path)
        self.registry_path = Path(registry_path) if registry_path else self.DEFAULT_REGISTRY_PATH
        # Start with fallback aliases; the registry (if present) can override/extend these.
        self.dataset_aliases: dict[str, str] = {**self.DEFAULT_DATASET_ALIASES, **(dataset_aliases or {})}

        self._specs_by_path: dict[str, DatasetSpec] = {}
        self._specs_by_name: dict[str, DatasetSpec] = {}
        self._load_registry_best_effort()

    def _load_registry_best_effort(self) -> None:
        """
        Load structured/opentraveldata_memory_input.json (if present) and build fast lookup indexes.

        This is best-effort: the loader still works without the registry as long as you pass filenames.
        """
        if not self.registry_path.exists():
            return

        raw = json.loads(self.registry_path.read_text(encoding="utf-8"))

        # v2 format (preferred):
        # {
        #   "version": 2,
        #   "aliases": {"airlines": "optd_airlines.csv", ...},
        #   "datasets": [{"name": "...", "path": "...", "columns": [...]}, ...]
        # }
        if isinstance(raw, dict):
            aliases = raw.get("aliases")
            if isinstance(aliases, dict):
                # Registry aliases override fallback aliases.
                for k, v in aliases.items():
                    if isinstance(k, str) and isinstance(v, str):
                        self.dataset_aliases[k.lower()] = v

            datasets = raw.get("datasets")
            if isinstance(datasets, list):
                for ds in datasets:
                    if not isinstance(ds, dict):
                        continue
                    name = ds.get("name") or ds.get("title")
                    rel_path = ds.get("path") or ds.get("url")
                    cols = ds.get("columns") or []

                    if not isinstance(name, str) or not isinstance(rel_path, str):
                        continue
                    if not isinstance(cols, list):
                        cols = []

                    spec = DatasetSpec(name=name, path=rel_path, columns=[str(c) for c in cols])
                    path_key = rel_path.replace("\\", "/")
                    self._specs_by_path[path_key] = spec
                    self._specs_by_name[name] = spec
            return

        # v1 format (legacy): list[{"title": ..., "metadata": {"path": ..., "columns": [...]}, ...}]
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue

                name = item.get("title")
                meta = item.get("metadata") or {}
                rel_path = meta.get("path") or item.get("url")
                cols = meta.get("columns") or []

                if not isinstance(name, str) or not isinstance(rel_path, str):
                    continue
                if not isinstance(cols, list):
                    cols = []

                spec = DatasetSpec(name=name, path=rel_path, columns=[str(c) for c in cols])
                path_key = rel_path.replace("\\", "/")
                self._specs_by_path[path_key] = spec
                self._specs_by_name[name] = spec

    def _read_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        # Accept either a bare filename ("optd_airlines.csv") or a relative path
        # ("curated/regions/optd_region_0001_geonames.csv").
        filename = filename.replace("\\", "/")
        file_path = self.base_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        # OpenTravelData uses caret (^) delimiters.
        # Allow overrides via kwargs, but default to correct parsing.
        if "sep" not in kwargs and "delimiter" not in kwargs:
            kwargs["sep"] = "^"
        if "low_memory" not in kwargs:
            kwargs["low_memory"] = False

        return pd.read_csv(file_path, **kwargs)

    def resolve_dataset(self, name_or_alias_or_path: str) -> str:
        """
        Resolve a dataset identifier to a relative CSV path under base_path.

        Accepts:
        - an alias (e.g., "airlines", "airports", "cities")
        - a filename (e.g., "optd_airlines.csv")
        - a relative path (e.g., "curated/regions/optd_region_0001_geonames.csv")
        """
        key = (name_or_alias_or_path or "").strip()
        if not key:
            raise ValueError("Dataset key is empty.")

        alias_target = self.dataset_aliases.get(key.lower())
        if alias_target:
            key = alias_target

        key_norm = key.replace("\\", "/")

        # If the registry knows it, return its normalized path.
        if key_norm in self._specs_by_path:
            return self._specs_by_path[key_norm].path.replace("\\", "/")
        if key in self._specs_by_name:
            return self._specs_by_name[key].path.replace("\\", "/")

        # Otherwise assume it's already a relative path/filename.
        return key_norm

    def load_airports(self, filename: str | None = None) -> pd.DataFrame:
        """
        Load an airport-related CSV.

        Default:
        - Loads the main POR table (`optd_por_public.csv`) which includes airports and cities,
          identified by columns like `iata_code`, `name`, `latitude`, `longitude`, `location_type`.
        """
        resolved = self.resolve_dataset(filename or "airports")
        return self._read_csv(resolved)

    def load_airlines(self, filename: str | None = None) -> pd.DataFrame:
        """
        Load an airline-related CSV.
        """
        resolved = self.resolve_dataset(filename or "airlines")
        return self._read_csv(resolved)

    def load_countries(self, filename: str | None = None) -> pd.DataFrame:
        """
        Load a country-related CSV (default: optd_countries.csv).
        """
        resolved = self.resolve_dataset(filename or "countries")
        return self._read_csv(resolved)

    def load_generic(self, filename: str) -> pd.DataFrame:
        """
        Load any CSV from the OpenTravelData folder.
        """
        resolved = self.resolve_dataset(filename)
        return self._read_csv(resolved)

    def list_csv_files(self) -> list[str]:
        """
        List available CSV files in the local OpenTravelData folder.
        """
        if not self.base_path.exists():
            return []
        return sorted([p.name for p in self.base_path.glob("*.csv")])

    def list_registry_datasets(self) -> list[DatasetSpec]:
        """
        List all datasets known by the registry JSON.
        """
        # Stable ordering for UI/debugging.
        return sorted(self._specs_by_path.values(), key=lambda s: s.path)

    # ------------------------------------------------------------------
    # Convenience helpers (no external deps)
    # ------------------------------------------------------------------
    @staticmethod
    def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Great-circle distance in kilometers between two (lat, lon) points.
        """
        r = 6371.0088  # mean Earth radius in km
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r * c

    def distance_km_between_iata(self, iata_a: str, iata_b: str, por_filename: str | None = None) -> float:
        """
        Compute distance (km) between two IATA codes using the POR dataset coordinates.

        - iata_a / iata_b: e.g., "JFK", "LAX"
        - por_filename: override the POR dataset (default alias: "por" -> optd_por_public.csv)
        """
        por = self.load_airports(por_filename or "por")

        # The main POR file uses `iata_code` + `latitude` + `longitude`.
        required = {"iata_code", "latitude", "longitude"}
        cols_lower = {c.lower(): c for c in por.columns}
        missing = [c for c in required if c not in cols_lower]
        if missing:
            raise ValueError(f"POR dataset missing required columns for distance: {missing}")

        iata_col = cols_lower["iata_code"]
        lat_col = cols_lower["latitude"]
        lon_col = cols_lower["longitude"]

        a = iata_a.strip().upper()
        b = iata_b.strip().upper()

        rows = por[por[iata_col].astype(str).str.upper().isin([a, b])][[iata_col, lat_col, lon_col]]
        if rows.shape[0] < 2:
            raise ValueError(f"Could not find both IATA codes in POR dataset: {a}, {b}")

        def _coords(code: str) -> tuple[float, float]:
            r = rows[rows[iata_col].astype(str).str.upper() == code].head(1)
            if r.empty:
                raise ValueError(f"IATA code not found: {code}")
            lat = float(r.iloc[0][lat_col])
            lon = float(r.iloc[0][lon_col])
            return lat, lon

        lat1, lon1 = _coords(a)
        lat2, lon2 = _coords(b)
        return self.haversine_km(lat1, lon1, lat2, lon2)
