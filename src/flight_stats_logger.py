from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable


class FlightStatsLogger:
    """Tracks per-day counts of flights by load fullness."""

    def __init__(self, reports_dir: str = "reports/flights") -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.reset()

    def reset(self) -> None:
        self.by_day: Dict[int, Dict[str, int]] = defaultdict(lambda: {"full": 0, "partial": 0, "empty": 0, "total": 0})

    def record(self, response: Dict[str, Any]) -> None:
        """Record stats from one /play/round response."""
        day = int(response.get("day", 0))
        stats = self.by_day[day]
        for upd in response.get("flightUpdates", []):
            # Only consider flights that have a loadedKits entry in the request via plane_tracker (or inferred)
            loaded = upd.get("loadedKits") or {}
            if not loaded:
                continue
            total_load = sum(int(loaded.get(k, 0)) for k in ["first", "business", "premiumEconomy", "economy"])
            pax = upd.get("passengers", {}) or {}
            total_pax = sum(int(pax.get(k, 0)) for k in ["first", "business", "premiumEconomy", "economy"])
            stats["total"] += 1
            if total_load == 0:
                stats["empty"] += 1
            elif total_load >= total_pax:
                stats["full"] += 1
            else:
                stats["partial"] += 1

    def _write_csv(self, rows: Iterable[Iterable[Any]], path: Path) -> None:
        lines = []
        for row in rows:
            parts = []
            for item in row:
                text = str(item)
                if any(ch in text for ch in [",", '"', "\n"]):
                    text = '"' + text.replace('"', '""') + '"'
                parts.append(text)
            lines.append(",".join(parts))
        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_json(self, data: Any, path: Path) -> None:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def write_reports(self) -> None:
        if not self.by_day:
            return
        rows = [["day", "total", "full", "partial", "empty"]]
        for day, vals in sorted(self.by_day.items()):
            rows.append([day, vals["total"], vals["full"], vals["partial"], vals["empty"]])
        self._write_csv(rows, self.reports_dir / "flight_load_stats_by_day.csv")
        self._write_json({day: vals for day, vals in sorted(self.by_day.items())}, self.reports_dir / "flight_load_stats.json")
