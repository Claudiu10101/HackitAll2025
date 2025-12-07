from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


class PenaltyLogger:
    """Collects penalties over time and writes reports/plots."""

    def __init__(self, reports_dir: str = "reports/penalties") -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.excluded_codes = {"END_OF_GAME_UNFULFILLED_FLIGHT_KITS"}
        self.reset()
        try:
            import matplotlib.pyplot as plt  # type: ignore

            self._plt = plt
        except Exception:
            self._plt = None

    def reset(self) -> None:
        """Clear accumulated penalties; useful when starting a new session."""
        self.by_day: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.totals: Dict[str, float] = defaultdict(float)
        self.excluded_totals: Dict[str, float] = defaultdict(float)
        self.total_cost: Optional[float] = None

    def record(self, response: Dict[str, Any]) -> None:
        """Record penalties from one /play/round or /session/end response."""
        penalties = response.get("penalties") or []
        for p in penalties:
            code = p.get("code", "UNKNOWN")
            if code in self.excluded_codes:
                self.excluded_totals[code] += float(p.get("penalty", 0.0))
                continue
            day = p.get("issuedDay", response.get("day", 0))
            amount = float(p.get("penalty", 0.0))
            self.by_day[day][code] += amount
            self.totals[code] += amount
        if "totalCost" in response:
            self.total_cost = response["totalCost"]

    def _write_json(self, data: Any, path: Path) -> None:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

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

    def write_reports(self) -> None:
        """Write CSV, JSON, and plot (if matplotlib available) to reports_dir."""
        if not self.by_day:
            return

        # CSV per day
        all_codes = sorted({code for day in self.by_day.values() for code in day.keys()})
        day_rows = [["day"] + all_codes]
        for day in sorted(self.by_day.keys()):
            row = [day] + [self.by_day[day].get(code, 0.0) for code in all_codes]
            day_rows.append(row)
        self._write_csv(day_rows, self.reports_dir / "penalties_by_day.csv")

        # Totals CSV
        total_rows = [["code", "total_penalty"]] + [[code, self.totals.get(code, 0.0)] for code in all_codes]
        self._write_csv(total_rows, self.reports_dir / "penalties_totals.csv")

        # JSON summary
        summary = {
            "totals": {code: self.totals[code] for code in all_codes},
            "byDay": {day: dict(values) for day, values in sorted(self.by_day.items())},
            "totalCost": self.total_cost,
            "excludedTotals": dict(self.excluded_totals),
            "adjustedTotalCost": (
                None
                if self.total_cost is None
                else self.total_cost - sum(self.excluded_totals.values())
            ),
        }
        self._write_json(summary, self.reports_dir / "penalties_summary.json")

        # Plot
        if self._plt:
            self._plot_lines(all_codes)

    def _plot_lines(self, codes: Iterable[str]) -> None:
        if not self._plt:
            return
        plt = self._plt
        days = sorted(self.by_day.keys())
        for code in codes:
            y = [self.by_day[day].get(code, 0.0) for day in days]
            plt.plot(days, y, marker="o", label=code)
        plt.xlabel("Day")
        plt.ylabel("Penalty amount")
        plt.title("Penalties by day and type")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(self.reports_dir / "penalties_by_day.png")
        plt.close()
