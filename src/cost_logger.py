from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class CostLogger:
    """
    Tracks cumulative and per-round costs and writes simple reports/plots.
    totalCost is cumulative in API responses; we store deltas per round.
    """

    def __init__(self, reports_dir: str = "reports/costs") -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.reset()
        try:
            import matplotlib.pyplot as plt  # type: ignore

            self._plt = plt
        except Exception:
            self._plt = None

    def reset(self) -> None:
        self.last_total: Optional[float] = None
        self.cumulative: List[float] = []
        self.per_round: List[float] = []
        self.by_day: Dict[int, float] = {}

    def record(self, response: Dict[str, Any]) -> None:
        """Record cost info from one /play/round or /session/end response."""
        if "totalCost" not in response:
            return
        total = float(response["totalCost"])
        self.cumulative.append(total)
        delta = 0.0 if self.last_total is None else total - self.last_total
        self.per_round.append(delta)
        day = int(response.get("day", len(self.per_round) - 1))
        self.by_day[day] = self.by_day.get(day, 0.0) + delta
        self.last_total = total

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
        if not self.cumulative:
            return
        rounds = list(range(len(self.cumulative)))
        rows = [["round", "totalCost", "roundCost"]] + [
            [r, self.cumulative[r], self.per_round[r]] for r in rounds
        ]
        self._write_csv(rows, self.reports_dir / "costs_by_round.csv")

        day_rows = [["day", "cost_delta"]] + [[d, v] for d, v in sorted(self.by_day.items())]
        self._write_csv(day_rows, self.reports_dir / "costs_by_day.csv")

        summary = {
            "cumulative": self.cumulative,
            "perRound": self.per_round,
            "byDay": self.by_day,
        }
        self._write_json(summary, self.reports_dir / "costs_summary.json")

        if self._plt:
            self._plot(rounds)

    def _plot(self, rounds: List[int]) -> None:
        if not self._plt:
            return
        plt = self._plt
        plt.figure(figsize=(8, 4))
        plt.plot(rounds, self.per_round, marker="o", label="Cost per round")
        plt.plot(rounds, self.cumulative, marker=".", linestyle="--", label="Total cost")
        plt.xlabel("Round")
        plt.ylabel("Cost")
        plt.title("Cost evolution")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.reports_dir / "costs.png")
        plt.close()
