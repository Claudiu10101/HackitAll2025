import csv
from pathlib import Path
from typing import Dict, List


class CostLogger:
    def __init__(self, out_dir: str = "../reports/costs"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.daily_costs: List[Dict] = []

    def record_day(self, day: int, stock_cost: float, flight_cost: float):
        self.daily_costs.append(
            {"day": day, "stock_cost": stock_cost, "flight_cost": flight_cost}
        )

    def write_csv(self, filename: str = "costs_by_day.csv"):
        path = self.out_dir / filename
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["day", "stock_cost", "flight_cost"])
            writer.writeheader()
            for row in self.daily_costs:
                writer.writerow(row)
